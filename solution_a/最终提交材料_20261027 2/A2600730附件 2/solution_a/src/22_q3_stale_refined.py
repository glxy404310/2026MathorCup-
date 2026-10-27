"""Grouped selection, auxiliary supervision and stability-first stale proxy."""
from __future__ import annotations

import json
import platform
import warnings
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.spatial.distance import cdist
from scipy.stats import ConstantInputWarning
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from threadpoolctl import threadpool_limits

from common import SOLUTION, data_dir
from q3_data import load_q3_data
from q3_stale_optimized import scores as previous_scores
from q3_stale_refined import (
    AUDIT_BOOTSTRAP_SEED, BIAS_PENALTY, FINAL_SEED, FixedChemicalTransform,
    NESTED_SEED, RefinedStaleModel, SELECTION_BOOTSTRAP_SEED,
    STABILITY_DRAWS, STABILITY_SEED, TARGET, VARIANCE_RATIO_MIN, refined_specs,
)

OUT = SOLUTION / "outputs" / "q3_refined" / "stale"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")


def scores(y, pred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = previous_scores(y, pred)
    if result["spearman"] is not None and not np.isfinite(result["spearman"]):
        result["spearman"] = None
    return result


def model(spec, context, seed):
    return RefinedStaleModel(context["features"], context["calibrated"],
                             context["aging"], context["target_names"], spec, seed)


def uses_auxiliary(spec):
    return spec["family"] in ("aux", "aux_kernel") or (
        spec["family"] == "ensemble" and any(uses_auxiliary(member) for member in spec["members"]))


def save_candidate_models(fitted_candidates, specs, context, final_table,
                          decision, stability, train, source):
    """Keep every selectable candidate, including failed development paths."""
    destination = OUT/"candidate_models"
    destination.mkdir(parents=True, exist_ok=True)
    table = final_table.set_index("method")
    stability_table = stability.set_index("method")
    manifest_rows = []
    for spec in specs:
        if not spec["selectable"]:
            continue
        name = spec["name"]
        row = table.loc[name]
        candidate_stability = stability_table.loc[name].to_dict() if name in stability_table.index else None
        internal_qualification = None if candidate_stability is None else bool(
            row.eligible_for_one_se and candidate_stability["pair_delta_signal_rms"] > 1e-4
            and candidate_stability["response_noise_ratio"] <= 1)
        selected = name == decision["selected_method"]
        auxiliary = uses_auxiliary(spec)
        artifact = {"pipeline": fitted_candidates[name], "features": context["features"],
            "target": TARGET, "target_scale": "attachment_4_standardized", "config": spec,
            "selected_config": spec, "train_global_indices": train, "excluded_global_indices": source,
            "excluded_style": "Pils/Lager", "not_selected": not selected,
            "selection_status": "selected_reference_copy" if selected else "not_selected",
            "development_diagnostic_only": True, "selected_in_frozen_protocol": selected,
            "eligible_for_one_se": bool(row.eligible_for_one_se), "within_one_se": bool(row.within_one_se),
            "loso_metrics": row.to_dict(), "candidate_stability": candidate_stability,
            "candidate_internal_qualification": internal_qualification,
            "qualified_for_proxy": bool(selected and decision["qualified_for_proxy"]),
            "qualification_scope": "candidate archive; unselected artifacts are development diagnostics, not deployment models",
            "uses_auxiliary_supervision": auxiliary,
            "training_auxiliary_targets": [name for name in context["target_names"] if name != TARGET] if auxiliary else [],
            "pils_status": "repeatedly viewed development diagnostic", "pils_used_for_selection": False,
            "final_seed": FINAL_SEED}
        joblib.dump(artifact, destination/f"{name}.joblib", compress=3)
        manifest_rows.append({"method": name, "file": f"candidate_models/{name}.joblib",
            "selected_in_frozen_protocol": selected, "eligible_for_one_se": bool(row.eligible_for_one_se),
            "within_one_se": bool(row.within_one_se), "development_diagnostic_only": True,
            "macro_style_rmse": float(row.macro_style_rmse), "pooled_rmse": float(row.rmse)})
    pd.DataFrame(manifest_rows).to_csv(OUT/"candidate_model_manifest.csv", index=False, lineterminator="\n")


def evaluate_loso(indices, specs, context, seed, scope):
    meta, x, y45 = context["meta"], context["x"], context["y45"]
    groups = meta.iloc[indices].tasting_category_fine.to_numpy()
    truth = y45.iloc[indices][TARGET].to_numpy()
    predictions = {spec["name"]: np.full(len(indices), np.nan) for spec in specs}
    folds, thresholds = np.zeros(len(indices), dtype=int), np.full(len(indices), np.nan)
    style_rows, audit_rows = [], []
    for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(indices, truth, groups), 1):
        train_global, test_global = indices[tr], indices[te]
        style = str(groups[te[0]])
        assert not meta.iloc[train_global].tasting_category_fine.eq(style).any()
        assert not meta.iloc[train_global].tasting_category_fine.eq("Pils/Lager").any()
        for spec in specs:
            if spec["family"] == "ensemble":
                values = np.mean([predictions[member["name"]][te] for member in spec["members"]], axis=0)
            else:
                fitted = model(spec, context, seed+fold).fit(x.iloc[train_global], y45.iloc[train_global],
                    meta.iloc[train_global].tasting_category_fine)
                values = fitted.predict(x.iloc[test_global])
            predictions[spec["name"]][te] = values
            style_rows.append({"scope": scope, "method": spec["name"], "heldout_style": style,
                               "fold": fold, "n": len(te), **scores(truth[te], values),
                               "bias": float(np.mean(values-truth[te]))})
        folds[te] = fold
        thresholds[te] = np.quantile(y45.iloc[train_global][TARGET], .9)
        audit_rows.extend({"scope": scope, "fold": fold, "heldout_style": style,
                           "role": role, "global_index": int(index)}
                          for role, sample in [("train", train_global), ("test", test_global)] for index in sample)
    per_style = pd.DataFrame(style_rows)
    summaries, long_rows = [], []
    tail = truth >= thresholds
    for spec in specs:
        pred = predictions[spec["name"]]
        assert np.isfinite(pred).all()
        subset = per_style[per_style.method.eq(spec["name"])]
        losses = subset.rmse.to_numpy()+BIAS_PENALTY*subset.bias.abs().to_numpy()
        metric = scores(truth, pred)
        summaries.append({"scope": scope, "method": spec["name"], "selectable": spec["selectable"],
            "priority": spec["priority"], "styles": len(subset),
            "macro_style_rmse": float(subset.rmse.mean()),
            "macro_style_abs_bias": float(subset.bias.abs().mean()),
            "selection_score": float(losses.mean()),
            "style_score_se": float(losses.std(ddof=1)/np.sqrt(len(losses))),
            "tail_n": int(tail.sum()), "tail_rmse": float(np.sqrt(np.mean((pred[tail]-truth[tail])**2))),
            **metric})
        long_rows.append(pd.concat([meta.iloc[indices].reset_index(drop=True), pd.DataFrame({
            "global_index": indices, "scope": scope, "method": spec["name"], "observed": truth,
            "prediction": pred, "heldout_style": groups, "fold": folds,
            "tail_threshold_training_q90": thresholds, "is_tail_eval": tail})], axis=1))
    return pd.DataFrame(summaries), per_style, pd.concat(long_rows, ignore_index=True), pd.DataFrame(audit_rows)


def feature_pairs(indices, context):
    transform = FixedChemicalTransform(context["features"], context["calibrated"], context["aging"]).fit(context["x"].iloc[indices])
    z = transform.transform(context["x"].iloc[indices])
    distances = cdist(z, z)
    np.fill_diagonal(distances, np.inf)
    # Third-nearest distinct training row; the fixed rule never sees target y.
    nearest_local = np.argsort(distances, axis=1)[:, 2]
    return indices[nearest_local]


def stability_compare(indices, candidates, context, seed, scope):
    pairs = feature_pairs(indices, context)
    records, pair_rows = [], []
    for spec in candidates:
        delta = []
        for b in range(STABILITY_DRAWS):
            index = np.random.default_rng(seed+b).choice(indices, size=len(indices), replace=True)
            fitted = model(spec, context, seed+10000+b).fit(context["x"].iloc[index], context["y45"].iloc[index],
                context["meta"].iloc[index].tasting_category_fine)
            delta.append(fitted.predict(context["x"].iloc[indices])-fitted.predict(context["x"].iloc[pairs]))
        delta = np.asarray(delta)
        noise = float(np.sqrt(np.mean(np.var(delta, axis=0, ddof=1))))
        signal = float(np.sqrt(np.mean(delta.mean(axis=0)**2)))
        ratio = noise/max(signal, 1e-8)
        records.append({"scope": scope, "method": spec["name"], "stability_draws": STABILITY_DRAWS,
                        "pair_delta_noise_rms": noise, "pair_delta_signal_rms": signal,
                        "response_noise_ratio": ratio, "priority": spec["priority"]})
    pair_rows = [{"scope": scope, "global_index": int(index), "paired_global_index": int(pair)}
                 for index, pair in zip(indices, pairs)]
    return pd.DataFrame(records), pd.DataFrame(pair_rows)


def select_stable(indices, summary, specs, context, seed, scope):
    table = summary.copy()
    mean = table[table.method.eq("mean_global")].iloc[0]
    eligible = table.selectable & (table.prediction_variance_ratio >= VARIANCE_RATIO_MIN)
    eligible &= (table.macro_style_rmse < mean.macro_style_rmse) & (table.rmse < mean.rmse)
    table["eligible_for_one_se"] = eligible
    candidates = table[eligible]
    fallback = candidates.empty
    if fallback:
        best = table[table.selectable].sort_values(["selection_score", "method"]).iloc[0]
        one_se = table[table.method.eq(best.method)]
    else:
        best = candidates.sort_values(["selection_score", "method"]).iloc[0]
        one_se = candidates[candidates.selection_score <= best.selection_score+best.style_score_se]
    table["within_one_se"] = table.method.isin(one_se.method)
    spec_map = {spec["name"]: spec for spec in specs}
    stability, pairs = stability_compare(indices, [spec_map[name] for name in one_se.method], context, seed, scope)
    chosen = stability.sort_values(["response_noise_ratio", "priority", "method"]).iloc[0]
    selected_method = str(chosen.method)
    chosen_metrics = table[table.method.eq(selected_method)].iloc[0]
    qualified = bool(not fallback and chosen.pair_delta_signal_rms > 1e-4 and chosen.response_noise_ratio <= 1.0)
    decision = {"selected_method": selected_method, "config": spec_map[selected_method],
        "qualified_for_proxy": qualified, "eligibility_fallback": bool(fallback),
        "lowest_score_method": str(best.method), "lowest_selection_score": float(best.selection_score),
        "one_se_width": float(best.style_score_se),
        "one_se_threshold": float(best.selection_score+best.style_score_se),
        "selected_loso_metrics": chosen_metrics.to_dict(), "selected_stability": chosen.to_dict(),
        "qualification_scope": "training-only grouped selection and response stability; not Pils or causal validation"}
    return decision, table, stability, pairs


def main():
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    meta, x, y45, manifest = load_q3_data()
    features = x.columns.tolist()
    calibrated = manifest.loc[manifest.representation.eq("calibrated"), "feature"].tolist()
    source = np.flatnonzero(meta.tasting_category_fine.eq("Pils/Lager"))
    train = np.flatnonzero(~meta.tasting_category_fine.eq("Pils/Lager"))
    thresholds = pd.read_excel(data_dir()/"8.xlsx", sheet_name="Sensory thresholds")
    aging_set = set(thresholds.loc[thresholds.aging.notna(), "field_name_R"])
    aging = [feature for feature in features if feature in aging_set]
    context = {"meta": meta, "x": x, "y45": y45, "features": features,
               "calibrated": calibrated, "aging": aging, "target_names": y45.columns.tolist()}
    specs = refined_specs()
    write_json(OUT/"run_manifest.json", {
        "candidate_specs": specs, "features": features, "aging_features": aging,
        "training_indices": train.tolist(), "excluded_indices": source.tolist(),
        "auxiliary_target_names": [name for name in y45 if name != TARGET],
        "auxiliary_representation": "weighted Ridge alpha100 to 44 standardized auxiliary outcomes; right singular directions rank4/8; stale excluded",
        "fixed_missing_representation": "always append all selected missing flags at weight0.2; never variance-scale flags",
        "tail_training_weight": "multiply top20% training-only stale outcomes by2, then normalize mean weight1",
        "score": "macro(style RMSE)+0.25*macro(abs(style mean error))",
        "one_se_eligibility": "selectable and OOF variance ratio>=0.05 and macro/pooled RMSE<global mean baseline",
        "one_se_rule": "score<=best score+SE across held-out styles; choose lowest response-noise/signal ratio from8 common row bootstraps; complexity priority breaks exact ties",
        "qualification": "eligibility and response signal RMS>1e-4 and response noise/signal<=1",
        "chemical_pair_rule": "third-nearest distinct non-Pils training row in fixed aging transform",
        "final_validation": "complete non-Pils21 LOSO", "nested_validation": "4 outer shuffled GroupKFold; complete inner LOSO; selector stability computed on outer training only",
        "stability_seed": STABILITY_SEED, "stability_draws": STABILITY_DRAWS,
        "nested_seed": NESTED_SEED, "final_seed": FINAL_SEED,
        "selection_bootstrap_seed": SELECTION_BOOTSTRAP_SEED, "audit_bootstrap_seed": AUDIT_BOOTSTRAP_SEED,
        "pils_labels_used_for_selection_or_intercept": False,
        "pils_status": "repeatedly viewed development diagnostic, not a new test",
        "all_candidates_retained": True, "no_beer81_exclusion": True,
    })
    # The full grouped candidate comparison is fixed before reading Pils errors.
    full, by_style, pred, audit = evaluate_loso(train, specs, context, FINAL_SEED, "full_loso")
    full.to_csv(OUT/"loso_metrics_overall_raw.csv", index=False, lineterminator="\n")
    by_style.to_csv(OUT/"loso_metrics_by_style.csv", index=False, lineterminator="\n")
    pred.to_csv(OUT/"loso_predictions.csv", index=False, lineterminator="\n")
    audit.to_csv(OUT/"loso_split_audit.csv", index=False, lineterminator="\n")
    decision, final_table, stability, pairs = select_stable(train, full, specs, context, STABILITY_SEED, "full_loso")
    final_table.to_csv(OUT/"loso_metrics_overall.csv", index=False, lineterminator="\n")
    stability.to_csv(OUT/"selection_stability.csv", index=False, lineterminator="\n")
    pairs.to_csv(OUT/"selection_chemical_pairs.csv", index=False, lineterminator="\n")
    write_json(OUT/"selection_result.json", decision)
    print("Refined full LOSO selected:", decision["selected_method"], "qualified", decision["qualified_for_proxy"], flush=True)

    # Evaluate the entire predeclared selector in an additional grouped layer.
    outer_rows, nested_choices, nested_inner_rows, nested_stability_rows = [], [], [], []
    nested_inner_audits, nested_outer_audits = [], []
    selector_specs = [spec for spec in specs if spec["family"] not in ("old_tree", "single_tree")]
    groups = meta.iloc[train].tasting_category_fine.to_numpy()
    for fold, (tr, te) in enumerate(GroupKFold(4, shuffle=True, random_state=NESTED_SEED).split(train, groups=groups), 1):
        tr_global, te_global = train[tr], train[te]
        summary, _, _, inner_audit = evaluate_loso(tr_global, selector_specs, context, NESTED_SEED+fold*1000, f"nested_{fold}")
        inner_audit["outer_fold"] = fold
        nested_inner_audits.append(inner_audit)
        nested_outer_audits.extend({"outer_fold":fold,"role":role,"global_index":int(index),
            "style":str(meta.iloc[index].tasting_category_fine)}
            for role,sample in [("train",tr_global),("test",te_global)] for index in sample)
        selected, inner_table, stab, _ = select_stable(tr_global, summary, selector_specs, context,
                                                     STABILITY_SEED+fold*1000, f"nested_{fold}")
        inner_table["outer_fold"] = fold
        stab["outer_fold"] = fold
        nested_inner_rows.append(inner_table)
        nested_stability_rows.append(stab)
        nested_choices.append({"outer_fold": fold, "selected_method": selected["selected_method"],
                               "qualified_for_proxy": selected["qualified_for_proxy"],
                               "heldout_styles": ";".join(sorted(meta.iloc[te_global].tasting_category_fine.unique())),
                               "train_n": len(tr_global), "test_n": len(te_global)})
        for method_name in ("nested_stability_rule", "mean_global", "old_multioutput_tree", "fixed_single_tree"):
            spec = selected["config"] if method_name == "nested_stability_rule" else next(spec for spec in specs if spec["name"] == method_name)
            fitted = model(spec, context, NESTED_SEED+fold).fit(x.iloc[tr_global], y45.iloc[tr_global], meta.iloc[tr_global].tasting_category_fine)
            values = fitted.predict(x.iloc[te_global])
            outer_rows.append(pd.concat([meta.iloc[te_global].reset_index(drop=True), pd.DataFrame({
                "global_index": te_global, "outer_fold": fold, "method": method_name,
                "observed": y45.iloc[te_global][TARGET].to_numpy(), "prediction": values,
                "selected_method": selected["selected_method"] if method_name == "nested_stability_rule" else method_name,
                "tail_threshold_training_q90": np.quantile(y45.iloc[tr_global][TARGET], .9)})], axis=1))
        print(f"Refined nested group fold {fold}/4: {selected['selected_method']}", flush=True)
    nested = pd.concat(outer_rows, ignore_index=True)
    nested.to_csv(OUT/"nested_predictions.csv", index=False, lineterminator="\n")
    pd.DataFrame(nested_choices).to_csv(OUT/"nested_choices.csv", index=False, lineterminator="\n")
    pd.concat(nested_inner_rows, ignore_index=True).to_csv(OUT/"nested_inner_metrics.csv", index=False, lineterminator="\n")
    pd.concat(nested_stability_rows, ignore_index=True).to_csv(OUT/"nested_selection_stability.csv", index=False, lineterminator="\n")
    pd.concat(nested_inner_audits, ignore_index=True).to_csv(OUT/"nested_inner_split_audit.csv", index=False, lineterminator="\n")
    pd.DataFrame(nested_outer_audits).to_csv(OUT/"nested_outer_split_audit.csv", index=False, lineterminator="\n")
    nested_metrics, nested_style_rows = [], []
    for name, group in nested.groupby("method"):
        style_metrics = []
        for style, z in group.groupby("tasting_category_fine"):
            value = {"method": name, "heldout_style": style, "n": len(z), **scores(z.observed, z.prediction)}
            nested_style_rows.append(value)
            style_metrics.append(value)
        tail = group.observed >= group.tail_threshold_training_q90
        nested_metrics.append({"method": name, "macro_style_rmse": float(np.mean([z["rmse"] for z in style_metrics])),
            "macro_style_abs_bias": float(np.mean([abs(z["mean_bias"]) for z in style_metrics])),
            "tail_n": int(tail.sum()), "tail_rmse": float(np.sqrt(np.mean((group.loc[tail,"prediction"]-group.loc[tail,"observed"])**2))),
            **scores(group.observed, group.prediction)})
    pd.DataFrame(nested_metrics).to_csv(OUT/"nested_metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(nested_style_rows).to_csv(OUT/"nested_metrics_by_style.csv", index=False, lineterminator="\n")

    selected_spec = decision["config"]
    selected_metrics = decision["selected_loso_metrics"]
    selected_uses_auxiliary = uses_auxiliary(selected_spec)
    metadata = {"features": features, "target": TARGET, "target_scale": "attachment_4_standardized",
        "config": selected_spec, "selected_config": selected_spec, "selected_method": decision["selected_method"],
        "qualified_for_proxy": decision["qualified_for_proxy"], "qualification_scope": decision["qualification_scope"],
        "inner_oof_variance_ratio": selected_metrics["prediction_variance_ratio"],
        "inner_oof_rmse": selected_metrics["rmse"], "loso_macro_style_rmse": selected_metrics["macro_style_rmse"],
        "loso_macro_style_abs_bias": selected_metrics["macro_style_abs_bias"],
        "train_global_indices": train, "excluded_global_indices": source, "excluded_style": "Pils/Lager",
        "uses_auxiliary_supervision": selected_uses_auxiliary,
        "training_auxiliary_targets": [name for name in y45 if name != TARGET] if selected_uses_auxiliary else [],
        "candidate_protocol_auxiliary_targets": [name for name in y45 if name != TARGET],
        "prediction_inputs": "chemical features only", "pils_used_for_selection": False}
    fitted_candidates = {spec["name"]: model(spec, context, FINAL_SEED).fit(x.iloc[train], y45.iloc[train], meta.iloc[train].tasting_category_fine) for spec in specs}
    save_candidate_models(fitted_candidates, specs, context, final_table, decision, stability, train, source)
    final = fitted_candidates[decision["selected_method"]]
    joblib.dump({"pipeline": final, **metadata}, OUT/"transfer_stale_model.joblib", compress=3)
    # Development diagnostics only: every model is already fixed above.
    diagnostic_rows, diagnostic_metrics = [], []
    for spec in specs:
        values = fitted_candidates[spec["name"]].predict(x.iloc[source])
        diagnostic_rows.append(pd.concat([meta.iloc[source].reset_index(drop=True), pd.DataFrame({
            "global_index": source, "method": spec["name"], "observed": y45.iloc[source][TARGET].to_numpy(),
            "prediction": values, "selected_without_pils_labels": spec["name"] == decision["selected_method"]})], axis=1))
        diagnostic_metrics.append({"method": spec["name"], "selected_without_pils_labels": spec["name"] == decision["selected_method"],
                                   "n": len(source), **scores(y45.iloc[source][TARGET], values)})
    pd.concat(diagnostic_rows, ignore_index=True).to_csv(OUT/"pils_diagnostic_predictions.csv", index=False, lineterminator="\n")
    pd.DataFrame(diagnostic_metrics).to_csv(OUT/"pils_diagnostic_metrics.csv", index=False, lineterminator="\n")

    models, indices, seeds = [], [], []
    for b in range(60):
        seed = SELECTION_BOOTSTRAP_SEED+b if b < 40 else AUDIT_BOOTSTRAP_SEED+b-40
        index = np.random.default_rng(seed).choice(train, size=len(train), replace=True)
        fitted = model(selected_spec, context, seed+10000).fit(x.iloc[index], y45.iloc[index], meta.iloc[index].tasting_category_fine)
        models.append(fitted)
        indices.append(index)
        seeds.append(seed)
    ensemble = {"models": models, "bootstrap_indices": indices, **metadata,
        "selection_model_indices": list(range(40)), "audit_model_indices": list(range(40,60)),
        "bootstrap_seeds": seeds, "estimator_seeds": [seed+10000 for seed in seeds],
        "pairing": "same_model_candidate_minus_source", "paired_quantile": .95,
        "hyperparameter_uncertainty_included": False,
        "audit_protocol": "40 selection;20 once-only audit;no reselection", "fixed_indicator_dimensions": True}
    joblib.dump(ensemble, OUT/"bootstrap_ensemble.joblib", compress=3)
    np.savez_compressed(OUT/"bootstrap_indices.npz", indices=np.asarray(indices), bootstrap_seeds=np.asarray(seeds))

    # Same chemical pairs compare stability with the prior local architecture.
    chemical_pairs = feature_pairs(train, context)
    comparisons = []
    for label, committee in [("refined_selected", models), ("domain_local_previous", joblib.load(SOLUTION/"outputs/q3_domain/stale/bootstrap_ensemble.joblib")["models"])]:
        delta = np.asarray([m.predict(x.iloc[train])-m.predict(x.iloc[chemical_pairs]) for m in committee])
        noise = float(np.sqrt(np.mean(np.var(delta, axis=0, ddof=1))))
        signal = float(np.sqrt(np.mean(delta.mean(axis=0)**2)))
        comparisons.append({"method": label, "models": len(committee), "pair_delta_noise_rms": noise,
                            "pair_delta_signal_rms": signal, "response_noise_ratio": noise/max(signal,1e-8),
                            "status": "descriptive fixed-pair architecture comparison; not used for selection"})
    pd.DataFrame(comparisons).to_csv(OUT/"bootstrap_response_comparison.csv", index=False, lineterminator="\n")
    assert np.asarray(indices).shape == (60,239)
    assert all(not set(index)&set(source) for index in indices)
    assert all(np.isfinite(m.predict(x)).all() for m in models)
    summary = {
        "selected_method": decision["selected_method"], "config": selected_spec,
        "qualified_for_proxy": decision["qualified_for_proxy"], "selection_result": decision,
        "training_samples": len(train), "pils_sources": len(source), "loso_styles":21,"candidate_count":len(specs),
        "mean_baseline_method":"mean_global","old_multioutput_baseline_method":"old_multioutput_tree","old_single_tree_baseline_method":"fixed_single_tree",
        "selected_loso_metrics": selected_metrics, "nested_metrics":nested_metrics,
        "all_pils_diagnostic_metrics":diagnostic_metrics,
        "selected_pils_diagnostic_metrics":next(row for row in diagnostic_metrics if row["method"]==decision["selected_method"]),
        "pils_status":"repeatedly viewed development diagnostic; never used to select or correct intercept",
        "bootstrap_response_comparison":comparisons,
        "bootstrap":{"models":60,"selection":40,"audit":20,"source_excluded":True,"configuration_fixed":True},
        "limitations":[
            "Auxiliary sensory values are training outcomes only; neither stale nor any measured sensory value is an inference feature.",
            "Every auxiliary fit, fixed indicator transform, tail threshold, style weight and stability draw excludes its held-out group.",
            "Full LOSO is used to fit the final selection rule; selected scores are optimistic. Four outer grouped folds assess that fixed rule internally.",
            "The same dataset has guided prior exploration; grouped resampling is not new external validation and Pils diagnostics are repeatedly viewed.",
            "Tail weighting uses only training quantiles and never the beer81 outcome, but can increase average bias.",
            "The one-SE rule uses a descriptive across-style SE; observed styles are not independent randomized domains.",
            "Model response stability does not establish accuracy on synthetic interventions or genuine stale-hops taste.",
            "60 bootstrap models fix the chosen configuration and omit algorithm-selection and new-domain uncertainty.",
            "The 20 audit models cannot be used to reselect a proposal; model quantiles are not calibrated confidence limits.",
            "The provided standardized target is retained; no original 0-6 score is inferred."],
        "verification":{"source_exclusion":True,"bootstrap_indices_shape":[60,239],"all_model_predictions_finite":True},
        "versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__},
        "elapsed_seconds":perf_counter()-started}
    write_json(OUT/"summary.json",summary)
    print(final_table[["method","macro_style_rmse","selection_score","prediction_variance_ratio","within_one_se"]].to_string(index=False),flush=True)
    print("Pils selected development diagnostic:",json.dumps(summary["selected_pils_diagnostic_metrics"],ensure_ascii=False),flush=True)
    print("Bootstrap response:",comparisons,flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
