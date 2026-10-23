"""Repeated nested validation and source-excluded stale-hops transfer proxy."""
from __future__ import annotations

import json
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from common import SOLUTION, data_dir
from q3_data import load_q3_data
from q3_stale_optimized import (
    AUDIT_BOOTSTRAP_SEED, OUTER_SEEDS, SELECTION_BOOTSTRAP_SEED,
    TRANSFER_CV_SEED, VARIANCE_RATIO_MIN, candidate_specs, choose_candidate,
    inner_compare, make_pipeline, predict_1d, raw_tree_pipeline, scores,
)

OUT = SOLUTION / "outputs" / "q3_optimized" / "stale"
TARGET = "stale_hops"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    meta, x, y45, manifest = load_q3_data()
    y = y45[TARGET]
    features = x.columns.tolist()
    calibrated = manifest.loc[manifest.representation.eq("calibrated"), "feature"].tolist()
    metadata = pd.read_excel(data_dir() / "8.xlsx", sheet_name="Sensory thresholds")
    aging = [feature for feature in features
             if feature in set(metadata.loc[metadata.aging.notna(), "field_name_R"])]
    specs = candidate_specs()
    spec_map = {spec["name"]: spec for spec in specs}
    families = sorted({spec["family"] for spec in specs})
    write_json(OUT / "run_manifest.json", {
        "target": TARGET, "features": features, "calibrated_features": calibrated,
        "aging_features": aging, "aging_rule": "8.xlsx Sensory thresholds aging nonmissing; exact field match",
        "candidate_specs": specs, "outer_seeds": list(OUTER_SEEDS), "outer_folds": 5,
        "inner_folds": 3, "inner_objective": "OOF RMSE subject to prediction variance ratio >=0.05",
        "nonconstant_variance_ratio_min": VARIANCE_RATIO_MIN,
        "fallback": "lowest RMSE if no nonconstant candidate; flagged unqualified",
        "outer_outcomes_not_used_for_final_config": True,
        "transfer_cv_seed": TRANSFER_CV_SEED,
        "selection_bootstrap_seed": SELECTION_BOOTSTRAP_SEED,
        "audit_bootstrap_seed": AUDIT_BOOTSTRAP_SEED,
        "forbidden_predictors": ["other observed sensory targets", "attachment 7 comments", "identity keys", "style"],
    })
    pd.DataFrame({"feature": aging}).to_csv(OUT / "predeclared_aging_features.csv", index=False)
    all_inner, choices, fold_metrics, oof_rows, metrics_rows = [], [], [], [], []
    methods = ["mean", "old_multioutput_extra_trees", "fixed_single_extra_trees", "nested_selected"]
    methods += [f"nested_family_{family}" for family in families]
    for repeat, outer_seed in enumerate(OUTER_SEEDS, 1):
        predictions = {method: np.full(len(x), np.nan) for method in methods}
        fold_ids = np.zeros(len(x), dtype=int)
        for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=outer_seed).split(x), 1):
            inner_seed = outer_seed + 100 * fold
            rows, _ = inner_compare(x.iloc[train], y.iloc[train], specs, features, calibrated, aging, inner_seed)
            for row in rows:
                all_inner.append({"repeat": repeat, "outer_seed": outer_seed, "outer_fold": fold,
                                  "inner_seed": inner_seed, **row})
            selected = {"nested_selected": choose_candidate(rows)}
            selected.update({f"nested_family_{family}": choose_candidate(rows, family) for family in families})
            fitted_cache = {}
            for method, row in selected.items():
                candidate = row["candidate"]
                if candidate not in fitted_cache:
                    fitted_cache[candidate] = make_pipeline(
                        spec_map[candidate], features, calibrated, aging, inner_seed + 90).fit(x.iloc[train], y.iloc[train])
                predictions[method][test] = predict_1d(fitted_cache[candidate], x.iloc[test])
                choices.append({"repeat": repeat, "outer_seed": outer_seed, "outer_fold": fold,
                                "method": method, "selected_candidate": candidate,
                                "inner_oof_rmse": row["rmse"],
                                "inner_oof_variance_ratio": row["prediction_variance_ratio"],
                                "nonconstant_qualified": row["nonconstant_qualified"]})
            old = raw_tree_pipeline(inner_seed + 90).fit(x.iloc[train], y45.iloc[train])
            predictions["old_multioutput_extra_trees"][test] = old.predict(x.iloc[test])[:, y45.columns.get_loc(TARGET)]
            single = raw_tree_pipeline(inner_seed + 90).fit(x.iloc[train], y.iloc[train])
            predictions["fixed_single_extra_trees"][test] = predict_1d(single, x.iloc[test])
            predictions["mean"][test] = float(y.iloc[train].mean())
            fold_ids[test] = fold
            for method in methods:
                fold_metrics.append({"repeat": repeat, "outer_seed": outer_seed, "fold": fold,
                                     "method": method, **scores(y.iloc[test], predictions[method][test])})
            pd.DataFrame(all_inner).to_csv(OUT / "nested_inner_cv_results.csv", index=False)
            pd.DataFrame(choices).to_csv(OUT / "nested_selected_configs.csv", index=False)
            print(f"Stale nested repeat {repeat}/2 fold {fold}/5: {selected['nested_selected']['candidate']}; "
                  f"outer RMSE={scores(y.iloc[test], predictions['nested_selected'][test])['rmse']:.6f}", flush=True)
        for method, pred in predictions.items():
            if not np.isfinite(pred).all():
                raise AssertionError(f"Incomplete OOF: repeat={repeat}, {method}")
            metrics_rows.append({"scope": "per_seed", "repeat": repeat, "outer_seed": outer_seed,
                                 "method": method, **scores(y, pred)})
            oof_rows.append(pd.concat([
                meta, pd.DataFrame({"repeat": repeat, "outer_seed": outer_seed, "fold": fold_ids,
                                    "method": method, "observed": y, "prediction": pred})], axis=1))
    oof = pd.concat(oof_rows, ignore_index=True)
    for method in methods:
        rows = oof[oof.method.eq(method)]
        metrics_rows.append({"scope": "pooled_repeated_oof", "repeat": 0, "outer_seed": 0,
                             "method": method, **scores(rows.observed, rows.prediction)})
    metrics = pd.DataFrame(metrics_rows)
    oof.to_csv(OUT / "oof_predictions.csv", index=False)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(OUT / "metrics_by_fold.csv", index=False)
    summary_metrics = metrics[metrics.scope.eq("pooled_repeated_oof")].set_index("method")
    mean_rmse = float(summary_metrics.loc["mean", "rmse"])

    # Only these 239 rows participate in transfer configuration selection.
    excluded = np.flatnonzero(meta.tasting_category_fine.eq("Pils/Lager"))
    transfer_train = np.flatnonzero(~meta.tasting_category_fine.eq("Pils/Lager"))
    if (len(transfer_train), len(excluded)) != (239, 11):
        raise ValueError("Expected all 11 Pils/Lager sources held out")
    transfer_rows, transfer_inner_pred = inner_compare(
        x.iloc[transfer_train], y.iloc[transfer_train], specs, features, calibrated, aging, TRANSFER_CV_SEED)
    pd.DataFrame(transfer_rows).to_csv(OUT / "transfer_inner_cv_results.csv", index=False)
    selected = choose_candidate(transfer_rows)
    selected_spec = spec_map[selected["candidate"]]
    transfer_baseline = np.full(len(transfer_train), np.nan)
    for train_local, test_local in KFold(3, shuffle=True, random_state=TRANSFER_CV_SEED).split(transfer_train):
        transfer_baseline[test_local] = y.iloc[transfer_train[train_local]].mean()
    transfer_baseline_rmse = scores(y.iloc[transfer_train], transfer_baseline)["rmse"]
    qualified = bool(selected["nonconstant_qualified"] and selected["rmse"] < transfer_baseline_rmse)
    final = make_pipeline(selected_spec, features, calibrated, aging, TRANSFER_CV_SEED + 90).fit(
        x.iloc[transfer_train], y.iloc[transfer_train])
    artifact = {"pipeline": final, "features": features, "target": TARGET,
                "target_scale": "attachment_4_standardized", "selected_config": selected_spec,
                "qualified_for_proxy": qualified,
                "inner_oof_variance_ratio": selected["prediction_variance_ratio"],
                "inner_oof_rmse": selected["rmse"], "inner_mean_baseline_rmse": transfer_baseline_rmse,
                "train_global_indices": transfer_train, "excluded_global_indices": excluded,
                "excluded_style": "Pils/Lager", "selection_seed": TRANSFER_CV_SEED,
                "qualification_scope": "training_only_inner_CV; not holdout or counterfactual validation",
                "qualification_rule": "training-only inner OOF variance ratio >=0.05 and RMSE < inner mean baseline"}
    joblib.dump(artifact, OUT / "transfer_stale_model.joblib", compress=3)
    pd.concat([meta.iloc[transfer_train].reset_index(drop=True), pd.DataFrame({
        "observed": y.iloc[transfer_train].to_numpy(),
        "selected_inner_oof_prediction": transfer_inner_pred[selected["candidate"]],
        "mean_inner_oof_prediction": transfer_baseline})], axis=1).to_csv(
            OUT / "transfer_selected_inner_oof.csv", index=False)

    # Source holdout is evaluated only after configuration selection/fitting.
    heldout = pd.concat([meta.iloc[excluded].reset_index(drop=True), pd.DataFrame({
        "observed": y.iloc[excluded].to_numpy(), "prediction": predict_1d(final, x.iloc[excluded]),
        "mean_baseline_prediction": y.iloc[transfer_train].mean()})], axis=1)
    heldout.to_csv(OUT / "pils_style_holdout_predictions.csv", index=False)
    holdout_metrics = [{"model": "optimized_single_target", **scores(heldout.observed, heldout.prediction)},
                       {"model": "non_pils_mean", **scores(heldout.observed, heldout.mean_baseline_prediction)}]
    old_path = SOLUTION / "outputs" / "q3" / "sensory" / "transfer_proxy.joblib"
    if old_path.exists():
        old = joblib.load(old_path)
        if set(old["train_global_indices"]) & set(excluded):
            raise AssertionError("Old transfer proxy leaks Pils source rows")
        old_pred = old["pipeline"].predict(x.iloc[excluded])[:, old["targets"].index(TARGET)]
        holdout_metrics.append({"model": "old_multioutput_transfer_proxy", **scores(heldout.observed, old_pred)})
    pd.DataFrame(holdout_metrics).to_csv(OUT / "pils_style_holdout_metrics.csv", index=False)

    models, bootstrap_indices, bootstrap_seeds, estimator_seeds = [], [], [], []
    for b in range(60):
        seed = SELECTION_BOOTSTRAP_SEED + b if b < 40 else AUDIT_BOOTSTRAP_SEED + (b - 40)
        index = np.random.default_rng(seed).choice(transfer_train, size=len(transfer_train), replace=True)
        estimator_seed = seed + 10000
        model = make_pipeline(selected_spec, features, calibrated, aging, estimator_seed).fit(x.iloc[index], y.iloc[index])
        models.append(model)
        bootstrap_indices.append(index)
        bootstrap_seeds.append(seed)
        estimator_seeds.append(estimator_seed)
        if (b + 1) % 10 == 0:
            print(f"Stale optimized bootstrap {b + 1}/60", flush=True)
    ensemble = {"models": models, "bootstrap_indices": bootstrap_indices,
                "features": features, "target": TARGET, "target_scale": "attachment_4_standardized",
                "selected_config": selected_spec, "qualified_for_proxy": qualified,
                "qualification_scope": "training_only_inner_CV; not holdout or counterfactual validation",
                "inner_oof_variance_ratio": selected["prediction_variance_ratio"],
                "inner_oof_rmse": selected["rmse"],
                "train_global_indices": transfer_train, "excluded_global_indices": excluded,
                "selection_model_indices": list(range(40)), "audit_model_indices": list(range(40, 60)),
                "bootstrap_seeds": bootstrap_seeds, "estimator_seeds": estimator_seeds,
                "pairing": "same_model_candidate_minus_source", "paired_quantile": 0.95,
                "hyperparameter_uncertainty_included": False,
                "audit_protocol": "40 screening models; 20 separate bootstrap draws for one-time audit; no reselection",
                "interpretation": "model proxy bootstrap quantiles; not calibrated confidence bounds"}
    joblib.dump(ensemble, OUT / "bootstrap_ensemble.joblib", compress=3)
    np.savez_compressed(OUT / "bootstrap_indices.npz", indices=np.asarray(bootstrap_indices),
                        bootstrap_seeds=np.asarray(bootstrap_seeds), estimator_seeds=np.asarray(estimator_seeds))
    assert np.asarray(bootstrap_indices).shape == (60, 239)
    assert all(not set(index) & set(excluded) for index in bootstrap_indices)
    assert set(ensemble["selection_model_indices"]).isdisjoint(ensemble["audit_model_indices"])
    assert np.isfinite(predict_1d(final, x)).all()
    selected_metrics = summary_metrics.loc["nested_selected"].to_dict()
    summary = {
        "samples": len(x), "features": len(features), "target": TARGET,
        "outer_seeds": list(OUTER_SEEDS), "outer_folds": 5, "inner_folds": 3,
        "candidate_count": len(specs), "nonconstant_variance_ratio_min": VARIANCE_RATIO_MIN,
        "main_result": "nested selection algorithm; not post-hoc best outer family",
        "pooled_repeat_metrics": [{"method": method, **summary_metrics.loc[method].to_dict()} for method in methods],
        "nested_rmse_reduction_vs_mean": 1 - selected_metrics["rmse"] / mean_rmse,
        "transfer_selected_config": selected_spec, "transfer_qualified_for_proxy": qualified,
        "transfer_qualification_scope": "training_only_inner_CV; not holdout or counterfactual validation",
        "pils_holdout_improves_mean": holdout_metrics[0]["rmse"] < holdout_metrics[1]["rmse"],
        "transfer_inner_oof_metrics": selected,
        "transfer_inner_mean_baseline_rmse": transfer_baseline_rmse,
        "pils_holdout_metrics": holdout_metrics,
        "bootstrap": {"models": 60, "selection": 40, "audit": 20, "population": "239 non-Pils only",
                      "selected_config_fixed": True, "hyperparameter_uncertainty_included": False},
        "limitations": [
            "Only chemical features available to generated profiles are used; no other sensory labels or comments are predictors.",
            "The old multi-output comparator uses 45 labels only during its training, never as prediction features.",
            "Repeated OOF rows are correlated repeats of 250 samples, not 500 independent beers.",
            "The candidate grid and variance gate are fixed; family outer metrics do not select the transfer configuration.",
            "All transfer tuning, preprocessing and bootstrap sampling exclude all 11 Pils/Lager sources.",
            "Pils style holdout has 11 samples in one style and is not used to choose the model.",
            "Bootstrap fixes the training-only selected configuration and does not include hyperparameter-selection uncertainty.",
            "Disjoint bootstrap draws share the same observed training data; the audit is not independent experimental validation.",
            "Passing paired model-proxy differences cannot guarantee improved real stale-hops taste or causal chemical effects.",
            "Attachment-standardized target values cannot be interpreted as original 0-6 sensory scores.",
        ],
        "verification": {"bootstrap_source_exclusion": True, "selection_audit_disjoint": True,
                         "final_predictions_finite": True, "bootstrap_indices_shape": [60, 239]},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": perf_counter() - started,
    }
    # Pandas integer dtypes in to_dict are normal Python values here.
    write_json(OUT / "summary.json", summary)
    print(summary_metrics[["rmse", "r2", "spearman", "prediction_variance_ratio"]].to_string(), flush=True)
    print("Transfer chosen:", selected_spec, "qualified:", qualified, flush=True)
    print("Pils holdout:", json.dumps(holdout_metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
