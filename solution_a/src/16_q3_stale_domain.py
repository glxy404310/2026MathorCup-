"""Non-Pils leave-one-style-out selection and repeatedly viewed Pils diagnosis."""
from __future__ import annotations

import json
import platform
import warnings
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import ConstantInputWarning
from scipy.spatial.distance import cdist
from sklearn.model_selection import LeaveOneGroupOut
from threadpoolctl import threadpool_limits

from common import SOLUTION, data_dir
from q3_data import load_q3_data
from q3_stale_optimized import predict_1d, scores as regression_scores
from q3_stale_domain import (
    AUDIT_BOOTSTRAP_SEED, BIAS_PENALTY, FINAL_SEED, LOSO_SEED,
    SELECTION_BOOTSTRAP_SEED, VARIANCE_RATIO_MIN, domain_candidate_specs,
    make_domain_pipeline, training_fit_params,
)

OUT = SOLUTION / "outputs" / "q3_domain" / "stale"
TARGET = "stale_hops"


def scores(y, pred):
    # A few tiny styles have constant labels: correlation is undefined, not 0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = regression_scores(y, pred)
    if result["spearman"] is not None and not np.isfinite(result["spearman"]):
        result["spearman"] = None
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def predict_target(model, x, spec, target_position):
    if spec["family"] == "old_multioutput":
        return np.asarray(model.predict(x))[:, target_position]
    return predict_1d(model, x)


def write_local_neighborhood_diagnostics(final_model, models, indices, train_indices,
                                       source_indices, meta, x):
    """Feature-only bootstrap neighbourhood checks, never used for selection."""
    xs = x.iloc[source_indices]
    final_regressor = final_model.named_steps["model"]
    k = final_regressor.n_neighbors
    final_distances = cdist(final_model[:-1].transform(xs), final_regressor.x_)
    final_neighbors = [set(train_indices[np.argpartition(row, k-1)[:k]]) for row in final_distances]
    rows = []
    for b, (model, index) in enumerate(zip(models, indices)):
        regressor = model.named_steps["model"]
        distances = cdist(model[:-1].transform(xs), regressor.x_)
        prediction = predict_1d(model, xs)
        for j, (source_index, row) in enumerate(zip(source_indices, distances)):
            near = np.argpartition(row, k-1)[:k]
            neighbors = set(index[near])
            weights = regressor.sample_weight_[near] / np.maximum(row[near], 1e-12)
            # Repeated resample positions are not independent source beers.
            # Keep the original position-level Kish count and also aggregate
            # all weights belonging to the same original beer before Kish.
            _, beer_inverse = np.unique(index[near], return_inverse=True)
            weights_by_beer = np.bincount(beer_inverse, weights=weights)
            effective_positions = float(weights.sum()**2/np.sum(weights**2))
            effective_beers = float(weights_by_beer.sum()**2/np.sum(weights_by_beer**2))
            if effective_beers > len(neighbors)+1e-10:
                raise AssertionError("Beer-level effective count exceeds unique neighbours")
            rows.append({"beer": int(meta.iloc[source_index].beer), "model_index": b,
                "role": "selection" if b < 40 else "audit", "prediction": prediction[j],
                "nearest_unique_beers": len(neighbors),
                "neighbor_jaccard_vs_final": len(neighbors & final_neighbors[j])/len(neighbors | final_neighbors[j]),
                "effective_resampled_position_count": effective_positions,
                "effective_unique_beer_count": effective_beers,
                "imputer_indicator_count": len(model.named_steps["imputer"].indicator_.features_)})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "neighborhood_bootstrap_diagnostics.csv", index=False)
    frame.groupby("beer").agg(
        bootstrap_prediction_sd=("prediction", "std"),
        median_neighbor_jaccard=("neighbor_jaccard_vs_final", "median"),
        mean_unique_neighbors=("nearest_unique_beers", "mean"),
        median_effective_resampled_position_count=("effective_resampled_position_count", "median"),
        median_effective_unique_beer_count=("effective_unique_beer_count", "median"),
    ).to_csv(OUT / "neighborhood_bootstrap_summary.csv")


def main():
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    meta, x, y45, manifest = load_q3_data()
    y, target_position = y45[TARGET], y45.columns.get_loc(TARGET)
    source_indices = np.flatnonzero(meta.tasting_category_fine.eq("Pils/Lager"))
    train_indices = np.flatnonzero(~meta.tasting_category_fine.eq("Pils/Lager"))
    assert len(source_indices) == 11 and len(train_indices) == 239
    xtr, ytr = x.iloc[train_indices], y.iloc[train_indices]
    groups = meta.iloc[train_indices].tasting_category_fine.to_numpy()
    features = x.columns.tolist()
    calibrated = manifest.loc[manifest.representation.eq("calibrated"), "feature"].tolist()
    threshold_metadata = pd.read_excel(data_dir() / "8.xlsx", sheet_name="Sensory thresholds")
    aging_set = set(threshold_metadata.loc[threshold_metadata.aging.notna(), "field_name_R"])
    aging = [feature for feature in features if feature in aging_set]
    specs = domain_candidate_specs()
    spec_map = {spec["name"]: spec for spec in specs}
    run_manifest = {
        "candidate_specs": specs, "target": TARGET, "features": features, "aging_features": aging,
        "population": "239 non-Pils samples", "heldout_groups": sorted(np.unique(groups).tolist()),
        "validation": "complete leave-one-style-out; 21 folds; every preprocessing operation fitted per training fold",
        "selection_rule": "minimum macro(style RMSE) + 0.25*macro(abs(style mean prediction error)) among selectable nonconstant candidates",
        "bias_penalty": BIAS_PENALTY, "variance_ratio_min": VARIANCE_RATIO_MIN,
        "qualification_rule": "LOSO prediction variance ratio >=0.05 and macro RMSE < better of global/style-balanced mean baselines",
        "fallback": "if every selectable candidate fails nonconstant gate, retain minimum score but mark unqualified",
        "loso_seed": LOSO_SEED, "final_seed": FINAL_SEED,
        "selection_bootstrap_seed": SELECTION_BOOTSTRAP_SEED, "audit_bootstrap_seed": AUDIT_BOOTSTRAP_SEED,
        "pils_labels_used_in_training_selection_or_intercept": False,
        "pils_status": "previously repeatedly viewed diagnostic set; not a new test set",
        "forbidden_prediction_features": ["style", "identity", "other observed sensory targets", "attachment 7 comments"],
    }
    # Freeze all candidates/rules before starting the new grouped evaluation.
    write_json(OUT / "run_manifest.json", run_manifest)
    pd.DataFrame({"feature": aging}).to_csv(OUT / "predeclared_aging_features.csv", index=False)
    predictions = {spec["name"]: np.full(len(train_indices), np.nan) for spec in specs}
    fold_ids = np.zeros(len(train_indices), dtype=int)
    rows_by_style, fold_audit = [], []
    for fold, (train_local, test_local) in enumerate(LeaveOneGroupOut().split(xtr, ytr, groups), 1):
        heldout_style = str(groups[test_local[0]])
        train_global, test_global = train_indices[train_local], train_indices[test_local]
        assert not set(train_global) & set(source_indices)
        assert not set(groups[train_local]) & {heldout_style}
        fold_seed = LOSO_SEED + fold
        for spec in specs:
            name = spec["name"]
            if spec["family"] == "ensemble":
                values = np.mean([predictions[member["name"]][test_local] for member in spec["members"]], axis=0)
            else:
                model = make_domain_pipeline(spec, features, calibrated, aging, fold_seed)
                target = y45.iloc[train_global] if spec["family"] == "old_multioutput" else y.iloc[train_global]
                model.fit(x.iloc[train_global], target,
                          **training_fit_params(spec, meta.iloc[train_global].tasting_category_fine))
                values = predict_target(model, x.iloc[test_global], spec, target_position)
            predictions[name][test_local] = values
            rows_by_style.append({"method": name, "heldout_style": heldout_style,
                                  "fold": fold, "n": len(test_local),
                                  **scores(y.iloc[test_global], values),
                                  "bias": float(np.mean(values-y.iloc[test_global].to_numpy()))})
        fold_ids[test_local] = fold
        fold_audit.extend({"fold": fold, "heldout_style": heldout_style, "role": role,
                           "global_index": int(index), "beer": int(meta.iloc[index].beer)}
                          for role, indices in [("train", train_global), ("test", test_global)] for index in indices)
        pd.DataFrame(rows_by_style).to_csv(OUT / "loso_metrics_by_style.csv", index=False)
        print(f"Stale domain LOSO {fold}/21: {heldout_style} ({len(test_local)} samples)", flush=True)
    loso_by_style = pd.DataFrame(rows_by_style)
    audit = pd.DataFrame(fold_audit)
    audit.to_csv(OUT / "loso_split_audit.csv", index=False)
    overall_rows, prediction_rows = [], []
    for spec in specs:
        name, pred = spec["name"], predictions[spec["name"]]
        assert np.isfinite(pred).all()
        per_style = loso_by_style[loso_by_style.method.eq(name)]
        metric = scores(ytr, pred)
        macro_rmse = float(per_style.rmse.mean())
        macro_abs_bias = float(per_style.bias.abs().mean())
        within_pred_ss = within_truth_ss = 0.0
        for style in np.unique(groups):
            mask = groups == style
            within_pred_ss += float(np.sum((pred[mask]-pred[mask].mean())**2))
            within_truth_ss += float(np.sum((ytr.to_numpy()[mask]-ytr.to_numpy()[mask].mean())**2))
        overall_rows.append({"method": name, "selectable": spec["selectable"],
            "balanced": spec["balanced"], "family": spec["family"], "styles": len(per_style),
            "macro_style_rmse": macro_rmse, "macro_style_mae": float(per_style.mae.mean()),
            "macro_style_abs_bias": macro_abs_bias, "macro_style_bias": float(per_style.bias.mean()),
            "selection_score": macro_rmse+BIAS_PENALTY*macro_abs_bias,
            "nonconstant_qualified": metric["prediction_variance_ratio"] >= VARIANCE_RATIO_MIN,
            "within_style_prediction_variance_ratio": within_pred_ss/within_truth_ss,
            **metric})
        prediction_rows.append(pd.concat([meta.iloc[train_indices].reset_index(drop=True),
            pd.DataFrame({"global_index": train_indices, "method": name, "observed": ytr.to_numpy(),
                          "prediction": pred, "heldout_style": groups, "fold": fold_ids})], axis=1))
    overall = pd.DataFrame(overall_rows)
    mean_floor = float(overall.loc[overall.method.isin(["mean_global", "mean_style_balanced"]), "macro_style_rmse"].min())
    overall["improves_best_mean_macro_rmse"] = overall.macro_style_rmse < mean_floor
    overall["qualified_for_proxy"] = overall.selectable & overall.nonconstant_qualified & overall.improves_best_mean_macro_rmse
    eligible = overall[overall.selectable & overall.nonconstant_qualified]
    fallback_used = eligible.empty
    if fallback_used:
        eligible = overall[overall.selectable]
    selected_row = eligible.sort_values(["selection_score", "method"]).iloc[0]
    selected_method = str(selected_row.method)
    selected_config = spec_map[selected_method]
    qualified = bool(selected_row.qualified_for_proxy)
    loso_predictions = pd.concat(prediction_rows, ignore_index=True)
    loso_predictions.to_csv(OUT / "loso_predictions.csv", index=False)
    overall.to_csv(OUT / "loso_metrics_overall.csv", index=False)
    selection_result = {"selected_method": selected_method, "config": selected_config,
        "selected_loso_metrics": selected_row.to_dict(), "qualified_for_proxy": qualified,
        "fallback_used": fallback_used, "best_mean_macro_rmse": mean_floor,
        "selection_complete_before_pils_diagnostic": True,
        "qualification_scope": "training-only LOSO diagnostic; not an unbiased estimate after selection, not causal validation"}
    write_json(OUT / "selection_result.json", selection_result)
    print(f"Stale domain selected without Pils labels: {selected_method}, qualified={qualified}", flush=True)

    # Fit all fixed candidates only after the selection decision is frozen.
    # All Pils labels are used solely for the diagnostic tables below.
    all_fitted = {}
    for spec in specs:
        model = make_domain_pipeline(spec, features, calibrated, aging, FINAL_SEED)
        target = y45.iloc[train_indices] if spec["family"] == "old_multioutput" else ytr
        model.fit(xtr, target, **training_fit_params(spec, groups))
        all_fitted[spec["name"]] = model
    common_metadata = {"features": features, "target": TARGET,
        "target_scale": "attachment_4_standardized", "config": selected_config,
        "selected_config": selected_config, "selected_method": selected_method,
        "qualified_for_proxy": qualified, "qualification_scope": selection_result["qualification_scope"],
        "inner_oof_variance_ratio": float(selected_row.prediction_variance_ratio),
        "inner_oof_rmse": float(selected_row.rmse),
        "loso_macro_style_rmse": float(selected_row.macro_style_rmse),
        "loso_macro_style_abs_bias": float(selected_row.macro_style_abs_bias),
        "loso_selection_score": float(selected_row.selection_score),
        "train_global_indices": train_indices, "excluded_global_indices": source_indices,
        "excluded_style": "Pils/Lager", "selection_protocol": run_manifest["selection_rule"],
        "pils_diagnostic_used_for_selection": False}
    final_model = all_fitted[selected_method]
    joblib.dump({"pipeline": final_model, **common_metadata}, OUT / "transfer_stale_model.joblib", compress=3)

    diagnostic_rows, diagnostic_metrics = [], []
    for spec in specs:
        name = spec["name"]
        values = predict_target(all_fitted[name], x.iloc[source_indices], spec, target_position)
        diagnostic_rows.append(pd.concat([meta.iloc[source_indices].reset_index(drop=True),
            pd.DataFrame({"global_index": source_indices, "method": name,
                          "observed": y.iloc[source_indices].to_numpy(), "prediction": values,
                          "selected_without_pils_labels": name == selected_method})], axis=1))
        diagnostic_metrics.append({"method": name, "selected_without_pils_labels": name == selected_method,
                                   "n": len(source_indices), **scores(y.iloc[source_indices], values)})
    pd.concat(diagnostic_rows, ignore_index=True).to_csv(OUT / "pils_diagnostic_predictions.csv", index=False)
    pd.DataFrame(diagnostic_metrics).to_csv(OUT / "pils_diagnostic_metrics.csv", index=False)

    models, indices, bootstrap_seeds, estimator_seeds = [], [], [], []
    for b in range(60):
        seed = SELECTION_BOOTSTRAP_SEED+b if b < 40 else AUDIT_BOOTSTRAP_SEED+b-40
        index = np.random.default_rng(seed).choice(train_indices, size=len(train_indices), replace=True)
        estimator_seed = seed+10000
        model = make_domain_pipeline(selected_config, features, calibrated, aging, estimator_seed)
        model.fit(x.iloc[index], y.iloc[index],
                  **training_fit_params(selected_config, meta.iloc[index].tasting_category_fine))
        models.append(model)
        indices.append(index)
        bootstrap_seeds.append(seed)
        estimator_seeds.append(estimator_seed)
        if (b+1) % 10 == 0:
            print(f"Stale domain bootstrap {b+1}/60", flush=True)
    ensemble = {"models": models, "bootstrap_indices": indices, **common_metadata,
        "selection_model_indices": list(range(40)), "audit_model_indices": list(range(40, 60)),
        "bootstrap_seeds": bootstrap_seeds, "estimator_seeds": estimator_seeds,
        "pairing": "same_model_candidate_minus_source", "paired_quantile": 0.95,
        "hyperparameter_uncertainty_included": False,
        "resampling": "iid rows from 239 non-Pils; style-balanced weights re-estimated per draw where used",
        "audit_protocol": "first 40 screening; final 20 once-only audit; never reselect after audit",
        "interpretation": "fixed-configuration model-proxy bootstrap; not a calibrated confidence interval"}
    joblib.dump(ensemble, OUT / "bootstrap_ensemble.joblib", compress=3)
    np.savez_compressed(OUT / "bootstrap_indices.npz", indices=np.asarray(indices),
                        bootstrap_seeds=np.asarray(bootstrap_seeds), estimator_seeds=np.asarray(estimator_seeds))
    assert len(np.unique(groups)) == 21
    assert len(loso_predictions) == len(train_indices)*len(specs)
    assert np.asarray(indices).shape == (60, 239)
    assert all(not set(index) & set(source_indices) for index in indices)
    assert len(set(bootstrap_seeds)) == 60
    assert np.isfinite(predict_1d(final_model, x)).all()
    if selected_config["family"] == "local":
        write_local_neighborhood_diagnostics(final_model, models, indices, train_indices,
                                             source_indices, meta, x)
    selected_pils = next(row for row in diagnostic_metrics if row["method"] == selected_method)
    summary = {
        "target": TARGET, "samples": len(x), "training_samples": len(train_indices),
        "excluded_pils_sources": len(source_indices), "features": len(features), "aging_features": len(aging),
        "loso_styles": len(np.unique(groups)), "candidate_count": len(specs),
        "mean_baseline_method": "mean_global",
        "style_balanced_mean_baseline_method": "mean_style_balanced",
        "old_multioutput_baseline_method": "old_multioutput_tree",
        "old_single_tree_baseline_method": "fixed_single_tree",
        **selection_result, "selected_pils_diagnostic_metrics": selected_pils,
        "all_pils_diagnostic_metrics": diagnostic_metrics,
        "pils_status": "previously repeatedly viewed diagnostic set; not a new test set",
        "pils_used_for_training_selection_or_intercept": False,
        "bootstrap": {"models": 60, "selection": 40, "audit": 20,
                      "indices_shape": [60, 239], "fixed_selected_configuration": True,
                      "style_balanced_weights_refitted": bool(selected_config["balanced"]),
                      "algorithm_selection_uncertainty_included": False},
        "limitations": [
            "Complete LOSO uses only the 239 non-Pils samples; style labels determine groups and training weights, never prediction covariates.",
            "LOSO macro metrics treat the 21 observed styles equally; styles are observational groupings, not independent randomized domains.",
            "The selected score is the minimum among fixed candidate comparisons and is optimistic after selection; no outer grouped evaluation of the selector is claimed.",
            "Pils labels have been repeatedly inspected in prior work; current Pils metrics are diagnostics, not a fresh test or basis for selection/intercept correction.",
            "All candidate Pils results are retained and never change the previously frozen selected method.",
            "The variance gate is not a guarantee of response to each candidate chemical change.",
            "Row bootstrap conditions on the observed style pool and fixed chosen configuration; it omits algorithm-selection uncertainty and new-domain uncertainty.",
            "Screening and audit model draws share the same data; audit quantiles are neither independent experiments nor calibrated confidence bounds.",
            "Chemical-only prediction may miss oxidation, storage and process causes; passing model constraints does not establish actual stale-hops taste.",
            "Targets retain the supplied standardized scale; original 0-6 sensory scores cannot be recovered.",
        ],
        "verification": {"source_exclusion": True, "loso_styles": 21, "loso_prediction_rows": len(loso_predictions),
                         "bootstrap_indices_shape": [60, 239], "bootstrap_seeds_unique": True,
                         "prediction_finite": True},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": perf_counter()-started,
    }
    write_json(OUT / "summary.json", summary)
    print(overall[["method", "macro_style_rmse", "macro_style_abs_bias", "selection_score",
                   "prediction_variance_ratio", "qualified_for_proxy"]].to_string(index=False), flush=True)
    print("Pils selected diagnostic:", json.dumps(selected_pils, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
