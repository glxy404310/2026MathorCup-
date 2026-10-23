"""Fixed single-target forest follow-up on two additional internal splits.

Parameters are not tuned. This follows exploratory algorithm comparisons on
the same 250 samples and is not an independent external confirmation dataset.
"""
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

from common import SOLUTION
from q3_data import load_q3_data
from q3_stale_optimized import VARIANCE_RATIO_MIN, predict_1d, raw_tree_pipeline, scores

OUT = SOLUTION / "outputs" / "q3_optimized" / "stale_fixed"
TARGET = "stale_hops"
SEEDS = (20263123, 20264123)
TRANSFER_CV_SEED = 20266123
SELECTION_BOOTSTRAP_SEED = 20267123
AUDIT_BOOTSTRAP_SEED = 20268123


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def paired_cluster_bootstrap(oof, repeats=2000, seed=20269123):
    """Resample beers, after averaging each beer's repeated squared errors."""
    errors = oof.assign(squared_error=(oof.prediction-oof.observed)**2)
    per_beer = errors.groupby(["beer", "method"], sort=True).squared_error.mean().unstack("method")
    fixed = per_beer["fixed_single_extra_trees"].to_numpy()
    old = per_beer["old_multioutput_extra_trees"].to_numpy()
    draws = np.random.default_rng(seed).integers(0, len(per_beer), size=(repeats, len(per_beer)))
    fixed_rmse, old_rmse = np.sqrt(fixed[draws].mean(axis=1)), np.sqrt(old[draws].mean(axis=1))
    difference, reduction = fixed_rmse-old_rmse, 1-fixed_rmse/old_rmse
    return {"seed": seed, "repeats": repeats, "independent_resampling_unit": "beer",
        "beers": len(per_beer), "pre_aggregation": "mean squared error within beer and method across two repeats",
        "comparison": "fixed single-target ExtraTrees minus old multi-output ExtraTrees",
        "point_rmse_difference": float(np.sqrt(fixed.mean())-np.sqrt(old.mean())),
        "rmse_difference_percentile95": np.quantile(difference, [.025, .975]).tolist(),
        "point_relative_rmse_reduction": float(1-np.sqrt(fixed.mean())/np.sqrt(old.mean())),
        "relative_rmse_reduction_percentile95": np.quantile(reduction, [.025, .975]).tolist(),
        "bootstrap_fraction_rmse_difference_below_zero": float(np.mean(difference<0)),
        "interpretation": "sampling uncertainty conditional on saved OOF predictions; not a selection-adjusted external-validation interval",
        "limitations": ["Same-data algorithm exploration/selection is not resampled or accounted for",
                        "Models are not refitted within these 2000 resamples",
                        "Shared training folds create dependencies beyond the repeated-beer clustering"]}


def main():
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    meta, x, y45, _ = load_q3_data()
    y = y45[TARGET]
    features = x.columns.tolist()
    selected_config = {"name": "fixed_single_extra_trees_300", "family": "extra_trees",
                       "feature_set": "all226", "representation": "raw_hybrid",
                       "params": {"n_estimators": 300, "min_samples_leaf": 3,
                                  "max_features": 0.7, "n_jobs": 1},
                       "bootstrap_n_estimators": 120, "tuning": "none"}
    write_json(OUT / "run_manifest.json", {
        "selected_config": selected_config, "outer_seeds": list(SEEDS), "outer_folds": 5,
        "transfer_cv_seed": TRANSFER_CV_SEED, "selection_bootstrap_seed": SELECTION_BOOTSTRAP_SEED,
        "audit_bootstrap_seed": AUDIT_BOOTSTRAP_SEED, "features": features,
        "preprocessing": "raw hybrid; training-only median imputation with missing indicators",
        "status": "fixed follow-up after exploratory algorithm comparison on the same data",
        "not_independent_external_confirmation": True, "parameter_tuning": False,
    })
    methods = ["mean", "old_multioutput_extra_trees", "fixed_single_extra_trees"]
    oof_rows, metric_rows, fold_rows = [], [], []
    stale_position = y45.columns.get_loc(TARGET)
    for repeat, seed in enumerate(SEEDS, 1):
        predictions = {name: np.full(len(x), np.nan) for name in methods}
        folds = np.zeros(len(x), dtype=int)
        for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=seed).split(x), 1):
            model_seed = seed + fold
            old = raw_tree_pipeline(model_seed).fit(x.iloc[train], y45.iloc[train])
            single = raw_tree_pipeline(model_seed).fit(x.iloc[train], y.iloc[train])
            predictions["old_multioutput_extra_trees"][test] = old.predict(x.iloc[test])[:, stale_position]
            predictions["fixed_single_extra_trees"][test] = predict_1d(single, x.iloc[test])
            predictions["mean"][test] = y.iloc[train].mean()
            folds[test] = fold
            for method in methods:
                fold_rows.append({"repeat": repeat, "outer_seed": seed, "fold": fold,
                                  "method": method, **scores(y.iloc[test], predictions[method][test])})
            print(f"Stale fixed repeat {repeat}/2 fold {fold}/5: "
                  f"single RMSE={scores(y.iloc[test], predictions['fixed_single_extra_trees'][test])['rmse']:.6f}", flush=True)
        for method, pred in predictions.items():
            assert np.isfinite(pred).all()
            metric_rows.append({"scope": "per_seed", "repeat": repeat, "outer_seed": seed,
                                "method": method, **scores(y, pred)})
            oof_rows.append(pd.concat([meta, pd.DataFrame({"repeat": repeat, "outer_seed": seed,
                "fold": folds, "method": method, "observed": y, "prediction": pred})], axis=1))
    oof = pd.concat(oof_rows, ignore_index=True)
    for method in methods:
        group = oof[oof.method.eq(method)]
        metric_rows.append({"scope": "pooled_repeated_oof", "repeat": 0, "outer_seed": 0,
                            "method": method, **scores(group.observed, group.prediction)})
    metrics = pd.DataFrame(metric_rows)
    oof.to_csv(OUT / "oof_predictions.csv", index=False)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "metrics_by_fold.csv", index=False)
    cluster_result = paired_cluster_bootstrap(oof)
    write_json(OUT / "paired_cluster_bootstrap.json", cluster_result)
    pooled = metrics[metrics.scope.eq("pooled_repeated_oof")].set_index("method")

    excluded = np.flatnonzero(meta.tasting_category_fine.eq("Pils/Lager"))
    train_global = np.flatnonzero(~meta.tasting_category_fine.eq("Pils/Lager"))
    assert len(excluded) == 11 and len(train_global) == 239
    train_oof, mean_oof = np.full(len(train_global), np.nan), np.full(len(train_global), np.nan)
    for fold, (tr, te) in enumerate(KFold(3, shuffle=True, random_state=TRANSFER_CV_SEED).split(train_global), 1):
        model = raw_tree_pipeline(TRANSFER_CV_SEED + fold).fit(x.iloc[train_global[tr]], y.iloc[train_global[tr]])
        train_oof[te] = predict_1d(model, x.iloc[train_global[te]])
        mean_oof[te] = y.iloc[train_global[tr]].mean()
    train_metrics, mean_metrics = scores(y.iloc[train_global], train_oof), scores(y.iloc[train_global], mean_oof)
    qualified = bool(train_metrics["prediction_variance_ratio"] >= VARIANCE_RATIO_MIN
                     and train_metrics["rmse"] < mean_metrics["rmse"])
    final = raw_tree_pipeline(TRANSFER_CV_SEED + 90).fit(x.iloc[train_global], y.iloc[train_global])
    common_metadata = {"features": features, "target": TARGET,
        "target_scale": "attachment_4_standardized", "selected_config": selected_config,
        "qualified_for_proxy": qualified, "inner_oof_variance_ratio": train_metrics["prediction_variance_ratio"],
        "inner_oof_rmse": train_metrics["rmse"], "inner_mean_baseline_rmse": mean_metrics["rmse"],
        "train_global_indices": train_global, "excluded_global_indices": excluded,
        "excluded_style": "Pils/Lager", "qualification_scope": "training_only_fixed_model_CV; not holdout or counterfactual validation",
        "qualification_rule": "training-only OOF variance ratio >=0.05 and RMSE < same-fold mean baseline",
        "parameter_tuning": False}
    artifact = {"pipeline": final, **common_metadata}
    joblib.dump(artifact, OUT / "transfer_stale_model.joblib", compress=3)
    pd.concat([meta.iloc[train_global].reset_index(drop=True), pd.DataFrame({"observed": y.iloc[train_global].to_numpy(),
        "fixed_inner_oof_prediction": train_oof, "mean_inner_oof_prediction": mean_oof})], axis=1).to_csv(
            OUT / "transfer_inner_oof.csv", index=False)
    heldout = pd.concat([meta.iloc[excluded].reset_index(drop=True), pd.DataFrame({
        "observed": y.iloc[excluded].to_numpy(), "prediction": predict_1d(final, x.iloc[excluded]),
        "mean_baseline_prediction": y.iloc[train_global].mean()})], axis=1)
    old_artifact = joblib.load(SOLUTION / "outputs" / "q3" / "sensory" / "transfer_proxy.joblib")
    assert not set(old_artifact["train_global_indices"]) & set(excluded)
    heldout["old_multioutput_prediction"] = old_artifact["pipeline"].predict(x.iloc[excluded])[:, stale_position]
    heldout.to_csv(OUT / "pils_style_holdout_predictions.csv", index=False)
    heldout_metrics = [{"model": name, **scores(heldout.observed, heldout[column])} for name, column in [
        ("fixed_single_extra_trees", "prediction"), ("non_pils_mean", "mean_baseline_prediction"),
        ("old_multioutput_transfer_proxy", "old_multioutput_prediction")]]
    pd.DataFrame(heldout_metrics).to_csv(OUT / "pils_style_holdout_metrics.csv", index=False)

    models, indices, seeds = [], [], []
    for b in range(60):
        seed = SELECTION_BOOTSTRAP_SEED + b if b < 40 else AUDIT_BOOTSTRAP_SEED + b - 40
        index = np.random.default_rng(seed).choice(train_global, size=len(train_global), replace=True)
        model = raw_tree_pipeline(seed + 10000, n_estimators=120).fit(x.iloc[index], y.iloc[index])
        models.append(model)
        indices.append(index)
        seeds.append(seed)
        if (b + 1) % 20 == 0:
            print(f"Stale fixed bootstrap {b + 1}/60", flush=True)
    ensemble = {"models": models, "bootstrap_indices": indices, **common_metadata,
        "bootstrap_seeds": seeds, "estimator_seeds": [seed + 10000 for seed in seeds],
        "selection_model_indices": list(range(40)), "audit_model_indices": list(range(40, 60)),
        "pairing": "same_model_candidate_minus_source", "paired_quantile": 0.95,
        "bootstrap_n_estimators": 120, "final_n_estimators": 300,
        "hyperparameter_uncertainty_included": False,
        "audit_protocol": "40 screening models; 20 separate bootstrap draws for one-time audit; no reselection",
        "interpretation": "fixed single-target forest proxy; quantiles are not calibrated confidence bounds"}
    joblib.dump(ensemble, OUT / "bootstrap_ensemble.joblib", compress=3)
    np.savez_compressed(OUT / "bootstrap_indices.npz", indices=np.asarray(indices), bootstrap_seeds=np.asarray(seeds))
    assert np.asarray(indices).shape == (60, 239)
    assert all(not set(index) & set(excluded) for index in indices)
    assert len(set(seeds)) == 60
    assert np.isfinite(predict_1d(final, x)).all()
    summary = {
        "status": "fixed follow-up after exploratory model comparisons on the same dataset",
        "samples": len(x), "features": len(features), "target": TARGET,
        "outer_seeds": list(SEEDS), "outer_folds": 5, "parameter_tuning": False,
        "selected_config": selected_config,
        "pooled_repeat_metrics": [{"method": method, **pooled.loc[method].to_dict()} for method in methods],
        "rmse_reduction_vs_mean": float(1-pooled.loc['fixed_single_extra_trees', 'rmse']/pooled.loc['mean', 'rmse']),
        "rmse_reduction_vs_old_multioutput": float(1-pooled.loc['fixed_single_extra_trees', 'rmse']/pooled.loc['old_multioutput_extra_trees', 'rmse']),
        "paired_cluster_bootstrap": cluster_result,
        "transfer_qualified_for_proxy": qualified, "transfer_inner_oof_metrics": train_metrics,
        "transfer_inner_mean_metrics": mean_metrics, "transfer_cv_seed": TRANSFER_CV_SEED,
        "transfer_qualification_scope": common_metadata["qualification_scope"],
        "pils_holdout_metrics": heldout_metrics,
        "pils_holdout_improves_mean": heldout_metrics[0]["rmse"] < heldout_metrics[1]["rmse"],
        "pils_holdout_improves_old": heldout_metrics[0]["rmse"] < heldout_metrics[2]["rmse"],
        "bootstrap": {"models": 60, "selection": 40, "audit": 20, "trees_per_model": 120,
                      "final_trees": 300, "population": "239 non-Pils only", "structure_fixed": True},
        "limitations": [
            "Additional split seeds reuse the same 250 samples; the exploratory algorithm choice is not independently externally confirmed.",
            "No parameter tuning or family search is performed in this follow-up.",
            "The 500 repeated OOF predictions do not constitute 500 independent beers.",
            "All transfer preprocessing, fitting and bootstrap sampling exclude every Pils/Lager source.",
            "Source-style holdout is reported after the fixed model is trained and never changes the chosen algorithm.",
            "The 0.05 OOF variance-ratio gate does not prove candidate chemical perturbation responsiveness.",
            "Bootstrap uses 120-tree approximations to the 300-tree final forest, with fixed parameters and no algorithm-selection uncertainty.",
            "Separate screening/audit bootstrap draws share the same observed training population and are not independent experimental validation.",
            "Passing paired proxy quantiles cannot establish real stale-hops taste or causal effects; original 0-6 scores remain unavailable.",
        ],
        "verification": {"bootstrap_indices_shape": [60, 239], "source_exclusion": True,
                         "bootstrap_seeds_unique": True, "predictions_finite": True},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": perf_counter()-started,
    }
    write_json(OUT / "summary.json", summary)
    print(pooled[["rmse", "r2", "spearman", "prediction_std_ratio"]].to_string(), flush=True)
    print("Fixed transfer qualified:", qualified, "holdout:", json.dumps(heldout_metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
