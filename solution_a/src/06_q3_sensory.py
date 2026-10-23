"""Train and validate the hybrid-representation sensory proxy for question 3.

Predictions are attachment-standardized values, not recovered 0--6 scores.
Bootstrap paired differences constrain a fitted proxy, not actual taste.
"""
from __future__ import annotations

import json
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from common import SEED, SOLUTION
from q3_data import DEFECTS, load_q3_data

OUT = SOLUTION / "outputs" / "q3" / "sensory"
N_BOOTSTRAP = 40
BOOTSTRAP_TREES = 120


def tree_pipeline(seed: int, n_estimators: int = 300) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True,
                                  keep_empty_features=True)),
        ("model", ExtraTreesRegressor(
            n_estimators=n_estimators, min_samples_leaf=3, max_features=0.7,
            random_state=seed, n_jobs=1)),
    ])


def scores(y, pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "macro_r2": float(r2_score(y, pred, multioutput="uniform_average")),
    }


def main() -> None:
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    meta, x, y, manifest = load_q3_data()
    # Input artifacts preserve missingness; imputation is fitted by each model.
    meta.to_csv(OUT / "meta.csv", index=False)
    x.to_csv(OUT / "hybrid_features.csv", index=False)
    y.to_csv(OUT / "y45.csv", index=False)
    manifest.to_csv(OUT / "feature_manifest.csv", index=False)
    (OUT / "defect_thresholds.json").write_text(
        json.dumps(DEFECTS, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    predictions = {name: np.full(y.shape, np.nan) for name in ("mean", "extra_trees")}
    fold_ids = np.zeros(len(x), dtype=int)
    fold_metrics = []
    for fold, (train, test) in enumerate(
            KFold(5, shuffle=True, random_state=SEED).split(x), 1):
        fold_ids[test] = fold
        model = tree_pipeline(SEED + fold).fit(x.iloc[train], y.iloc[train])
        predictions["extra_trees"][test] = model.predict(x.iloc[test])
        predictions["mean"][test] = y.iloc[train].mean().to_numpy()
        for name, values in predictions.items():
            fold_metrics.append({"fold": fold, "model": name, "train_n": len(train),
                                 "test_n": len(test), **scores(y.iloc[test], values[test])})
        print(f"Q3 sensory OOF fold {fold}/5: "
              f"RMSE={scores(y.iloc[test], predictions['extra_trees'][test])['rmse']:.6f}", flush=True)

    metric_rows, target_rows = [], []
    for name, values in predictions.items():
        if not np.isfinite(values).all():
            raise AssertionError(f"Incomplete OOF predictions: {name}")
        metric_rows.append({"model": name, **scores(y, values)})
        for j, target in enumerate(y):
            target_rows.append({"model": name, "target": target,
                                **scores(y.iloc[:, j], values[:, j]),
                                "spearman": float(spearmanr(y.iloc[:, j], values[:, j]).statistic)})
        pd.concat([meta, pd.DataFrame({"fold": fold_ids}),
                   pd.DataFrame(values, columns=y.columns)], axis=1).to_csv(
                       OUT / f"oof_{name}.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUT / "metrics_overall.csv", index=False)
    pd.DataFrame(target_rows).to_csv(OUT / "metrics_by_target.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(OUT / "metrics_by_fold.csv", index=False)
    pd.concat([meta, y], axis=1).to_csv(OUT / "observed_standardized.csv", index=False)

    # A full-data inference model is distinct from held-out evaluation above.
    final = tree_pipeline(SEED + 99).fit(x, y)
    artifact = {"pipeline": final, "features": x.columns.tolist(),
                "targets": y.columns.tolist(), "target_scale": "attachment_4_standardized",
                "representation": "69_calibrated_plus_157_relative", "seed": SEED}
    joblib.dump(artifact, OUT / "sensory_model.joblib", compress=3)
    pd.concat([meta, pd.DataFrame(final.predict(x), columns=y.columns)], axis=1).to_csv(
        OUT / "fitted_standardized_NOT_validation.csv", index=False)

    # Migration sources must not appear in their own proxy's training set.
    source_mask = meta["tasting_category_fine"].eq("Pils/Lager").to_numpy()
    transfer_train = np.flatnonzero(~source_mask)
    transfer_test = np.flatnonzero(source_mask)
    if (len(transfer_train), len(transfer_test)) != (239, 11):
        raise ValueError("Expected 239 non-Pils training samples and 11 held-out sources")
    transfer_model = tree_pipeline(SEED + 199).fit(x.iloc[transfer_train], y.iloc[transfer_train])
    transfer_artifact = {
        "pipeline": transfer_model, "features": x.columns.tolist(), "targets": y.columns.tolist(),
        "target_scale": "attachment_4_standardized", "representation": "69_calibrated_plus_157_relative",
        "seed": SEED, "excluded_style": "Pils/Lager", "train_global_indices": transfer_train,
        "heldout_global_indices": transfer_test, "source_exclusion_reason": "prevent_source_memorization",
    }
    joblib.dump(transfer_artifact, OUT / "transfer_proxy.joblib", compress=3)
    heldout_pred = transfer_model.predict(x.iloc[transfer_test])
    heldout_mean = np.broadcast_to(y.iloc[transfer_train].mean().to_numpy(), heldout_pred.shape)
    heldout_metrics, heldout_targets = [], []
    for name, values in [("transfer_proxy", heldout_pred), ("non_pils_mean", heldout_mean)]:
        heldout_metrics.append({"model": name, "test_n": len(transfer_test),
                                **scores(y.iloc[transfer_test], values)})
        for j, target in enumerate(y):
            heldout_targets.append({"model": name, "target": target,
                                    **scores(y.iloc[transfer_test, j], values[:, j])})
    pd.concat([meta.iloc[transfer_test].reset_index(drop=True),
               pd.DataFrame(heldout_pred, columns=y.columns)], axis=1).to_csv(
                   OUT / "transfer_pils_heldout_predictions.csv", index=False)
    pd.concat([meta.iloc[transfer_test].reset_index(drop=True),
               y.iloc[transfer_test].reset_index(drop=True)], axis=1).to_csv(
                   OUT / "transfer_pils_heldout_observed.csv", index=False)
    pd.DataFrame(heldout_metrics).to_csv(OUT / "transfer_pils_heldout_metrics.csv", index=False)
    pd.DataFrame(heldout_targets).to_csv(OUT / "transfer_pils_heldout_metrics_by_target.csv", index=False)

    # Row-paired bootstrap: one fitted model predicts BOTH source and candidate.
    # Every Pils/Lager source is excluded. Save global resample positions so
    # exclusion and the train/select-audit split can be independently checked.
    rng = np.random.default_rng(SEED + 2000)
    models, indices = [], []
    for b in range(N_BOOTSTRAP):
        index = rng.choice(transfer_train, size=len(transfer_train), replace=True)
        fitted = tree_pipeline(SEED + 3000 + b, BOOTSTRAP_TREES).fit(
            x.iloc[index], y["stale_hops"].iloc[index])
        models.append(fitted)
        indices.append(index)
        if (b + 1) % 10 == 0:
            print(f"Q3 stale_hops bootstrap {b + 1}/{N_BOOTSTRAP}", flush=True)
    bootstrap = {"models": models, "bootstrap_indices": indices,
                 "features": x.columns.tolist(), "target": "stale_hops",
                 "target_scale": "attachment_4_standardized", "seed": SEED,
                 "bootstrap_seed": SEED + 2000, "paired_quantile": 0.95,
                 "pairing": "same_model_source_candidate_difference",
                 "train_global_indices": transfer_train,
                 "excluded_global_indices": transfer_test,
                 "excluded_style": "Pils/Lager",
                 "selection_model_indices": list(range(30)),
                 "audit_model_indices": list(range(30, N_BOOTSTRAP)),
                 "audit_protocol": "screen_with_first_30; audit_last_10_once_without_reselection",
                 "interpretation": "model_proxy_uncertainty_only_not_taste_guarantee"}
    joblib.dump(bootstrap, OUT / "stale_bootstrap.joblib", compress=3)
    np.savez_compressed(OUT / "stale_bootstrap_indices.npz", indices=np.asarray(indices),
                        source_beer=meta["beer"].to_numpy())
    bootstrap_predictions = np.vstack([model.predict(x) for model in models])
    pd.concat([meta, pd.DataFrame({
        "bootstrap_mean": bootstrap_predictions.mean(axis=0),
        "bootstrap_q025": np.quantile(bootstrap_predictions, 0.025, axis=0),
        "bootstrap_q975": np.quantile(bootstrap_predictions, 0.975, axis=0),
        "observed_stale_hops": y["stale_hops"],
    })], axis=1).to_csv(OUT / "stale_bootstrap_fitted_NOT_validation.csv", index=False)

    # Reuse of the established Q1 folds is explicitly checked if available.
    reference_path = SOLUTION / "outputs" / "q1" / "oof_extra_trees.csv"
    identical_q1_folds = None
    if reference_path.exists():
        reference = pd.read_csv(reference_path)
        identical_q1_folds = bool(np.array_equal(reference["beer"], meta["beer"])
                                  and np.array_equal(reference["fold"], fold_ids))
        if not identical_q1_folds:
            raise AssertionError("Q3 OOF does not match existing Q1 folds")
    overall = pd.DataFrame(metric_rows).set_index("model")
    stale = [row for row in target_rows if row["target"] == "stale_hops"]
    summary = {
        "seed": SEED, "samples": len(x), "features": x.shape[1], "targets": y.shape[1],
        "representation_counts": manifest["representation"].value_counts().to_dict(),
        "calibrated_negatives_invalidated": int(manifest["negative_invalid_count"].sum()),
        "hybrid_missing_values": int(x.isna().sum().sum()),
        "validation": "fixed 5-fold shuffled KFold shared with Q1; fold-local median imputation",
        "identical_q1_folds": identical_q1_folds,
        "parameters": {"n_estimators": 300, "min_samples_leaf": 3, "max_features": 0.7,
                       "n_jobs": 1, "imputer": "median_with_missing_indicator"},
        "metrics": metric_rows, "stale_hops_metrics": stale,
        "rmse_reduction_vs_mean": float(1 - overall.loc["extra_trees", "rmse"] /
                                           overall.loc["mean", "rmse"]),
        "bootstrap": {"repeats": N_BOOTSTRAP, "n_estimators": BOOTSTRAP_TREES,
                      "target": "stale_hops", "row_resampling": "paired X/y iid with replacement",
                      "population": "239 non-Pils/Lager samples; all 11 Pils sources excluded",
                      "selection_models": 30, "independent_audit_models": 10,
                      "audit_protocol": "no reselection after audit",
                      "paired_difference_quantile": 0.95,
                      "coverage_calibrated": False,
                      "audit": "stale_bootstrap_indices.npz and saved bootstrap_indices"},
        "transfer_proxy": {"train_n": len(transfer_train), "heldout_n": len(transfer_test),
                           "excluded_style": "Pils/Lager", "metrics": heldout_metrics,
                           "stale_hops_metrics": [row for row in heldout_targets
                                                  if row["target"] == "stale_hops"],
                           "limitations": "11 held-out sources in one style; not a calibrated transfer test"},
        "defect_thresholds": DEFECTS,
        "limitations": [
            "Targets are attachment-standardized sensory values, not original 0-6 scores.",
            "Random folds assess new samples in the observed style mixture, not unseen-style transfer.",
            "Attachment-level preprocessing and calibration cannot be re-estimated within folds.",
            "Optimization candidates may be out of distribution despite passing chemical constraints.",
            "Stale hops no-increase is a model-proxy constraint, not an actual tasting guarantee.",
            "40 iid row bootstrap models give descriptive paired-model uncertainty; the upper 95% "
            "quantile has neither calibrated coverage nor a physical causal interpretation.",
            "Full-data fitted predictions and bootstrap fitted bands are not validation.",
        ],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": perf_counter() - started,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(pd.DataFrame(metric_rows).to_string(index=False), flush=True)
    print("stale_hops:", json.dumps(stale, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
