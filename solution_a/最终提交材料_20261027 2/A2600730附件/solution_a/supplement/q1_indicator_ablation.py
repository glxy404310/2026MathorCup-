"""Paired Q1 missing-indicator ablation; frozen primary results are read only.

Run this file to reproduce the comparison. Pass --verify-only to independently
recompute saved metrics, folds, masks, preprocessing and input hashes, without
fitting models or changing files. The sole configured model difference is
SimpleImputer.add_indicator. No parameter is selected from these results.
"""
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from common import SEED, data_dir, load_xy

OUT = HERE / "results/q1_indicator_ablation"
ARMS = {"with_indicator": True, "without_indicator": False}
RATES = {"original": 0.0, "mask10": 0.1, "mask20": 0.2}
TREE_PARAMETERS = {"n_estimators": 300, "min_samples_leaf": 3,
                   "max_features": 0.7, "n_jobs": 1}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(actual, predicted):
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "macro_r2": float(r2_score(actual, predicted, multioutput="uniform_average"))}


def cohorts(x):
    missing = x.isna().any(axis=1).to_numpy()
    assert int(missing.sum()) == 63 and int((~missing).sum()) == 187
    return {"all": np.ones(len(x), dtype=bool), "original_missing": missing,
            "original_complete": ~missing}


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, lineterminator="\n")


def json_file(name, record):
    (OUT / name).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8", newline="\n")


def baseline_comparison(predictions, meta, y, fold_ids):
    rows = []
    for scenario in RATES:
        suffix = "" if scenario == "original" else "_" + scenario
        frozen = pd.read_csv(ROOT / f"outputs/q1/oof_extra_trees{suffix}.csv")
        assert frozen.beer.tolist() == meta.beer.tolist()
        np.testing.assert_array_equal(frozen.fold, fold_ids)
        old = frozen[y.columns].to_numpy()
        current = predictions[("with_indicator", scenario)]
        difference = np.abs(current - old)
        rows.append({"scenario": scenario, "frozen_rmse": scores(y, old)["rmse"],
                     "refit_rmse": scores(y, current)["rmse"],
                     "max_abs_prediction_difference": float(difference.max()),
                     "mean_abs_prediction_difference": float(difference.mean()),
                     "same_predictions_at_1e_10": bool(np.allclose(current, old, rtol=0, atol=1e-10))})
    return rows


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [p for folder in [data_dir(), ROOT / "outputs"]
                 for p in sorted(folder.rglob("*")) if p.is_file()]
    protected += [ROOT / "src/common.py", ROOT / "src/02_q1_baseline.py"]
    hashes = {p.relative_to(REPO).as_posix(): sha256(p) for p in protected}
    meta, x, y = load_xy()
    groups = cohorts(x)
    actual = y.to_numpy()
    predictions = {(arm, scenario): np.full(y.shape, np.nan)
                   for arm in ARMS for scenario in RATES}
    fold_ids = np.zeros(len(x), dtype=int)
    masks = {scenario: np.zeros(x.shape, dtype=bool) for scenario in ["mask10", "mask20"]}
    preprocessing, mask_rows = [], []
    for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(x), 1):
        fold_ids[test] = fold
        xtr, xte = x.iloc[train], x.iloc[test]
        uniforms = np.random.default_rng(SEED + 100 + fold).random(xte.shape)
        inputs = {"original": xte}
        for scenario in ["mask10", "mask20"]:
            mask = (uniforms < RATES[scenario]) & xte.notna().to_numpy()
            masks[scenario][test] = mask
            inputs[scenario] = xte.mask(mask)
            mask_rows.append({"fold": fold, "scenario": scenario, "rate": RATES[scenario],
                              "additional_hidden": int(mask.sum()),
                              "original_observed": int(xte.notna().sum().sum())})
        for arm, indicator in ARMS.items():
            fitted = Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=indicator,
                                          keep_empty_features=True)),
                ("model", ExtraTreesRegressor(random_state=SEED + fold, **TREE_PARAMETERS))])
            fitted.fit(xtr, y.iloc[train])
            imputer, tree = fitted.named_steps["imputer"], fitted.named_steps["model"]
            np.testing.assert_allclose(imputer.statistics_, xtr.median().to_numpy())
            preprocessing.append({"fold": fold, "arm": arm, "train_n": len(train),
                                  "test_n": len(test), "median_statistics": imputer.statistics_.tolist(),
                                  "indicator_indices": imputer.indicator_.features_.tolist() if indicator else [],
                                  "n_model_features": tree.n_features_in_,
                                  "n_candidate_features_per_split": tree.estimators_[0].max_features_})
            for scenario, test_input in inputs.items():
                predictions[(arm, scenario)][test] = fitted.predict(test_input)
        print(f"Indicator ablation fold {fold}/5 completed", flush=True)
    assert all(np.isfinite(p).all() for p in predictions.values())
    assignment = meta.assign(fold=fold_ids, original_missing=groups["original_missing"],
                             original_missing_count=x.isna().sum(axis=1))
    assignment.to_csv(OUT / "fold_assignments.csv", index=False, lineterminator="\n")
    np.savez_compressed(OUT / "test_masks.npz", **masks)
    json_file("preprocessing.json", preprocessing)
    csv("mask_counts.csv", mask_rows)
    metric_rows, fold_rows, paired_rows, error_rows = [], [], [], []
    for (arm, scenario), predicted in predictions.items():
        pd.concat([assignment, pd.DataFrame(predicted, columns=y.columns)], axis=1).to_csv(
            OUT / f"oof_{arm}_{scenario}.csv", index=False, lineterminator="\n")
        for group, selected in groups.items():
            metric_rows.append({"arm": arm, "scenario": scenario, "rate": RATES[scenario],
                                "cohort": group, "n": int(selected.sum()),
                                **scores(actual[selected], predicted[selected])})
            for fold in range(1, 6):
                subset = selected & (fold_ids == fold)
                fold_rows.append({"arm": arm, "scenario": scenario, "cohort": group,
                                  "fold": fold, "n": int(subset.sum()),
                                  **scores(actual[subset], predicted[subset])})
    for scenario in RATES:
        with_pred, without_pred = [predictions[(arm, scenario)] for arm in ARMS]
        square = [np.mean((pred - actual) ** 2, axis=1) for pred in [with_pred, without_pred]]
        absolute = [np.mean(np.abs(pred - actual), axis=1) for pred in [with_pred, without_pred]]
        for i in range(len(x)):
            error_rows.append({"beer": int(meta.beer.iloc[i]), "fold": int(fold_ids[i]),
                               "scenario": scenario, "original_missing": bool(groups["original_missing"][i]),
                               "mse_with_indicator": float(square[0][i]),
                               "mse_without_indicator": float(square[1][i]),
                               "delta_mse": float(square[0][i] - square[1][i]),
                               "mae_with_indicator": float(absolute[0][i]),
                               "mae_without_indicator": float(absolute[1][i])})
        for group, selected in groups.items():
            a, b = [scores(actual[selected], pred[selected]) for pred in [with_pred, without_pred]]
            paired_rows.append({"scenario": scenario, "cohort": group, "n": int(selected.sum()),
                                **{f"{metric}_with_indicator": value for metric, value in a.items()},
                                **{f"{metric}_without_indicator": value for metric, value in b.items()},
                                **{f"delta_{metric}": a[metric] - b[metric] for metric in a},
                                "mean_paired_delta_mse": float((square[0] - square[1])[selected].mean()),
                                "beers_with_lower_mse_using_indicator": int((square[0][selected] < square[1][selected]).sum())})
    csv("metrics.csv", metric_rows)
    csv("metrics_by_fold.csv", fold_rows)
    csv("paired_metrics.csv", paired_rows)
    csv("paired_errors.csv", error_rows)
    comparison = baseline_comparison(predictions, meta, y, fold_ids)
    csv("baseline_consistency.csv", comparison)
    assert hashes == {p.relative_to(REPO).as_posix(): sha256(p) for p in protected}
    json_file("manifest.json", {
        "purpose": "Post-hoc paired missing-indicator ablation; primary model and results retained",
        "seed": SEED, "model_seed": "SEED + outer fold (1 through 5)",
        "n": len(x), "features": list(x.columns), "targets": list(y.columns),
        "cohorts": {name: int(rows.sum()) for name, rows in groups.items()},
        "sole_configured_difference": "SimpleImputer.add_indicator True versus False",
        "preprocessing": "training-fold median, keep_empty_features=True, no X/Y scaling",
        "tree_parameters": TREE_PARAMETERS, "folds": "five shuffled folds, 200 train / 50 test",
        "mask_logic": "default_rng(SEED + 100 + fold); observed test cells only; same uniforms for both rates and arms",
        "mask_repeats": 1, "cohort_definition": "raw original missingness, fixed before test masking",
        "target_scale": "Attachment 4 standardized continuous values; not original 0-6 scores",
        "paired_difference_direction": "with indicator minus without indicator; negative error delta favors indicator",
        "baseline_consistency": comparison, "input_sha256": hashes,
        "script_sha256": sha256(Path(__file__)), "frozen_inputs_unchanged": True,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "One fixed split seed and one mask draw per rate; not independent external validation or a confidence interval.",
            "Natural-missing versus complete cohorts differ in more than missingness; their contrast is not causal.",
            "Adding indicators changes effective feature dimension and the integer candidate count implied by max_features=0.7; it can change random tree paths even for complete rows.",
            "Indicators cover only columns missing in the training fold, not every newly hidden test column.",
            "Refitted floating-point tree outputs are compared explicitly to frozen outputs; exact identity is not assumed.",
            "MCAR-style test masking does not represent detection-limit censoring, training missingness or instrument failure.",
            "No hyperparameter or target-wise model selection is performed; existing original-0-6 limitations remain."]})
    report = verify_saved()
    json_file("verification.json", report)
    print(pd.DataFrame(paired_rows).to_string(index=False), flush=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def numpy_scores(actual, predicted):
    """Metric recomputation independent of the sklearn functions used above."""
    squared = (actual - predicted) ** 2
    numerator = squared.sum(axis=0)
    denominator = ((actual - actual.mean(axis=0)) ** 2).sum(axis=0)
    ratios = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    r2 = np.where(denominator != 0, 1 - ratios, np.where(numerator == 0, 1., 0.))
    return {"mae": float(np.abs(actual - predicted).mean()),
            "rmse": float(np.sqrt(squared.mean())), "macro_r2": float(r2.mean())}


def verify_saved():
    meta, x, y = load_xy()
    actual, groups = y.to_numpy(), cohorts(x)
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert manifest["script_sha256"] == sha256(Path(__file__))
    assert all(sha256(REPO / path) == value for path, value in manifest["input_sha256"].items())
    saved = pd.read_csv(OUT / "fold_assignments.csv")
    assert saved.beer.tolist() == meta.beer.tolist()
    np.testing.assert_array_equal(saved.original_missing, groups["original_missing"])
    np.testing.assert_array_equal(saved.original_missing_count, x.isna().sum(axis=1))
    stored_masks = np.load(OUT / "test_masks.npz")
    preprocess = json.loads((OUT / "preprocessing.json").read_text())
    counts = pd.read_csv(OUT / "mask_counts.csv")
    for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(x), 1):
        assert saved.fold.iloc[test].eq(fold).all()
        uniforms = np.random.default_rng(SEED + 100 + fold).random((len(test), x.shape[1]))
        for scenario in ["mask10", "mask20"]:
            expected = (uniforms < RATES[scenario]) & x.iloc[test].notna().to_numpy()
            np.testing.assert_array_equal(stored_masks[scenario][test], expected)
            row = counts.loc[counts.fold.eq(fold) & counts.scenario.eq(scenario)].iloc[0]
            assert row.additional_hidden == int(expected.sum())
            assert row.original_observed == int(x.iloc[test].notna().sum().sum())
        for arm, indicator in ARMS.items():
            item = next(r for r in preprocess if r["fold"] == fold and r["arm"] == arm)
            np.testing.assert_allclose(item["median_statistics"], np.nanmedian(x.iloc[train], axis=0))
            indices = np.flatnonzero(x.iloc[train].isna().any(axis=0)).tolist() if indicator else []
            assert item["indicator_indices"] == indices
            assert item["n_model_features"] == x.shape[1] + len(indices)
            assert item["n_candidate_features_per_split"] == int(.7 * item["n_model_features"])
    assert not (stored_masks["mask10"] & ~stored_masks["mask20"]).any()
    metric = pd.read_csv(OUT / "metrics.csv")
    by_fold = pd.read_csv(OUT / "metrics_by_fold.csv")
    pair_metric = pd.read_csv(OUT / "paired_metrics.csv")
    pair_error = pd.read_csv(OUT / "paired_errors.csv")
    predictions = {}
    for arm in ARMS:
        for scenario in RATES:
            frame = pd.read_csv(OUT / f"oof_{arm}_{scenario}.csv")
            assert frame.beer.tolist() == meta.beer.tolist()
            np.testing.assert_array_equal(frame.fold, saved.fold)
            np.testing.assert_array_equal(frame.original_missing, groups["original_missing"])
            predicted = frame[y.columns].to_numpy()
            assert predicted.shape == actual.shape and np.isfinite(predicted).all()
            predictions[(arm, scenario)] = predicted
            for group, selected in groups.items():
                row = metric.loc[metric.arm.eq(arm) & metric.scenario.eq(scenario) & metric.cohort.eq(group)].iloc[0]
                assert row.n == int(selected.sum())
                for key, value in numpy_scores(actual[selected], predicted[selected]).items():
                    np.testing.assert_allclose(row[key], value, rtol=1e-10, atol=1e-12)
                for fold in range(1, 6):
                    subset = selected & saved.fold.eq(fold).to_numpy()
                    row = by_fold.loc[by_fold.arm.eq(arm) & by_fold.scenario.eq(scenario)
                                      & by_fold.cohort.eq(group) & by_fold.fold.eq(fold)].iloc[0]
                    assert row.n == int(subset.sum())
                    for key, value in numpy_scores(actual[subset], predicted[subset]).items():
                        np.testing.assert_allclose(row[key], value, rtol=1e-10, atol=1e-12)
    for scenario in RATES:
        a, b = [predictions[(arm, scenario)] for arm in ARMS]
        delta = ((a - actual) ** 2 - (b - actual) ** 2).mean(axis=1)
        rows = pair_error.loc[pair_error.scenario.eq(scenario)]
        assert rows.beer.tolist() == meta.beer.tolist()
        np.testing.assert_allclose(rows.delta_mse, delta, rtol=1e-10, atol=1e-12)
        for group, selected in groups.items():
            row = pair_metric.loc[pair_metric.scenario.eq(scenario) & pair_metric.cohort.eq(group)].iloc[0]
            sa, sb = [numpy_scores(actual[selected], pred[selected]) for pred in [a, b]]
            for key in sa:
                for field, value in [(f"{key}_with_indicator", sa[key]),
                                     (f"{key}_without_indicator", sb[key]),
                                     (f"delta_{key}", sa[key] - sb[key])]:
                    np.testing.assert_allclose(row[field], value, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(row.mean_paired_delta_mse, delta[selected].mean(), rtol=1e-10, atol=1e-12)
            assert row.beers_with_lower_mse_using_indicator == int((delta[selected] < 0).sum())
    comparison = baseline_comparison(predictions, meta, y, saved.fold.to_numpy())
    pd.testing.assert_frame_equal(pd.DataFrame(comparison),
                                  pd.DataFrame(manifest["baseline_consistency"]),
                                  check_exact=False, rtol=1e-10, atol=1e-12)
    return {"status": "PASS", "oof_tables": len(predictions), "metric_rows": len(metric),
            "fold_metric_rows": len(by_fold), "paired_metric_rows": len(pair_metric),
            "paired_beer_scenarios": len(pair_error), "folds": 5, "test_mask_scenarios": 2,
            "protected_files_unchanged": len(manifest["input_sha256"]),
            "baseline_consistency": comparison,
            "scope": "Read-only recomputation of saved numerical results and data contracts; no second model fit."}


if __name__ == "__main__":
    if sys.argv[1:] == ["--verify-only"]:
        print(json.dumps(verify_saved(), ensure_ascii=False, indent=2))
    else:
        assert not sys.argv[1:], "Only --verify-only is supported"
        with threadpool_limits(limits=1):
            main()
