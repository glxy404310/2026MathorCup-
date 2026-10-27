"""Exploratory Q1 checks prompted by the supplied suggestions.

This script writes the supplementary Q1 comparisons. It does not
replace the frozen experiment, choose a model for submission, or recover
original 0-6 sensory labels. Parameters are fixed before this script is run.
"""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLUTION = HERE.parent
sys.path.insert(0, str(SOLUTION / "src"))

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from common import data_dir, load_xy

SEED = 20261024
OUT = HERE / "results/q1"


def scores(actual, predicted):
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "macro_r2": float(r2_score(actual, predicted, multioutput="uniform_average"))}


def estimator(name, seed):
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "extra_trees":
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                                    max_features=.7, random_state=seed, n_jobs=1)
    elif name == "gbr_per_target":
        model = MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=100, learning_rate=.05, max_depth=2, min_samples_leaf=5,
            loss="squared_error", random_state=seed), n_jobs=1)
    elif name == "hgb_per_target":
        model = MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=100, learning_rate=.05, max_leaf_nodes=7, min_samples_leaf=10,
            l2_regularization=1., early_stopping=False, random_state=seed), n_jobs=1)
    else:
        raise ValueError(name)
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True,
                                                keep_empty_features=True)), ("model", model)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, x, y = load_xy()
    y_np = y.to_numpy()
    random_splits = list(KFold(3, shuffle=True, random_state=SEED).split(x))
    group_splits = list(GroupKFold(3, shuffle=True, random_state=SEED).split(
        x, groups=meta.tasting_category_fine))
    summary_rows, fold_rows, target_rows, masks = [], [], [], []
    started = time.perf_counter()
    for protocol, splits, names in [
        ("random_3fold", random_splits, ["mean", "extra_trees", "gbr_per_target", "hgb_per_target"]),
        ("unseen_style_3fold", group_splits, ["mean", "extra_trees"]),
    ]:
        fold_ids = np.zeros(len(x), dtype=int)
        predictions = {name: np.full(y.shape, np.nan) for name in names}
        stressed = {rate: np.full((20, *y.shape), np.nan) for rate in (.1, .2)} if protocol == "random_3fold" else {}
        timings = {name: 0. for name in names}
        for fold, (train, test) in enumerate(splits, 1):
            fold_ids[test] = fold
            if protocol == "unseen_style_3fold":
                assert not set(meta.iloc[train].tasting_category_fine) & set(meta.iloc[test].tasting_category_fine)
            for name in names:
                start = time.perf_counter()
                model = estimator(name, SEED + fold)
                model.fit(x.iloc[train], y.iloc[train])
                pred = model.predict(x.iloc[test])
                timings[name] += time.perf_counter() - start
                predictions[name][test] = pred
                if name != "mean":
                    np.testing.assert_allclose(model.named_steps["imputer"].statistics_,
                                               np.nanmedian(x.iloc[train], axis=0))
                fold_rows.append({"protocol": protocol, "model": name, "fold": fold,
                                  "train_n": len(train), "test_n": len(test),
                                  "test_styles": "|".join(sorted(meta.iloc[test].tasting_category_fine.unique())),
                                  **scores(y.iloc[test], pred)})
                if protocol == "random_3fold" and name == "extra_trees":
                    for repeat in range(20):
                        uniforms = np.random.default_rng(SEED + 10000 + 100 * repeat + fold).random((len(test), x.shape[1]))
                        for rate in (.1, .2):
                            mask = (uniforms < rate) & x.iloc[test].notna().to_numpy()
                            stressed[rate][repeat, test] = model.predict(x.iloc[test].mask(mask))
                            masks.append({"repeat": repeat + 1, "fold": fold, "rate": rate,
                                          "additional_hidden": int(mask.sum())})
                print(f"{protocol} fold {fold}/3 {name}: RMSE={scores(y.iloc[test], pred)['rmse']:.6f}", flush=True)
        for name, pred in predictions.items():
            assert np.isfinite(pred).all()
            summary_rows.append({"protocol": protocol, "model": name,
                                 "fit_and_predict_seconds": timings[name], **scores(y, pred)})
            for j, target in enumerate(y.columns):
                target_rows.append({"protocol": protocol, "model": name,
                                    "target": target, **scores(y_np[:, j], pred[:, j])})
            pd.concat([meta, pd.DataFrame({"fold": fold_ids}), pd.DataFrame(pred, columns=y.columns)], axis=1).to_csv(
                OUT / f"oof_{protocol}_{name}.csv", index=False, lineterminator="\n")
        pd.concat([meta, pd.DataFrame({"fold": fold_ids})], axis=1).to_csv(
            OUT / f"fold_assignment_{protocol}.csv", index=False, lineterminator="\n")
        if stressed:
            np.savez_compressed(OUT / "test_mask_predictions.npz",
                                rate10=stressed[.1], rate20=stressed[.2])
        for rate, values in stressed.items():
            for repeat in range(20):
                summary_rows.append({"protocol": "random_3fold_test_mask", "model": "extra_trees",
                                     "rate": rate, "repeat": repeat + 1, **scores(y, values[repeat])})
    for filename, rows in [("metrics_overall.csv", summary_rows), ("metrics_by_fold.csv", fold_rows),
                           ("metrics_by_target.csv", target_rows), ("mask_counts.csv", masks)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, lineterminator="\n")
    manifest = {"purpose": "Supplementary post-hoc comparison; primary experiment retained", "seed": SEED,
                "samples": len(x), "features": x.shape[1], "targets": list(y.columns),
                "target_scale": "Attachment 4 standardized continuous scale, not original 0-6 scores",
                "total_seconds": time.perf_counter() - started,
                "protocols": {"random_3fold": "Same 3 shuffled folds and training-only imputation for all four models",
                              "unseen_style_3fold": "Whole fine styles excluded from training; Mean and ExtraTrees only",
                              "stress": "20 test-mask draws per 10%/20% rate; masks are paired and nested; models remain unchanged"},
                "models": {name: repr(estimator(name, SEED + 1)) for name in ["mean", "extra_trees", "gbr_per_target", "hgb_per_target"]},
                "input_sha256": {filename: hashlib.sha256((data_dir()/filename).read_bytes()).hexdigest() for filename in ["1.csv", "4.csv"]},
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__},
                "limitations": ["Same 250 beers as development; a new split is not an independent data set.",
                                "One fixed parameter set per model; not a tuned-family superiority claim.",
                                "45-output ExtraTrees shares splits; boosting models fit each target separately.",
                                "Random and style-group protocols answer different deployment questions.",
                                "MCAR test masking does not model detection limits, training-set missingness, or instrument failures.",
                                "Attachment creators' precomputed global transformations cannot be undone."]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary_rows).head(6).to_string(index=False), flush=True)
    print(f"Completed in {manifest['total_seconds']:.2f} seconds", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
