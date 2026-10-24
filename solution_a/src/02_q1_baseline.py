"""Leakage-controlled prediction of 45 *standardized* sensory dimensions.

This is a starting baseline, NOT a recovered 0-6 sensory score. The attachment
does not provide the per-panelist inverse calibration needed for that recovery.
"""
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common import SEED, SOLUTION, load_xy

OUT = SOLUTION / "outputs" / "q1"
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]


def ridge_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True,
                                  keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", TransformedTargetRegressor(
            regressor=Ridge(solver="svd"), transformer=StandardScaler())),
    ])


def make_search(seed):
    # Scaling/imputation, including y scaling, are refit in EVERY inner fold.
    return GridSearchCV(
        ridge_pipeline(), {"model__regressor__alpha": ALPHAS},
        cv=KFold(n_splits=4, shuffle=True, random_state=seed),
        scoring="neg_mean_squared_error", n_jobs=1, refit=True,
        error_score="raise",
    )


def tree_pipeline(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True,
                                  keep_empty_features=True)),
        ("model", ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.7,
            random_state=seed, n_jobs=1)),
    ])


def scores(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "macro_r2": float(r2_score(y, pred, multioutput="uniform_average")),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, x, y = load_xy()
    targets = y.columns.tolist()
    # Prespecified nested five-fold baseline. IDs and styles are not predictors.
    outer = KFold(n_splits=5, shuffle=True, random_state=SEED)
    predictions = {name: np.full(y.shape, np.nan) for name in
                   ("mean", "ridge", "extra_trees", "ridge_mask10", "ridge_mask20",
                    "extra_trees_mask10", "extra_trees_mask20")}
    fold_ids = np.zeros(len(x), dtype=int)
    folds = []
    stress_counts = []
    for fold, (train, test) in enumerate(outer.split(x), start=1):
        fold_ids[test] = fold
        xtr, xte, ytr, yte = x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test]
        search = make_search(SEED + fold)
        search.fit(xtr, ytr)
        candidates = {
            "ridge": search.best_estimator_,
            "mean": DummyRegressor(strategy="mean").fit(xtr, ytr),
            # A fixed nonlinear comparison, not tuned on outer test outcomes.
            "extra_trees": tree_pipeline(SEED + fold).fit(xtr, ytr),
        }
        for name, model in candidates.items():
            pred = model.predict(xte)
            predictions[name][test] = pred
            folds.append({"fold": fold, "model": name, "train_n": len(train),
                          "test_n": len(test), **scores(yte, pred),
                          "ridge_alpha": search.best_params_["model__regressor__alpha"]
                          if name == "ridge" else None})
        # Deployment stress: randomly hide observed TEST covariates only.
        # Identical random uniforms make 10% masks a subset of the 20% masks.
        uniforms = np.random.default_rng(SEED + 100 + fold).random(xte.shape)
        for rate, suffix in [(0.1, "mask10"), (0.2, "mask20")]:
            mask = (uniforms < rate) & xte.notna().to_numpy()
            hidden = xte.mask(mask)
            stress_counts.append({"fold": fold, "rate": rate,
                                  "additional_hidden": int(mask.sum()),
                                  "original_observed": int(xte.notna().sum().sum())})
            for base in ("ridge", "extra_trees"):
                name = f"{base}_{suffix}"
                pred = candidates[base].predict(hidden)
                predictions[name][test] = pred
                folds.append({"fold": fold, "model": name, "train_n": len(train),
                              "test_n": len(test), **scores(yte, pred),
                              "ridge_alpha": search.best_params_["model__regressor__alpha"]
                              if base == "ridge" else None})
        print(f"Fold {fold}/5: alpha={search.best_params_['model__regressor__alpha']}, "
              f"ridge RMSE={scores(yte, predictions['ridge'][test])['rmse']:.4f}", flush=True)

    metric_rows, target_rows = [], []
    truth = y.to_numpy()
    for name, pred in predictions.items():
        if not np.isfinite(pred).all():
            raise AssertionError(f"Incomplete predictions: {name}")
        metric_rows.append({"model": name, **scores(truth, pred)})
        for j, target in enumerate(targets):
            target_rows.append({"model": name, "target": target,
                                **scores(truth[:, j], pred[:, j]),
                                "spearman": float(spearmanr(truth[:, j], pred[:, j]).statistic)})
        table = pd.concat([meta, pd.DataFrame({"fold": fold_ids}),
                           pd.DataFrame(pred, columns=targets)], axis=1)
        table.to_csv(OUT / f"oof_{name}.csv", index=False, lineterminator="\n")
    pd.DataFrame(metric_rows).to_csv(OUT / "metrics_overall.csv", index=False, lineterminator="\n")
    pd.DataFrame(target_rows).to_csv(OUT / "metrics_by_target.csv", index=False, lineterminator="\n")
    pd.DataFrame(folds).to_csv(OUT / "metrics_by_fold.csv", index=False, lineterminator="\n")
    pd.DataFrame(stress_counts).to_csv(OUT / "missingness_stress.csv", index=False, lineterminator="\n")

    # Full-data model is for later synthesis/inference; fitted values are not a test.
    final = make_search(SEED + 99)
    final.fit(x, y)
    joblib.dump({"pipeline": final.best_estimator_, "features": x.columns.tolist(),
                 "targets": targets, "target_scale": "attachment_4_standardized"},
                OUT / "ridge_model.joblib")
    fitted = final.predict(x)
    pd.concat([meta, pd.DataFrame(fitted, columns=targets)], axis=1).to_csv(
        OUT / "fitted_standardized_NOT_validation.csv", index=False, lineterminator="\n")
    pd.concat([meta, y], axis=1).to_csv(OUT / "observed_standardized.csv", index=False, lineterminator="\n")
    tree = tree_pipeline(SEED + 99).fit(x, y)
    joblib.dump({"pipeline": tree, "features": x.columns.tolist(), "targets": targets,
                 "target_scale": "attachment_4_standardized"}, OUT / "extra_trees_model.joblib")
    summary = {
        "seed": SEED, "samples": len(x), "features": x.shape[1], "targets": targets,
        "validation": "5 outer shuffled folds, 4 inner folds; one seed; no style/ID covariates",
        "unit": "attachment 4 standardized sensory values; NOT original 0-6 scores",
        "final_ridge_alpha": final.best_params_["model__regressor__alpha"],
        "alpha_grid": ALPHAS, "metrics": metric_rows,
        "limitations": [
            "Original 0-6 scale is not identifiable from blank cards and panelist-normalized means.",
            "Random CV estimates new beers in the existing style mixture, not unseen-style transfer.",
            "Global preprocessing used by attachment creators cannot be undone; metrics are conditional on supplied data.",
            "Missingness stress is synthetic random masking, not evidence about detection-limit censoring.",
            "One split seed is a baseline, not completed robustness or uncertainty validation.",
        ],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(pd.DataFrame(metric_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
