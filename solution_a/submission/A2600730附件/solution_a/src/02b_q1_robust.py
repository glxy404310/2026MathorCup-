"""Exploratory robust Ridge follow-up after diagnosing extreme covariates.

Clipping bounds, imputation, X/Y scaling and alpha selection all stay inside
the training folds. The original split is reused: this is a robustness check,
not an untouched confirmatory test. No samples or targets are deleted.
"""
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from common import QuantileClipper, SEED, SOLUTION, load_xy

spec = importlib.util.spec_from_file_location("baseline", Path(__file__).with_name("02_q1_baseline.py"))
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
OUT = SOLUTION / "outputs" / "q1_robust"


def search(seed):
    plain = baseline.ridge_pipeline()
    model = Pipeline([("clip", QuantileClipper()), *plain.steps])
    return GridSearchCV(
        model, {"model__regressor__alpha": baseline.ALPHAS},
        cv=KFold(n_splits=4, shuffle=True, random_state=seed),
        scoring="neg_mean_squared_error", n_jobs=1, error_score="raise",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, x, y = load_xy()
    pred = {name: np.full(y.shape, np.nan) for name in
            ("ridge_winsor", "ridge_winsor_mask10", "ridge_winsor_mask20")}
    fold_ids = np.zeros(len(x), dtype=int)
    fold_metrics = []
    clipping = []
    for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(x), 1):
        fold_ids[test] = fold
        fitted = search(SEED + fold).fit(x.iloc[train], y.iloc[train])
        alpha = fitted.best_params_["model__regressor__alpha"]
        xte = x.iloc[test]
        bounds = fitted.best_estimator_.named_steps["clip"].bounds_
        counts = ((xte.to_numpy() < bounds[0]) | (xte.to_numpy() > bounds[1])).sum(axis=0)
        clipping.extend({"fold": fold, "feature": feature, "test_values_clipped": int(n),
                         "train_lower_01": float(bounds[0, j]), "train_upper_99": float(bounds[1, j])}
                        for j, (feature, n) in enumerate(zip(x.columns, counts)))
        uniforms = np.random.default_rng(SEED + 100 + fold).random(xte.shape)
        for rate, name in [(0, "ridge_winsor"), (.1, "ridge_winsor_mask10"), (.2, "ridge_winsor_mask20")]:
            test_input = xte.mask((uniforms < rate) & xte.notna().to_numpy())
            pred[name][test] = fitted.predict(test_input)
            fold_metrics.append({"fold": fold, "model": name, "ridge_alpha": alpha,
                                 **baseline.scores(y.iloc[test], pred[name][test])})
        print(f"Robust fold {fold}/5: alpha={alpha}", flush=True)
    metrics, by_target = [], []
    for name, values in pred.items():
        assert np.isfinite(values).all()
        metrics.append({"model": name, **baseline.scores(y, values)})
        for j, target in enumerate(y.columns):
            by_target.append({"model": name, "target": target,
                              **baseline.scores(y.iloc[:, j], values[:, j]),
                              "spearman": float(spearmanr(y.iloc[:, j], values[:, j]).statistic)})
        pd.concat([meta, pd.DataFrame({"fold": fold_ids}), pd.DataFrame(values, columns=y.columns)], axis=1).to_csv(
            OUT / f"oof_{name}.csv", index=False, lineterminator="\n")
    pd.DataFrame(metrics).to_csv(OUT / "metrics_overall.csv", index=False, lineterminator="\n")
    pd.DataFrame(by_target).to_csv(OUT / "metrics_by_target.csv", index=False, lineterminator="\n")
    pd.DataFrame(fold_metrics).to_csv(OUT / "metrics_by_fold.csv", index=False, lineterminator="\n")
    pd.DataFrame(clipping).to_csv(OUT / "clipping_bounds.csv", index=False, lineterminator="\n")
    final = search(SEED + 99).fit(x, y)
    joblib.dump({"pipeline": final.best_estimator_, "features": x.columns.tolist(),
                 "targets": y.columns.tolist(), "target_scale": "attachment_4_standardized"},
                OUT / "ridge_winsor_model.joblib")
    summary = {
        "purpose": "Exploratory robustness follow-up; original five folds reused after anomaly diagnosis",
        "seed": SEED, "train_quantiles": [.01, .99], "samples_retained": len(x),
        "metrics": metrics, "final_alpha": final.best_params_["model__regressor__alpha"],
        "limitations": ["Not a fresh holdout test", "Fixed 1%/99% clipping can suppress genuine extremes",
                        "Original 0-6 scores still not identified", "One split seed only"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
