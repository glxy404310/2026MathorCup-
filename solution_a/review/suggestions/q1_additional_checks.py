"""Supplement the exploratory comparison with balanced groups and a NaN probe.

The shuffled group split had markedly unequal sample counts. This balanced
split was added after noticing that issue; both sets of results are retained.
"""
import json

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

import q1_comparison as c


def main():
    x_small = np.array([[1., 2.], [2., np.nan], [np.nan, 4.], [4., 5.],
                        [5., 6.], [6., 7.], [7., 8.], [8., 9.]])
    y_small = np.arange(1., 9.)
    checks = {}
    for model in [GradientBoostingRegressor(n_estimators=3, random_state=1),
                  HistGradientBoostingRegressor(max_iter=3, min_samples_leaf=2,
                                                early_stopping=False, random_state=1),
                  ExtraTreesRegressor(n_estimators=3, random_state=1)]:
        name = type(model).__name__
        try:
            model.fit(x_small, y_small)
            pred = model.predict(np.array([[np.nan, 3.], [3., np.nan]]))
            checks[name] = {"fit_success": True, "predict_nan_success": bool(np.isfinite(pred).all()),
                            "predictions": pred.tolist()}
        except Exception as error:
            checks[name] = {"fit_success": False, "exception_type": type(error).__name__,
                            "message": str(error)}
    (c.OUT / "nan_support_check.json").write_text(json.dumps(
        {"sklearn": sklearn.__version__, "sample_n": 8, "checks": checks},
        ensure_ascii=False, indent=2), encoding="utf-8")
    meta, x, y = c.load_xy()
    rows, per_target = [], []
    predictions = {name: np.full(y.shape, np.nan) for name in ["mean", "extra_trees"]}
    fold_ids = np.zeros(len(x), dtype=int)
    for fold, (train, test) in enumerate(GroupKFold(3, shuffle=False).split(
            x, groups=meta.tasting_category_fine), 1):
        fold_ids[test] = fold
        assert not set(meta.iloc[train].tasting_category_fine) & set(meta.iloc[test].tasting_category_fine)
        for name in predictions:
            model = c.estimator(name, c.SEED + fold).fit(x.iloc[train], y.iloc[train])
            pred = model.predict(x.iloc[test])
            predictions[name][test] = pred
            rows.append({"protocol": "unseen_style_balanced_3fold", "model": name, "fold": fold,
                         "train_n": len(train), "test_n": len(test), **c.scores(y.iloc[test], pred)})
    overall = []
    for name, pred in predictions.items():
        overall.append({"protocol": "unseen_style_balanced_3fold", "model": name, **c.scores(y, pred)})
        for j, target in enumerate(y.columns):
            per_target.append({"model": name, "target": target, **c.scores(y.iloc[:, j], pred[:, j])})
        pd.concat([meta, pd.DataFrame({"fold": fold_ids}), pd.DataFrame(pred, columns=y.columns)], axis=1).to_csv(
            c.OUT / f"oof_unseen_style_balanced_3fold_{name}.csv", index=False, lineterminator="\n")
    for name, values in [("by_fold", rows), ("overall", overall), ("by_target", per_target)]:
        pd.DataFrame(values).to_csv(c.OUT / f"balanced_group_metrics_{name}.csv", index=False, lineterminator="\n")
    print(pd.DataFrame(overall).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
