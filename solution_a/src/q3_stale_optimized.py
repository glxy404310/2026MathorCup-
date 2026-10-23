"""Fold-local chemical preprocessing and single-target stale-hops models."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

VARIANCE_RATIO_MIN = 0.05
OUTER_SEEDS = (20261123, 20262123)
TRANSFER_CV_SEED = 20263123
SELECTION_BOOTSTRAP_SEED = 20264123
AUDIT_BOOTSTRAP_SEED = 20265123


class FoldChemicalTransform(TransformerMixin, BaseEstimator):
    """Select predeclared chemical fields, log calibrated values, then clip.

    Positive-median scaling, 1%/99% clipping bounds and all downstream
    imputation/scaling are learned anew in each training subset. Relative
    representations are not exponentiated or treated as concentrations.
    """

    def __init__(self, features, calibrated_features, selected_features=None):
        self.features = features
        self.calibrated_features = calibrated_features
        self.selected_features = selected_features

    def _values(self, x):
        raw = x.loc[:, self.features].to_numpy(dtype=float) if hasattr(x, "loc") else np.asarray(x, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != len(self.features):
            raise ValueError("Expected the declared 226-feature hybrid representation")
        return raw[:, self.positions_].copy()

    def _log(self, values):
        for position in self.calibrated_positions_:
            column = values[:, position]
            if np.any(column[np.isfinite(column)] < 0):
                raise ValueError("Calibrated concentrations/indicators must be nonnegative")
            values[:, position] = np.log1p(column / self.positive_scales_[position])
        return values

    def fit(self, x, y=None):
        selected = self.features if self.selected_features is None else self.selected_features
        self.positions_ = np.asarray([self.features.index(feature) for feature in selected])
        self.selected_features_ = list(selected)
        self.calibrated_positions_ = [i for i, feature in enumerate(selected)
                                      if feature in self.calibrated_features]
        values = self._values(x)
        self.positive_scales_ = np.ones(values.shape[1])
        for position in self.calibrated_positions_:
            positive = values[:, position][np.isfinite(values[:, position]) & (values[:, position] > 0)]
            self.positive_scales_[position] = float(np.median(positive)) if len(positive) else 1.0
        values = self._log(values)
        self.bounds_ = np.nanquantile(values, [0.01, 0.99], axis=0)
        self.n_features_in_ = len(self.features)
        return self

    def transform(self, x):
        return np.clip(self._log(self._values(x)), self.bounds_[0], self.bounds_[1])


def candidate_specs() -> list[dict]:
    """Small grid declared before inspecting any new outer-fold outcomes."""
    specs = []
    def add(family, label, **params):
        specs.append({"name": f"{family}_{label}", "family": family,
                      "feature_set": "all226", "params": params})
    for alpha in (10.0, 100.0, 1000.0):
        add("ridge", f"alpha{alpha:g}", alpha=alpha)
    for alpha in (0.01, 0.1):
        add("elastic_net", f"alpha{alpha:g}", alpha=alpha, l1_ratio=0.5)
    for n_components in (3, 8):
        add("pls", f"n{n_components}", n_components=n_components)
    for c in (0.3, 1.0, 3.0):
        add("svr", f"C{c:g}", C=c, epsilon=0.05, gamma="scale")
    for leaf in (3, 8):
        add("extra_trees", f"leaf{leaf}", n_estimators=160,
            min_samples_leaf=leaf, max_features=0.7)
        add("random_forest", f"leaf{leaf}", n_estimators=160,
            min_samples_leaf=leaf, max_features=0.7)
    for loss in ("squared_error", "huber"):
        add("gradient_boosting", loss, loss=loss, n_estimators=150,
            learning_rate=0.03, max_depth=2, min_samples_leaf=5)
    specs.append({"name": "ridge_aging_alpha10", "family": "ridge", "feature_set": "aging",
                  "params": {"alpha": 10.0}})
    specs.append({"name": "svr_aging_C1", "family": "svr", "feature_set": "aging",
                  "params": {"C": 1.0, "epsilon": 0.05, "gamma": "scale"}})
    return specs


def make_pipeline(spec, features, calibrated_features, aging_features, seed):
    selected = aging_features if spec["feature_set"] == "aging" else None
    family, params = spec["family"], dict(spec["params"])
    if family == "ridge":
        estimator = Ridge(solver="svd", **params)
    elif family == "elastic_net":
        estimator = ElasticNet(max_iter=20000, tol=1e-5, random_state=seed, **params)
    elif family == "pls":
        estimator = PLSRegression(scale=False, max_iter=1000, **params)
    elif family == "svr":
        estimator = SVR(**params)
    elif family == "extra_trees":
        estimator = ExtraTreesRegressor(random_state=seed, n_jobs=1, **params)
    elif family == "random_forest":
        estimator = RandomForestRegressor(random_state=seed, n_jobs=1, **params)
    elif family == "gradient_boosting":
        estimator = GradientBoostingRegressor(random_state=seed, **params)
    else:
        raise ValueError(f"Unsupported family: {family}")
    return Pipeline([
        ("chemical", FoldChemicalTransform(features, calibrated_features, selected)),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def raw_tree_pipeline(seed, n_estimators=300):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("model", ExtraTreesRegressor(n_estimators=n_estimators, min_samples_leaf=3,
                                      max_features=0.7, random_state=seed, n_jobs=1)),
    ])


def predict_1d(model, x):
    return np.asarray(model.predict(x), dtype=float).reshape(-1)


def scores(y, pred):
    truth, prediction = np.asarray(y, dtype=float).reshape(-1), np.asarray(pred, dtype=float).reshape(-1)
    truth_var, pred_var = float(np.var(truth)), float(np.var(prediction))
    correlation = float(spearmanr(truth, prediction).statistic) if pred_var > 1e-20 else None
    target_range = float(np.quantile(truth, .95) - np.quantile(truth, .05))
    prediction_range = float(np.quantile(prediction, .95) - np.quantile(prediction, .05))
    return {"mae": float(mean_absolute_error(truth, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
            "r2": float(r2_score(truth, prediction)), "spearman": correlation,
            "prediction_variance": pred_var, "target_variance": truth_var,
            "prediction_variance_ratio": pred_var / truth_var if truth_var > 0 else None,
            "prediction_std_ratio": float(np.sqrt(pred_var / truth_var)) if truth_var > 0 else None,
            "prediction_range90": prediction_range, "target_range90": target_range,
            "prediction_range90_ratio": prediction_range / target_range if target_range > 0 else None,
            "mean_bias": float(np.mean(prediction - truth))}


def inner_compare(x, y, specs, features, calibrated, aging, seed):
    """Training-only OOF comparison, with a prespecified nonconstant gate."""
    rows, predictions = [], {}
    splits = list(KFold(3, shuffle=True, random_state=seed).split(x))
    for spec in specs:
        pred = np.full(len(x), np.nan)
        for fold, (train, test) in enumerate(splits, 1):
            model = make_pipeline(spec, features, calibrated, aging, seed + fold)
            model.fit(x.iloc[train], y.iloc[train])
            pred[test] = predict_1d(model, x.iloc[test])
        if not np.isfinite(pred).all():
            raise AssertionError(f"Nonfinite inner predictions: {spec['name']}")
        metrics = scores(y, pred)
        rows.append({"candidate": spec["name"], "family": spec["family"],
                     "feature_set": spec["feature_set"], **metrics,
                     "nonconstant_qualified": metrics["prediction_variance_ratio"] >= VARIANCE_RATIO_MIN})
        predictions[spec["name"]] = pred
    return rows, predictions


def choose_candidate(rows, family=None):
    choices = [row for row in rows if family is None or row["family"] == family]
    eligible = [row for row in choices if row["nonconstant_qualified"]]
    # A failed gate never disappears: callers persist nonconstant_qualified.
    return min(eligible or choices, key=lambda row: (row["rmse"], row["candidate"]))
