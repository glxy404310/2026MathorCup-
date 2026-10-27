"""Style-held-out stale-hops candidates with chemical-only prediction inputs."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from q3_stale_optimized import FoldChemicalTransform

BIAS_PENALTY = 0.25
VARIANCE_RATIO_MIN = 0.05
LOSO_SEED = 20270123
FINAL_SEED = 20271123
SELECTION_BOOTSTRAP_SEED = 20272123
AUDIT_BOOTSTRAP_SEED = 20273123


def style_balanced_weights(styles):
    """Each observed training style receives equal total regression weight."""
    values = np.asarray(styles)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return len(values) / (len(counts) * counts[inverse])


class CenteredKernelRidge(RegressorMixin, BaseEstimator):
    """RBF ridge with a training-only weighted target mean for far points."""
    def __init__(self, alpha=1.0, gamma_factor=0.1):
        self.alpha = alpha
        self.gamma_factor = gamma_factor

    def fit(self, x, y, sample_weight=None):
        values = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        self.target_mean_ = float(np.average(target, weights=sample_weight))
        self.n_features_in_ = values.shape[1]
        self.model_ = KernelRidge(alpha=self.alpha, kernel="rbf",
                                  gamma=self.gamma_factor / self.n_features_in_)
        self.model_.fit(values, target - self.target_mean_, sample_weight=sample_weight)
        return self

    def predict(self, x):
        return self.target_mean_ + self.model_.predict(np.asarray(x, dtype=float))


class StyleWeightedLocalMean(RegressorMixin, BaseEstimator):
    """Local chemical neighbours, inverse distance times training-style weight."""
    def __init__(self, n_neighbors=25):
        self.n_neighbors = n_neighbors

    def fit(self, x, y, sample_weight=None):
        self.x_ = np.asarray(x, dtype=float)
        self.y_ = np.asarray(y, dtype=float).reshape(-1)
        self.sample_weight_ = np.ones(len(self.y_)) if sample_weight is None else np.asarray(sample_weight)
        self.n_features_in_ = self.x_.shape[1]
        return self

    def predict(self, x):
        values = np.asarray(x, dtype=float)
        result = np.empty(len(values))
        k = min(self.n_neighbors, len(self.y_))
        for start in range(0, len(values), 2048):
            distances = cdist(values[start:start+2048], self.x_)
            for local, row in enumerate(distances):
                exact = row <= 1e-12
                if exact.any():
                    result[start + local] = np.average(self.y_[exact], weights=self.sample_weight_[exact])
                else:
                    indices = np.argpartition(row, k-1)[:k]
                    weights = self.sample_weight_[indices] / np.maximum(row[indices], 1e-12)
                    result[start + local] = np.average(self.y_[indices], weights=weights)
        return result


class EqualDomainEnsemble(RegressorMixin, BaseEstimator):
    """A fixed equal-weight average; no stacking or fitted mixture weights."""
    def __init__(self, features, calibrated_features, aging_features, member_specs, seed):
        self.features = features
        self.calibrated_features = calibrated_features
        self.aging_features = aging_features
        self.member_specs = member_specs
        self.seed = seed

    def fit(self, x, y, sample_weight=None):
        self.models_ = []
        for spec in self.member_specs:
            model = make_domain_pipeline(spec, self.features, self.calibrated_features,
                                         self.aging_features, self.seed)
            params = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
            model.fit(x, y, **params)
            self.models_.append(model)
        self.n_features_in_ = len(self.features)
        return self

    def predict(self, x):
        return np.mean([np.asarray(model.predict(x)).reshape(-1) for model in self.models_], axis=0)


def domain_candidate_specs():
    """Freeze 14 comparisons before any new LOSO or Pils diagnostic results."""
    specs = [
        {"name": "mean_global", "family": "mean", "balanced": False, "selectable": False},
        {"name": "mean_style_balanced", "family": "mean", "balanced": True, "selectable": False},
        {"name": "old_multioutput_tree", "family": "old_multioutput", "balanced": False, "selectable": False},
        {"name": "fixed_single_tree", "family": "tree", "balanced": False, "selectable": True},
        {"name": "style_balanced_tree", "family": "tree", "balanced": True, "selectable": True},
        {"name": "ridge_all_balanced_100", "family": "ridge", "feature_set": "all226", "alpha": 100.0, "balanced": True, "selectable": True},
        {"name": "ridge_all_balanced_1000", "family": "ridge", "feature_set": "all226", "alpha": 1000.0, "balanced": True, "selectable": True},
        {"name": "ridge_aging_balanced_10", "family": "ridge", "feature_set": "aging", "alpha": 10.0, "balanced": True, "selectable": True},
        {"name": "ridge_aging_balanced_100", "family": "ridge", "feature_set": "aging", "alpha": 100.0, "balanced": True, "selectable": True},
        {"name": "krr_aging_balanced_g01", "family": "krr", "feature_set": "aging", "alpha": 1.0, "gamma_factor": 0.1, "balanced": True, "selectable": True},
        {"name": "krr_aging_balanced_g1", "family": "krr", "feature_set": "aging", "alpha": 1.0, "gamma_factor": 1.0, "balanced": True, "selectable": True},
        {"name": "svr_aging_balanced_C1", "family": "svr", "feature_set": "aging", "C": 1.0, "epsilon": 0.05, "balanced": True, "selectable": True},
        {"name": "local_aging_pca8_k25", "family": "local", "feature_set": "aging", "n_components": 8, "n_neighbors": 25, "balanced": True, "selectable": True},
    ]
    ensemble_names = ["style_balanced_tree", "ridge_aging_balanced_100", "krr_aging_balanced_g01"]
    specs.append({"name": "ensemble_equal3", "family": "ensemble", "balanced": True,
                  "selectable": True, "weights": [1/3, 1/3, 1/3],
                  "members": [dict(spec) for spec in specs if spec["name"] in ensemble_names]})
    return specs


def make_domain_pipeline(spec, features, calibrated, aging, seed):
    family = spec["family"]
    if family == "ensemble":
        return Pipeline([("model", EqualDomainEnsemble(features, calibrated, aging, spec["members"], seed))])
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if family in ("tree", "mean", "old_multioutput"):
        regressor = DummyRegressor(strategy="mean") if family == "mean" else ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.7, random_state=seed, n_jobs=1)
        return Pipeline([("imputer", imputer), ("model", regressor)])
    selected = aging if spec["feature_set"] == "aging" else None
    steps = [("chemical", FoldChemicalTransform(features, calibrated, selected)),
             ("imputer", imputer), ("scale", StandardScaler())]
    if family == "ridge":
        regressor = Ridge(alpha=spec["alpha"], solver="svd")
    elif family == "krr":
        regressor = CenteredKernelRidge(alpha=spec["alpha"], gamma_factor=spec["gamma_factor"])
    elif family == "svr":
        regressor = SVR(kernel="rbf", C=spec["C"], epsilon=spec["epsilon"], gamma="scale")
    elif family == "local":
        steps.append(("pca", PCA(n_components=spec["n_components"], svd_solver="full")))
        regressor = StyleWeightedLocalMean(n_neighbors=spec["n_neighbors"])
    else:
        raise ValueError(f"Unknown domain candidate: {family}")
    return Pipeline([*steps, ("model", regressor)])


def training_fit_params(spec, styles):
    return {"model__sample_weight": style_balanced_weights(styles)} if spec["balanced"] else {}
