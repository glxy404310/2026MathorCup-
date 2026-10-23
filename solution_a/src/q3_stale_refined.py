"""Smooth chemical-only stale prediction with training-only auxiliary targets."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

TARGET = "stale_hops"
BIAS_PENALTY = 0.25
VARIANCE_RATIO_MIN = 0.05
STABILITY_DRAWS = 8
STABILITY_SEED = 20280123
NESTED_SEED = 20281123
FINAL_SEED = 20282123
SELECTION_BOOTSTRAP_SEED = 20283123
AUDIT_BOOTSTRAP_SEED = 20284123


def style_weights(styles):
    _, inverse, counts = np.unique(np.asarray(styles), return_inverse=True, return_counts=True)
    return len(inverse)/(len(counts)*counts[inverse])


class FixedChemicalTransform(TransformerMixin, BaseEstimator):
    """Fold-local log/clip/median/scale plus a fixed-width missing block.

    Missing indicators are always present and weighted by 0.2; unlike rare
    Bernoulli scaling, a single missing value cannot make a huge new axis.
    """
    def __init__(self, features, calibrated_features, selected_features=None,
                 indicator_weight=0.2):
        self.features = features
        self.calibrated_features = calibrated_features
        self.selected_features = selected_features
        self.indicator_weight = indicator_weight

    def _raw(self, x):
        values = x.loc[:, self.features].to_numpy(dtype=float) if hasattr(x, "loc") else np.asarray(x, dtype=float)
        if values.shape[1] != len(self.features):
            raise ValueError("Expected declared hybrid feature order")
        return values[:, self.positions_].copy()

    def _log(self, values):
        for j in self.calibrated_positions_:
            if np.any(values[np.isfinite(values[:, j]), j] < 0):
                raise ValueError("Negative calibrated input")
            values[:, j] = np.log1p(values[:, j]/self.positive_scales_[j])
        return values

    def fit(self, x, y=None):
        selected = self.features if self.selected_features is None else self.selected_features
        self.selected_features_ = list(selected)
        self.positions_ = np.asarray([self.features.index(feature) for feature in selected])
        self.calibrated_positions_ = [j for j, feature in enumerate(selected)
                                      if feature in self.calibrated_features]
        values = self._raw(x)
        self.positive_scales_ = np.ones(values.shape[1])
        for j in self.calibrated_positions_:
            positive = values[np.isfinite(values[:, j]) & (values[:, j] > 0), j]
            self.positive_scales_[j] = np.median(positive) if len(positive) else 1.0
        values = self._log(values)
        self.bounds_ = np.nanquantile(values, [.01, .99], axis=0)
        values = np.clip(values, self.bounds_[0], self.bounds_[1])
        self.medians_ = np.nanmedian(values, axis=0)
        self.medians_ = np.nan_to_num(self.medians_, nan=0.0)
        filled = np.where(np.isnan(values), self.medians_, values)
        self.means_ = filled.mean(axis=0)
        self.scales_ = filled.std(axis=0)
        self.scales_[self.scales_ < 1e-12] = 1.0
        self.n_features_in_ = len(self.features)
        self.n_output_features_ = 2*len(selected)
        return self

    def transform(self, x):
        values = self._raw(x)
        missing = np.isnan(values)
        values = np.clip(self._log(values), self.bounds_[0], self.bounds_[1])
        filled = np.where(np.isnan(values), self.medians_, values)
        return np.column_stack([(filled-self.means_)/self.scales_,
                                self.indicator_weight*missing.astype(float)])


def refined_specs():
    specs = [
        {"name": "mean_global", "family": "mean", "selectable": False, "priority": 99},
        {"name": "old_multioutput_tree", "family": "old_tree", "selectable": False, "priority": 99},
        {"name": "fixed_single_tree", "family": "single_tree", "selectable": False, "priority": 99},
        {"name": "stable_ridge_aging100", "family": "ridge", "features": "aging", "alpha": 100.0, "priority": 1, "selectable": True},
        {"name": "stable_ridge_all1000", "family": "ridge", "features": "all226", "alpha": 1000.0, "priority": 2, "selectable": True},
        {"name": "tail_ridge_aging100", "family": "ridge", "features": "aging", "alpha": 100.0, "tail": True, "priority": 3, "selectable": True},
        {"name": "aux4_ridge10", "family": "aux", "features": "all226", "rank": 4, "alpha": 10.0, "priority": 4, "selectable": True},
        {"name": "aux8_ridge10", "family": "aux", "features": "all226", "rank": 8, "alpha": 10.0, "priority": 5, "selectable": True},
        {"name": "aux4_tail_ridge10", "family": "aux", "features": "all226", "rank": 4, "alpha": 10.0, "tail": True, "priority": 6, "selectable": True},
        {"name": "aux4_smooth_kernel", "family": "aux_kernel", "features": "all226", "rank": 4, "alpha": 1.0, "gamma_factor": .25, "priority": 7, "selectable": True},
        {"name": "style_center_within_ridge", "family": "within", "features": "all226", "center_alpha": 1000.0, "within_alpha": 100.0, "style_shrinkage": 5.0, "priority": 8, "selectable": True},
    ]
    members = ["stable_ridge_all1000", "aux4_ridge10", "tail_ridge_aging100"]
    specs.append({"name": "equal3_supervised_ensemble", "family": "ensemble", "priority": 9,
                  "selectable": True, "members": [dict(spec) for spec in specs if spec["name"] in members],
                  "weights": [1/3, 1/3, 1/3]})
    return specs


class RefinedStaleModel(RegressorMixin, BaseEstimator):
    """All sensory values are training targets; inference uses chemistry only."""
    def __init__(self, features, calibrated_features, aging_features, target_names,
                 config, seed):
        self.features = features
        self.calibrated_features = calibrated_features
        self.aging_features = aging_features
        self.target_names = target_names
        self.config = config
        self.seed = seed

    def fit(self, x, y, styles):
        all_targets = y.loc[:, self.target_names].to_numpy(dtype=float) if hasattr(y, "loc") else np.asarray(y, dtype=float)
        if all_targets.ndim != 2 or all_targets.shape[1] != len(self.target_names):
            raise ValueError("Training requires the declared sensory target matrix")
        self.target_position_ = self.target_names.index(TARGET)
        target = all_targets[:, self.target_position_]
        self.n_features_in_ = len(self.features)
        self.n_outputs_ = 1
        family = self.config["family"]
        if family == "mean":
            self.target_mean_ = float(target.mean())
            return self
        if family in ("old_tree", "single_tree"):
            self.imputer_ = SimpleImputer(strategy="median", add_indicator=True,
                                          keep_empty_features=True).fit(x)
            self.model_ = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                max_features=.7, random_state=self.seed, n_jobs=1)
            self.model_.fit(self.imputer_.transform(x), all_targets if family == "old_tree" else target)
            return self
        if family == "ensemble":
            self.members_ = [RefinedStaleModel(self.features, self.calibrated_features,
                self.aging_features, self.target_names, member, self.seed).fit(x, y, styles)
                for member in self.config["members"]]
            return self
        selected = self.aging_features if self.config["features"] == "aging" else None
        self.chemical_ = FixedChemicalTransform(self.features, self.calibrated_features, selected).fit(x)
        z = self.chemical_.transform(x)
        base_weights = style_weights(styles)
        weights = base_weights.copy()
        self.tail_training_threshold_ = float(np.quantile(target, .8))
        if self.config.get("tail", False):
            weights *= np.where(target >= self.tail_training_threshold_, 2.0, 1.0)
            weights *= len(weights)/weights.sum()
        if family == "within":
            center = np.empty(len(target))
            values = np.asarray(styles)
            global_mean = np.average(target, weights=base_weights)
            for style in np.unique(values):
                mask = values == style
                shrinkage = mask.sum()/(mask.sum()+self.config["style_shrinkage"])
                center[mask] = global_mean+shrinkage*(target[mask].mean()-global_mean)
            self.center_ = Ridge(alpha=self.config["center_alpha"], solver="svd").fit(z, center, sample_weight=base_weights)
            self.within_ = Ridge(alpha=self.config["within_alpha"], solver="svd").fit(z, target-center, sample_weight=base_weights)
            return self
        if family in ("aux", "aux_kernel"):
            # Stale is explicitly excluded from auxiliary supervision.
            self.auxiliary_target_indices_ = [j for j in range(len(self.target_names)) if j != self.target_position_]
            self.auxiliary_target_names_ = [self.target_names[j] for j in self.auxiliary_target_indices_]
            if TARGET in self.auxiliary_target_names_:
                raise AssertionError("Primary target cannot supervise its auxiliary directions")
            auxiliary = all_targets[:, self.auxiliary_target_indices_]
            self.auxiliary_mean_ = np.average(auxiliary, axis=0, weights=base_weights)
            self.auxiliary_scale_ = np.sqrt(np.average((auxiliary-self.auxiliary_mean_)**2, axis=0, weights=base_weights))
            self.auxiliary_scale_[self.auxiliary_scale_ < 1e-12] = 1.0
            standardized_auxiliary = (auxiliary-self.auxiliary_mean_)/self.auxiliary_scale_
            self.auxiliary_ridge_ = Ridge(alpha=100.0, solver="svd").fit(z, standardized_auxiliary, sample_weight=base_weights)
            _, self.auxiliary_singular_values_, vt = np.linalg.svd(self.auxiliary_ridge_.coef_, full_matrices=False)
            self.directions_ = vt[:self.config["rank"]].T
            latent = z@self.directions_
            self.latent_mean_ = np.average(latent, axis=0, weights=base_weights)
            self.latent_scale_ = np.sqrt(np.average((latent-self.latent_mean_)**2, axis=0, weights=base_weights))
            self.latent_scale_[self.latent_scale_ < 1e-12] = 1.0
            z = (latent-self.latent_mean_)/self.latent_scale_
        if family == "aux_kernel":
            self.target_mean_ = float(np.average(target, weights=weights))
            self.model_ = KernelRidge(alpha=self.config["alpha"], kernel="rbf",
                                     gamma=self.config["gamma_factor"]/z.shape[1])
            self.model_.fit(z, target-self.target_mean_, sample_weight=weights)
        else:
            self.model_ = Ridge(alpha=self.config["alpha"], solver="svd").fit(z, target, sample_weight=weights)
        return self

    def predict(self, x):
        family = self.config["family"]
        if family == "mean":
            return np.full(len(x), self.target_mean_)
        if family in ("old_tree", "single_tree"):
            prediction = self.model_.predict(self.imputer_.transform(x))
            return prediction[:, self.target_position_] if family == "old_tree" else prediction
        if family == "ensemble":
            return np.mean([member.predict(x) for member in self.members_], axis=0)
        z = self.chemical_.transform(x)
        if family == "within":
            return self.center_.predict(z)+self.within_.predict(z)
        if family in ("aux", "aux_kernel"):
            z = (z@self.directions_-self.latent_mean_)/self.latent_scale_
        prediction = self.model_.predict(z)
        return prediction+self.target_mean_ if family == "aux_kernel" else prediction
