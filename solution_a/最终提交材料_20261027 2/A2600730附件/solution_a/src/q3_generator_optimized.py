"""Nested selection of small-sample generators with style-specific margins."""
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from q3_copula import ConditionalGaussianCopula


EVALUATION_STYLES = ("Tripel", "Kriek", "Lambic", "Pils/Lager")
CANDIDATE_METHODS = ("style_margin_copula", "style_smoothed_mixture", "legacy_copula")
MIN_INNER_NN_RATIO = 0.35
MAX_INNER_MATCH_FRACTION = 0.95
SMOOTHING_NOISE = 0.45


def empirical_tables(values):
    unique_values, midpoints, cumulative = [], [], []
    for column in values.T:
        unique, counts = np.unique(column, return_counts=True)
        unique_values.append(unique)
        midpoints.append((counts.cumsum() - counts / 2) / len(column))
        cumulative.append(counts.cumsum() / len(column))
    return unique_values, midpoints, cumulative


def normal_scores(values, tables):
    z = np.empty_like(values, dtype=float)
    for j, (unique, midpoints) in enumerate(zip(tables[0], tables[1])):
        z[:, j] = ndtri(np.interp(values[:, j], unique, midpoints))
    return z


def regularized_correlation(z):
    if len(z) < 2 or np.max(np.abs(z)) < 1e-12:
        return np.eye(z.shape[1])
    covariance = LedoitWolf().fit(z).covariance_
    covariance += 1e-6 * np.eye(z.shape[1])
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / scale[:, None] / scale[None, :]
    return (correlation + correlation.T) / 2


class StyleMarginCopula:
    """Use per-style inverse empirical margins and shrunken dependence.

    ``mode='copula'`` uses a zero-mean, unit-marginal Gaussian whose covariance
    mixes local and shared style-rank correlations. ``mode='smoothed'`` draws
    an empirical style row in normal-score space and adds correlated noise
    with fixed standard deviation 0.45. It smooths empirical combinations,
    rather than choosing an almost-zero perturbation to avoid exact copies.

    Zeros have probability intervals in the inverse CDF. Gaussian mode has
    standard-normal latent margins, so empirical zero rates are preserved in
    expectation; smoothed mode need not reproduce them exactly.
    """

    def __init__(self, mode="copula", cov_strength=10.0, noise=SMOOTHING_NOISE):
        if mode not in ("copula", "smoothed"):
            raise ValueError("Unknown style-margin generation mode")
        self.mode = mode
        self.cov_strength = float(cov_strength)
        self.noise = float(noise)

    def fit(self, x, styles):
        self.columns_ = x.columns.tolist()
        values = x.to_numpy(dtype=float)
        labels = np.asarray(styles, dtype=str)
        self.global_median_ = np.nanmedian(values, axis=0)
        if np.isnan(self.global_median_).any() or np.isinf(values).any():
            raise ValueError("A feature is entirely missing or infinite")
        self.tables_, self.style_scores_, self.style_median_ = {}, {}, {}
        self.style_counts_ = {}
        for style in np.unique(labels):
            local = values[labels == style]
            median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any()
                               else self.global_median_[j] for j, c in enumerate(local.T)])
            local = np.where(np.isnan(local), median, local)
            self.style_median_[style] = median
            self.tables_[style] = empirical_tables(local)
            self.style_scores_[style] = normal_scores(local, self.tables_[style])
            self.style_counts_[style] = len(local)
        pooled = np.vstack(list(self.style_scores_.values()))
        self.shared_correlation_ = regularized_correlation(pooled)
        self.style_correlation_, self.style_cholesky_ = {}, {}
        for style, scores in self.style_scores_.items():
            weight = len(scores) / (len(scores) + self.cov_strength)
            correlation = (weight * regularized_correlation(scores)
                           + (1 - weight) * self.shared_correlation_)
            self.style_correlation_[style] = correlation
            self.style_cholesky_[style] = np.linalg.cholesky(correlation)
        return self

    def sample(self, style, n=300, seed=20261123):
        if n < 1:
            raise ValueError("At least one sample is required")
        rng = np.random.default_rng(seed)
        z = rng.normal(size=(n, len(self.columns_))) @ self.style_cholesky_[style].T
        if self.mode == "smoothed":
            centers = self.style_scores_[style]
            indices = rng.integers(0, len(centers), size=n)
            z = centers[indices] + self.noise * z
        probability = ndtr(z)
        values = np.empty_like(z)
        unique, _, cumulative = self.tables_[style]
        for j in range(values.shape[1]):
            indices = np.searchsorted(cumulative[j], probability[:, j], side="left")
            values[:, j] = unique[j][np.minimum(indices, len(unique[j]) - 1)]
        return pd.DataFrame(values, columns=self.columns_)


def fit_candidates(x, labels):
    legacy = ConditionalGaussianCopula(5.0, 20.0).fit(x, labels)
    copula = StyleMarginCopula(mode="copula").fit(x, labels)
    # Both families have identical fitted margins and covariance; reuse fitted
    # arrays and change only the declared sampling kernel.
    mixture = StyleMarginCopula(mode="smoothed")
    mixture.__dict__.update(copula.__dict__)
    mixture.mode = "smoothed"
    return {"style_margin_copula": copula, "style_smoothed_mixture": mixture,
            "legacy_copula": legacy}


def selection_metrics(reference_model, pca, pc_scale, style_train, heldout, generated):
    train_z = reference_model.transform(style_train)
    real_z = reference_model.transform(heldout)
    fake_z = reference_model.transform(generated)
    real_pc, fake_pc = pca.transform(real_z) / pc_scale, pca.transform(fake_z) / pc_scale
    energy = float(2 * cdist(real_pc, fake_pc).mean()
                   - cdist(real_pc, real_pc).mean() - cdist(fake_pc, fake_pc).mean())
    distances = cdist(fake_z, train_z) / np.sqrt(train_z.shape[1])
    nn = distances.min(axis=1)
    real_nn = cdist(real_z, train_z).min(axis=1) / np.sqrt(train_z.shape[1])
    nearest = distances.argmin(axis=1)
    fake = reference_model.impute(generated)
    training = reference_model.impute(style_train)[nearest]
    equal_fraction = np.isclose(fake, training, rtol=1e-10, atol=1e-12).mean(axis=1)
    observed = style_train.to_numpy(dtype=float)
    copied_observed = (np.isclose(fake[:, None, :], observed[None, :, :],
                                  rtol=1e-10, atol=1e-12)
                       | np.isnan(observed)[None, :, :]).all(axis=2).any(axis=1)
    return {"energy_pca5": energy,
            "nn_ratio": float(np.median(nn) / max(np.median(real_nn), 1e-12)),
            "copy_rate": float(np.mean(nn < 1e-12)),
            "observed_training_copy_rate": float(copied_observed.mean()),
            "nearest_equal_feature_fraction": float(np.median(equal_fraction))}


class OptimizedStyleGenerator:
    """Choose one of three fixed generators using training-only inner folds.

    Inner validation uses two folds stratified across every training style.
    The generator for each of the four evaluation styles minimizes inner
    five-dimensional energy distance, subject to no exact copies, NN ratio
    >= 0.35, and median matching fraction <= 0.95 in every inner fold.
    Other styles default to the legacy generator. Bootstrap is evaluated as a
    benchmark but is excluded from selection because it is a copying model.
    """

    def __init__(self, selection_seed=20261123, inner_n_generated=300):
        self.selection_seed = int(selection_seed)
        self.inner_n_generated = int(inner_n_generated)

    def fit(self, x, styles):
        self.columns_ = x.columns.tolist()
        labels = pd.Series(np.asarray(styles, dtype=str), index=x.index)
        if labels.value_counts().min() < 2:
            raise ValueError("Inner stratified two-fold requires two rows per training style")
        selected_styles = [style for style in EVALUATION_STYLES if style in set(labels)]
        records = []
        splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.selection_seed)
        for fold, (train, validation) in enumerate(splitter.split(x, labels), start=1):
            inner_train = x.iloc[train]
            inner_labels = labels.iloc[train]
            fitted = fit_candidates(inner_train, inner_labels)
            reference = fitted["legacy_copula"]
            train_z = reference.transform(inner_train)
            pca = PCA(n_components=min(5, len(train_z) - 1), svd_solver="full").fit(train_z)
            pc_scale = np.maximum(np.sqrt(pca.explained_variance_), 1e-8)
            for style_index, style in enumerate(selected_styles):
                training_style = inner_train.loc[inner_labels == style]
                heldout = x.iloc[validation].loc[labels.iloc[validation] == style]
                for method, candidate in fitted.items():
                    seed = self.selection_seed + fold * 1000 + style_index * 10
                    generated = candidate.sample(style, self.inner_n_generated, seed)
                    metrics = selection_metrics(reference, pca, pc_scale, training_style, heldout, generated)
                    eligible = (metrics["copy_rate"] == 0
                                and metrics["observed_training_copy_rate"] == 0
                                and metrics["nn_ratio"] >= MIN_INNER_NN_RATIO
                                and metrics["nearest_equal_feature_fraction"] <= MAX_INNER_MATCH_FRACTION)
                    records.append({"inner_fold": fold, "style": style, "method": method,
                                    "seed": seed, "train_n_style": len(training_style),
                                    "validation_n_style": len(heldout), "eligible": eligible, **metrics})
        self.inner_scores_ = pd.DataFrame(records)
        self.selected_methods_ = {style: "legacy_copula" for style in labels.unique()}
        self.selection_table_ = []
        for style in selected_styles:
            subset = self.inner_scores_.loc[self.inner_scores_["style"] == style]
            grouped = subset.groupby("method").agg(energy_pca5=("energy_pca5", "mean"),
                eligible=("eligible", "all"), worst_nn_ratio=("nn_ratio", "min"),
                largest_matching_fraction=("nearest_equal_feature_fraction", "max"))
            accepted = grouped.loc[grouped.eligible]
            if len(accepted):
                selected = accepted.energy_pca5.idxmin()
                status = "selected_by_training_only_inner_energy_with_novelty_gates"
            else:
                selected = "legacy_copula"
                status = "no_candidate_passed_inner_gates_legacy_fallback_requires_diagnostics"
            self.selected_methods_[style] = selected
            for method, row in grouped.iterrows():
                self.selection_table_.append({"style": style, "method": method,
                    "selected": method == selected, "selection_status": status, **row.to_dict()})
        self.fitted_candidates_ = fit_candidates(x, labels)
        return self

    def sample(self, style, n=300, seed=20261123):
        if style is None:
            return self.fitted_candidates_["legacy_copula"].sample(None, n, seed)
        return self.fitted_candidates_[self.selected_methods_[style]].sample(style, n, seed)
