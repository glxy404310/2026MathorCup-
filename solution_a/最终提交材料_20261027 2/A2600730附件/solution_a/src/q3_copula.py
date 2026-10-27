"""Small-sample style conditional Gaussian copula in the mixed Q3 feature space."""
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import ndtr, ndtri
from sklearn.covariance import LedoitWolf


class ConditionalGaussianCopula:
    """Empirical margins, shrunken style means, and positive definite covariances.

    ``fit`` learns *all* preprocessing from its input rows. Missing entries are
    replaced by training medians. The inverse empirical CDF returns observed
    column values, so zeros and ties have explicit probability intervals rather
    than being replaced by interpolated concentrations. Their generated
    frequencies need not equal empirical frequencies after conditioning and
    the latent Gaussian approximation. Sampling creates new
    combinations, not necessarily new individual measurement values.

    ``sample(None, ...)`` is the unconditional copula comparison. ``distance``
    is sqrt(Mahalanobis squared / number of features); its fitted training
    reference must be used to interpret it, not a chi-square calibration.
    """

    def __init__(self, mean_strength=5.0, cov_strength=20.0):
        if mean_strength < 0 or cov_strength < 0:
            raise ValueError("Shrinkage strengths must be nonnegative")
        self.mean_strength = float(mean_strength)
        self.cov_strength = float(cov_strength)

    def _array(self, x):
        if isinstance(x, pd.DataFrame):
            missing = set(self.columns_) - set(x.columns)
            extra = set(x.columns) - set(self.columns_)
            if missing or extra:
                raise ValueError(f"Feature mismatch: missing={missing}, extra={extra}")
            values = x.loc[:, self.columns_].to_numpy(dtype=float)
        else:
            values = np.asarray(x, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.columns_):
            raise ValueError("Expected a matrix with the fitted feature count")
        if np.isinf(values).any():
            raise ValueError("Infinite chemical measurements are unsupported")
        return values

    def impute(self, x):
        values = self._array(x)
        return np.where(np.isnan(values), self.median_, values)

    @staticmethod
    def _regularized_covariance(z, assume_centered=False):
        if len(z) > 1:
            covariance = LedoitWolf(assume_centered=assume_centered).fit(z).covariance_
        else:
            covariance = np.eye(z.shape[1])
        # A tiny explicit ridge also handles completely constant columns.
        ridge = 1e-6 * max(float(np.trace(covariance) / len(covariance)), 1.0)
        return (covariance + covariance.T) / 2 + ridge * np.eye(z.shape[1])

    def fit(self, x, styles):
        if not isinstance(x, pd.DataFrame) or not x.columns.is_unique:
            raise ValueError("fit requires a DataFrame with unique feature names")
        self.columns_ = x.columns.tolist()
        values = self._array(x)
        labels = np.asarray(styles, dtype=str)
        if labels.ndim != 1 or len(labels) != len(values) or len(values) < 2:
            raise ValueError("One style per row and at least two training rows required")
        if np.isnan(values).all(axis=0).any():
            raise ValueError("A feature is completely missing in this training fold")
        self.median_ = np.nanmedian(values, axis=0)
        values = np.where(np.isnan(values), self.median_, values)
        self.n_features_in_ = values.shape[1]
        self.n_samples_ = len(values)
        self.empirical_values_ = []
        self.empirical_midprob_ = []
        self.empirical_cumprob_ = []
        for column in values.T:
            unique, counts = np.unique(column, return_counts=True)
            cumulative = counts.cumsum() / len(column)
            midpoint = (counts.cumsum() - counts / 2) / len(column)
            self.empirical_values_.append(unique)
            self.empirical_midprob_.append(midpoint)
            self.empirical_cumprob_.append(cumulative)
        z = self.transform(x)
        self.global_center_ = z.mean(axis=0)
        self.global_cov_ = self._regularized_covariance(z)
        self.global_cholesky_ = np.linalg.cholesky(self.global_cov_)
        self.style_counts_ = {s: int(np.sum(labels == s)) for s in np.unique(labels)}
        raw_centers = {s: z[labels == s].mean(axis=0) for s in self.style_counts_}
        residuals = np.vstack([z[labels == s] - raw_centers[s]
                               for s in self.style_counts_])
        self.pooled_within_cov_ = self._regularized_covariance(
            residuals, assume_centered=True)
        self.style_center_ = {}
        self.style_cov_ = {}
        self.style_cholesky_ = {}
        self.fit_diagnostics_ = []
        for style, count in self.style_counts_.items():
            a = count / (count + self.mean_strength)
            b = count / (count + self.cov_strength)
            center = a * raw_centers[style] + (1 - a) * self.global_center_
            local_cov = self._regularized_covariance(z[labels == style])
            covariance = b * local_cov + (1 - b) * self.pooled_within_cov_
            self.style_center_[style] = center
            self.style_cov_[style] = covariance
            self.style_cholesky_[style] = np.linalg.cholesky(covariance)
            self.fit_diagnostics_.append({
                "style": style, "n": count, "mean_local_weight": a,
                "cov_local_weight": b,
                "minimum_cov_eigenvalue": float(np.linalg.eigvalsh(covariance)[0]),
            })
        return self

    def transform(self, x):
        values = self.impute(x)
        z = np.empty_like(values)
        # Mid-CDF is exact at observed ties. Linear interpolation is used only
        # to score intermediate inputs (e.g. projected migration candidates).
        tail = 0.5 / self.n_samples_
        for j in range(values.shape[1]):
            probability = np.interp(values[:, j], self.empirical_values_[j],
                                    self.empirical_midprob_[j])
            z[:, j] = ndtri(np.clip(probability, tail, 1 - tail))
        return z

    def inverse_transform(self, z):
        latent = np.asarray(z, dtype=float)
        if latent.ndim != 2 or latent.shape[1] != self.n_features_in_:
            raise ValueError("Latent feature count differs from training")
        probabilities = ndtr(latent)
        values = np.empty_like(latent)
        for j in range(latent.shape[1]):
            indices = np.searchsorted(self.empirical_cumprob_[j],
                                      probabilities[:, j], side="left")
            indices = np.minimum(indices, len(self.empirical_values_[j]) - 1)
            values[:, j] = self.empirical_values_[j][indices]
        return pd.DataFrame(values, columns=self.columns_)

    def sample(self, style, n=300, seed=20261023):
        if n < 1:
            raise ValueError("At least one sample is required")
        center = self.global_center_ if style is None else self.style_center_[style]
        chol = self.global_cholesky_ if style is None else self.style_cholesky_[style]
        z = np.random.default_rng(seed).normal(size=(n, self.n_features_in_))
        return self.inverse_transform(center + z @ chol.T)

    def distance(self, x, style=None):
        center = self.global_center_ if style is None else self.style_center_[style]
        chol = self.global_cholesky_ if style is None else self.style_cholesky_[style]
        residual = (self.transform(x) - center).T
        whitened = solve_triangular(chol, residual, lower=True, check_finite=False)
        return np.sqrt(np.mean(whitened ** 2, axis=0))
