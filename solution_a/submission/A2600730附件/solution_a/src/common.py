"""Shared data contract for the A-track modelling experiments."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[2]
SOLUTION = ROOT / "solution_a"
SEED = 20261023
KEYS = ["beer", "beer_id", "tasting_category_fine"]
EXCLUDED_TARGETS = ["alcohol", "aftertaste", "body", "co2", "overall"]


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Learn clipping bounds on the training fold; preserve missing entries."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, x, y=None):
        values = np.asarray(x, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.bounds_ = np.nanquantile(values, [self.lower, self.upper], axis=0)
        return self

    def transform(self, x):
        values = np.asarray(x, dtype=float)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from training")
        return np.clip(values, self.bounds_[0], self.bounds_[1])


def data_dir():
    candidates = list(ROOT.glob("*/2026*赛道A*/附件"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one A-track attachment directory; got {candidates}")
    return candidates[0]


def resolve_recorded_path(value):
    """Resolve repository-relative provenance and the original machine's paths."""
    recorded = str(value).replace("\\", "/")
    if recorded.startswith("solution_a/"):
        return ROOT / recorded
    if "/solution_a/" in recorded:
        return SOLUTION / recorded.split("/solution_a/", 1)[1]
    if "/附件/" in recorded:
        return data_dir() / recorded.rsplit("/", 1)[1]
    direct = Path(recorded)
    return direct if direct.is_absolute() else ROOT / direct


def load_xy():
    """Join by exact beer number, never by the rounded barcodes in 4.csv."""
    chem = pd.read_csv(data_dir() / "1.csv", dtype={"beer_id": str})
    sensory = pd.read_csv(data_dir() / "4.csv", dtype={"beer_id": str})
    for frame in (chem, sensory):
        if frame["beer"].isna().any() or not frame["beer"].is_unique:
            raise ValueError("beer must be a complete, unique sample key")
    if set(chem.beer) != set(sensory.beer):
        raise ValueError("Chemical and sensory populations differ")
    sensory = sensory.set_index("beer").loc[chem.beer].reset_index()
    if not chem.tasting_category_fine.equals(sensory.tasting_category_fine):
        raise ValueError("Styles disagree after joining by beer")
    features = [c for c in chem if c not in KEYS]
    targets = [c for c in sensory if c not in KEYS + EXCLUDED_TARGETS]
    x = chem[features].apply(pd.to_numeric, errors="raise")
    y = sensory[targets].apply(pd.to_numeric, errors="raise")
    if y.isna().any().any():
        raise ValueError("Missing labels need target-specific training, not imputation")
    return chem[KEYS], x, y
