"""Explicit hybrid chemical representation shared by the question-3 models.

This module reads attachments without modifying them or fitting preprocessing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import KEYS, data_dir, load_xy

DERIVED_FEATURES = (
    "kcalperc", "esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum"
)
DEFECT_FEATURES = ("DMS", "X2.3.butanedione", "X4VG", "X.E..2.nonenal")
DEFECT_FIELD_MAP = {
    "DMS": "DMS",
    "Diacetyl": "X2.3.butanedione",
    "4VG": "X4VG",
    "trans_2_nonenal": "X.E..2.nonenal",
}


def read_defect_thresholds() -> dict:
    """Read exact field matches and threshold units from attachment 8.

    Missing thresholds are Python None, deliberately not inferred or imported
    from external sources. Values are in the calibrated representation only.
    """
    table = pd.read_excel(data_dir() / "8.xlsx", sheet_name="Sensory thresholds")
    if table["field_name_R"].duplicated().any():
        raise ValueError("Ambiguous sensory-threshold field names")
    table = table.set_index("field_name_R")
    result = {}
    for label, feature in DEFECT_FIELD_MAP.items():
        row = table.loc[feature]
        result[label] = {
            "feature": feature,
            "chemical_name": str(row["chemical_name"]),
            "unit": str(row["unit"]) if pd.notna(row["unit"]) else None,
            "threshold_median": float(row["threshold_median"])
            if pd.notna(row["threshold_median"]) else None,
            "threshold_mean": float(row["threshold_mean"])
            if pd.notna(row["threshold_mean"]) else None,
            "threshold_source": "8.xlsx / Sensory thresholds / exact field_name_R",
        }
    return result


# Exported immutable-by-convention attachment metadata for downstream scripts.
DEFECTS = read_defect_thresholds()


def load_q3_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return identity metadata, 226 raw hybrid features, 45 targets, manifest.

    Rows and features preserve load_xy()/1.csv order. The 69 overlapping,
    non-derived calibrated indicators replace their 1.csv representations by
    exact beer key. The remaining 157 variables keep their supplied 1.csv
    representations. Calibrated negatives become missing; relative negatives
    are retained because their transformation/zero point is not identified.
    No imputation, standardization or fitted data transformation occurs here.
    """
    meta, original, y = load_xy()
    workbook = data_dir() / "8.xlsx"
    calibrated = pd.read_excel(workbook, sheet_name="Calibrated chemical dataset")
    thresholds = pd.read_excel(workbook, sheet_name="Sensory thresholds")
    if calibrated["beer"].isna().any() or not calibrated["beer"].is_unique:
        raise ValueError("Calibrated beer keys must be complete and unique")
    if set(calibrated["beer"]) != set(meta["beer"]):
        raise ValueError("Original and calibrated sample populations differ")
    calibrated = calibrated.set_index("beer").loc[meta["beer"]].reset_index()
    if not calibrated["tasting_category_fine"].equals(meta["tasting_category_fine"]):
        raise ValueError("Calibrated styles disagree after exact beer join")
    if thresholds["field_name_R"].duplicated().any():
        raise ValueError("Unit metadata must have unique exact field names")
    unit_map = thresholds.set_index("field_name_R")["unit"].to_dict()
    features = [feature for feature in original if feature not in DERIVED_FEATURES]
    calibrated_features = [feature for feature in calibrated if feature not in KEYS
                           and feature not in DERIVED_FEATURES]
    if not set(calibrated_features).issubset(features):
        raise ValueError("Calibrated indicators are absent from 1.csv")
    if len(features) != 226 or len(calibrated_features) != 69:
        raise ValueError("Attachment shape differs from verified 226/69 contract")
    x = original.loc[:, features].copy()
    records = []
    for position, feature in enumerate(features):
        is_calibrated = feature in calibrated_features
        values = pd.to_numeric(calibrated[feature] if is_calibrated
                               else original[feature], errors="raise").copy()
        original_missing = int(values.isna().sum())
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite attachment feature: {feature}")
        negative = values.lt(0) if is_calibrated else pd.Series(False, index=values.index)
        values.loc[negative] = np.nan
        x.loc[:, feature] = values.to_numpy()
        raw_unit = unit_map.get(feature)
        unit_available = is_calibrated and pd.notna(raw_unit)
        records.append({
            "position": position,
            "feature": feature,
            "representation": "calibrated" if is_calibrated else "relative",
            "unit": str(raw_unit) if unit_available else
                    ("not_provided_in_attachment" if is_calibrated
                     else "supplied_relative_or_transformed_units"),
            "unit_source": "8.xlsx / Sensory thresholds / exact field_name_R"
                           if unit_available else "unidentified",
            "source": "8.xlsx / Calibrated chemical dataset" if is_calibrated else "1.csv",
            "original_missing_count": original_missing,
            "original_1csv_missing_count": int(original[feature].isna().sum()),
            "negative_invalid_count": int(negative.sum()),
            "final_missing_count": int(values.isna().sum()),
            "zero_count": int(values.eq(0).sum()),
            "negative_policy": "replace_with_NaN" if is_calibrated else "retain_supplied_representation",
            "is_defect_concentration": feature in DEFECT_FEATURES,
        })
    manifest = pd.DataFrame(records)
    if x.shape != (250, 226) or y.shape != (250, 45):
        raise ValueError("Unexpected question-3 input or target dimensions")
    if manifest["representation"].value_counts().to_dict() != {"relative": 157, "calibrated": 69}:
        raise AssertionError("Hybrid representation count mismatch")
    return meta.loc[:, KEYS].copy(), x, y.copy(), manifest
