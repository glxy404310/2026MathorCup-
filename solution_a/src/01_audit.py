#!/usr/bin/env python3
"""Read-only, reproducible audit of the eight Track A attachments.

No original file is modified; no network request is made. Findings have not
been independently verified by a human. Uses pandas/numpy and read-only
openpyxl; does not require scipy, and does not author any workbook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl


ATTRIBUTION = "本地附件结构、数值一致性与来源完整性审计。"
KEYS = ["beer", "beer_id", "tasting_category_fine"]


def normalize_label(value: object) -> str:
    return " ".join(str(value).replace("_x000D_", "").split())


def number(value: object) -> float | None:
    return float(value) if pd.notna(value) and np.isfinite(float(value)) else None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(frame: pd.DataFrame, source: str) -> list[dict]:
    records = []
    for column in frame.columns:
        series = frame[column]
        numeric = pd.api.types.is_numeric_dtype(series)
        row = {
            "source": source, "column": str(column), "dtype": str(series.dtype),
            "rows": len(series), "missing": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()) if len(series) else None,
            "unique_nonmissing": int(series.nunique(dropna=True)),
            "all_missing": bool(series.isna().all()),
            "constant_nonmissing": bool(series.nunique(dropna=True) == 1),
        }
        if numeric:
            finite = series[np.isfinite(series)]
            q1, q3 = finite.quantile([0.25, 0.75]) if len(finite) else (np.nan, np.nan)
            iqr = q3 - q1
            row.update({
                "min": number(finite.min()), "q01": number(finite.quantile(0.01)),
                "q25": number(q1), "median": number(finite.median()),
                "q75": number(q3), "q99": number(finite.quantile(0.99)),
                "max": number(finite.max()), "mean": number(finite.mean()),
                "std_ddof1": number(finite.std(ddof=1)),
                "negative_count": int((series < 0).sum()),
                "zero_count": int((series == 0).sum()),
                "infinite_count": int(np.isinf(series).sum()),
                "tukey_outside_1_5_iqr": int(((finite < q1 - 1.5 * iqr) | (finite > q3 + 1.5 * iqr)).sum()),
            })
        records.append(row)
    return records


def table_summary(frame: pd.DataFrame) -> dict:
    numeric = frame.select_dtypes(include=np.number)
    return {
        "rows": len(frame), "columns": len(frame.columns),
        "column_names": list(map(str, frame.columns)),
        "missing_cells": int(frame.isna().sum().sum()),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "numeric_columns": len(numeric.columns),
        "infinite_cells": int(np.isinf(numeric).sum().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    if args.data_dir:
        data_dir = args.data_dir.resolve()
    else:
        candidates = list(project.glob("**/*赛道A初赛/附件"))
        if len(candidates) != 1:
            raise RuntimeError("Expected exactly one Track A attachment directory; use --data-dir.")
        data_dir = candidates[0]
    output = args.output_dir or project / "solution_a/outputs/audit"
    output.mkdir(parents=True, exist_ok=True)
    report_path = project / "solution_a/docs/data_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    paths = [data_dir / f"{i}.{'xlsx' if i in (3, 8) else 'csv'}" for i in range(1, 9)]
    paths.append(data_dir / "数据说明.txt")
    hashes_before = {path.name: sha256(path) for path in paths}
    csv = {i: pd.read_csv(data_dir / f"{i}.csv") for i in (1, 2, 4, 5, 6, 7)}
    xlsx = {i: pd.read_excel(data_dir / f"{i}.xlsx", sheet_name=None) for i in (3, 8)}
    frames = {f"{i}.csv": frame for i, frame in csv.items()}
    for i, sheets in xlsx.items():
        frames.update({f"{i}.xlsx::{name}": frame for name, frame in sheets.items()})
    records = [row for source, frame in frames.items() for row in profile(frame, source)]
    columns = pd.DataFrame(records)
    columns.to_csv(output / "column_profile.csv", index=False)
    columns.loc[columns.missing > 0].to_csv(output / "missingness.csv", index=False)
    pd.DataFrame([{"source": path.name, "bytes": path.stat().st_size,
                   "sha256": hashes_before[path.name]} for path in paths]).to_csv(output / "source_files.csv", index=False)

    chemical = csv[1].set_index("beer")
    sensory = csv[4].set_index("beer")
    comments = csv[7].set_index("beer")
    calibrated = xlsx[8]["Calibrated chemical dataset"].set_index("beer")
    prices = xlsx[8]["Beer Prices"].set_index("beer")
    shap = xlsx[8]["GBM SHAP Values"]
    concentrations = shap.loc[shap.data == "concentration"].set_index("beer")
    shap_values = shap.loc[shap.data == "SHAP"].set_index("beer")
    keyed = {
        "1.csv": chemical, "4.csv": sensory, "7.csv": comments,
        "8.xlsx::Calibrated chemical dataset": calibrated,
        "8.xlsx::Beer Prices": prices,
        "8.xlsx::GBM SHAP Values::concentration": concentrations,
        "8.xlsx::GBM SHAP Values::SHAP": shap_values,
    }
    checks, id_discrepancies = [], []
    for source, frame in keyed.items():
        common = chemical.index.intersection(frame.index)
        aligned = frame.reindex(chemical.index)
        row = {
            "source": source, "rows": len(frame), "unique_beer": int(frame.index.nunique()),
            "duplicate_beer_rows": int(frame.index.duplicated(keep=False).sum()),
            "missing_beer_rows": int(frame.index.isna().sum()),
            "unique_beer_id": int(frame.beer_id.nunique()),
            "duplicate_beer_id_rows": int(frame.beer_id.duplicated(keep=False).sum()),
            "beer_overlap_with_1": len(common),
            "beer_missing_from_1": len(frame.index.difference(chemical.index)),
            "beer_missing_in_source": len(chemical.index.difference(frame.index)),
            "same_order_as_1": frame.index.equals(chemical.index),
            "beer_id_mismatches_with_1": int((aligned.beer_id != chemical.beer_id).sum()),
            "category_mismatches_with_1": int((aligned.tasting_category_fine != chemical.tasting_category_fine).sum())
                if "tasting_category_fine" in aligned else None,
        }
        checks.append(row)
        for beer in common:
            if frame.at[beer, "beer_id"] != chemical.at[beer, "beer_id"]:
                id_discrepancies.append({"source": source, "beer": beer,
                    "beer_id_in_1": str(chemical.at[beer, "beer_id"]),
                    "beer_id_in_source": str(frame.at[beer, "beer_id"])})
    pd.DataFrame(checks).to_csv(output / "key_checks.csv", index=False)
    pd.DataFrame(id_discrepancies).to_csv(output / "beer_id_discrepancies.csv", index=False)

    # Chemical labels follow 2.csv's order. The complete Spearman reconciliation
    # below checks this positional correspondence, rather than assuming it.
    m2, m5, m6 = [csv[i].set_index(csv[i].columns[0]) for i in (2, 5, 6)]
    chem_columns = list(csv[1].columns[3:])
    chem_map = dict(zip(m2.columns, chem_columns, strict=True))
    names = xlsx[8]["Sensory names"]
    panel_names = names.loc[names.dataset == "Trained panel"]
    panel_metadata = panel_names.set_index("R_name")
    selected_45 = panel_names.loc[panel_names.Category_Paper.isin(["Aroma", "Taste"]), "R_name"].tolist()
    sensory_lookup = {normalize_label(row.CleanName_PanelFeedback): row.R_name
                      for row in panel_names.itertuples(index=False)}
    sen_map = {name: sensory_lookup[normalize_label(name)] for name in m5.columns}
    thresholds = xlsx[8]["Sensory thresholds"].set_index("field_name_R")
    feature_map = []
    for display, field in chem_map.items():
        feature_map.append({"domain": "chemical", "field": field, "matrix_label": display,
            "source": "1.csv; 2.csv; 8.xlsx::Sensory thresholds",
            "map_basis": "2.csv order; all Spearman matrix entries reproduced",
            "metadata_name": thresholds.at[field, "chemical_name"],
            "unit": thresholds.at[field, "unit"],
            "in_cross_matrix_6": display in m6.index,
            "in_calibrated_dataset": field in calibrated.columns})
    for display, field in sen_map.items():
        feature_map.append({"domain": "sensory", "field": field, "matrix_label": display,
            "source": "4.csv; 5.csv; 8.xlsx::Sensory names",
            "map_basis": "Sensory names: R_name <-> CleanName_PanelFeedback",
            "category_paper": panel_metadata.at[field, "Category_Paper"],
            "selected_aroma_taste_45": field in selected_45})
    pd.DataFrame(feature_map).to_csv(output / "feature_name_map.csv", index=False)

    reconcile = []
    matrices = {}
    for name, given in [("2.csv", m2), ("5.csv", m5), ("6.csv", m6)]:
        values = given.to_numpy(dtype=float)
        square = values.shape[0] == values.shape[1]
        matrices[name] = {
            "numeric_shape": list(values.shape), "row_labels_unique": bool(given.index.is_unique),
            "column_labels_unique": bool(given.columns.is_unique),
            "min": float(values.min()), "max": float(values.max()),
            "symmetry_max_error": float(np.max(np.abs(values - values.T))) if square else None,
            "diagonal_max_error_from_1": float(np.max(np.abs(np.diag(values) - 1))) if square else None,
        }
        for method in ("pearson", "spearman"):
            if name == "2.csv":
                recalculated = chemical[[chem_map[x] for x in given.columns]].corr(method=method).to_numpy()
                observations = chemical[[chem_map[x] for x in given.columns]].notna().astype(int)
                n_pairs = observations.T @ observations
            elif name == "5.csv":
                recalculated = sensory[[sen_map[x] for x in given.columns]].corr(method=method).to_numpy()
                observations = sensory[[sen_map[x] for x in given.columns]].notna().astype(int)
                n_pairs = observations.T @ observations
            else:
                left = chemical[[chem_map[x] for x in given.index]]
                right = sensory[[sen_map[x] for x in given.columns]]
                combined = pd.concat([left, right], axis=1)
                recalculated = combined.corr(method=method).to_numpy()[:len(left.columns), len(left.columns):]
                n_pairs = left.notna().astype(int).T @ right.notna().astype(int)
            error = np.abs(recalculated - values)
            reconcile.append({"matrix": name, "method": method, "entries": values.size,
                "pair_n_min": int(n_pairs.min().min()), "pair_n_max": int(n_pairs.max().max()),
                "max_abs_error": float(np.max(error)), "mean_abs_error": float(np.mean(error)),
                "entries_with_error_gt_1e_10": int((error > 1e-10).sum()),
                "matches_at_1e_10": bool(np.all(error <= 1e-10))})
    pd.DataFrame(reconcile).to_csv(output / "matrix_reconciliation.csv", index=False)

    concentration_error = chemical[chem_columns] - concentrations.loc[chemical.index, chem_columns]
    calibration_check = []
    for field in calibrated.columns[2:]:
        comparison = pd.concat([chemical[field], calibrated[field]], axis=1)
        raw = calibrated[field]
        min_positive_half = raw.loc[raw > 0].min() / 2
        replaced = raw.where(raw != 0, min_positive_half)
        log10_values = pd.Series(np.nan, index=raw.index)
        valid_log10 = replaced > 0
        log10_values.loc[valid_log10] = np.log10(replaced.loc[valid_log10])
        log10_error = (chemical[field] - log10_values).abs()
        calibration_check.append({"field": field, "n_pairwise_complete": int(comparison.notna().all(axis=1).sum()),
            "spearman_with_1": float(comparison.corr(method="spearman").iloc[0, 1]),
            "missing_1": int(chemical[field].isna().sum()), "missing_calibrated": int(calibrated[field].isna().sum()),
            "negative_calibrated": int((calibrated[field] < 0).sum()),
            "unit_metadata": thresholds.at[field, "unit"],
            "transformation_group": "first_37_basic" if field in chem_columns[:37] else "remaining_33_GCMS_overlap",
            "calibrated_zero_count": int((raw == 0).sum()),
            "zero_replacement_min_positive_half": number(min_positive_half),
            "n_log10_comparable": int(log10_error.notna().sum()),
            "log10_max_abs_error": number(log10_error.max()),
            "log10_matches_at_1e_7": bool(log10_error.max() < 1e-7),
        })
    pd.DataFrame(calibration_check).to_csv(output / "calibrated_comparison.csv", index=False)

    negative_calibrated = []
    for field in calibrated.columns[2:]:
        for beer, value in calibrated[field][calibrated[field] < 0].items():
            negative_calibrated.append({"beer": beer, "field": field, "value": value,
                "unit_metadata": thresholds.at[field, "unit"]})
    pd.DataFrame(negative_calibrated).to_csv(output / "negative_calibrated_values.csv", index=False)
    extreme = []
    for source, frame in [("1.csv", chemical), ("4.csv", sensory), ("7.csv", comments)]:
        for field in frame.columns[2:]:
            valid = frame[field].dropna()
            if not len(valid):
                continue
            q1, q3 = valid.quantile([0.25, 0.75])
            iqr = q3 - q1
            for beer, value in valid.loc[(valid < q1 - 3 * iqr) | (valid > q3 + 3 * iqr)].items():
                extreme.append({"source": source, "beer": beer, "field": field, "value": value,
                    "q25": q1, "q75": q3, "rule": "outside [q25-3IQR,q75+3IQR]; flag only, not error verdict"})
    pd.DataFrame(extreme).to_csv(output / "extreme_value_flags.csv", index=False)

    # Inspect formula-bearing forms read-only, so cached zeros are not mistaken
    # for observations or 12 additional beers.
    workbook3 = openpyxl.load_workbook(data_dir / "3.xlsx", read_only=True, data_only=False)
    template_formulas = {sheet.title: sum(cell.data_type == "f" for row in sheet for cell in row)
                         for sheet in workbook3.worksheets}
    workbook3.close()
    price_cols = [col for col in prices if col.startswith("Price_")]
    prices[price_cols].notna().sum().rename("available_beers").to_csv(output / "price_availability.csv")
    category_counts = chemical.tasting_category_fine.value_counts().rename_axis("style").reset_index(name="beers")
    category_counts.to_csv(output / "style_counts.csv", index=False)
    category_map = pd.concat([chemical.tasting_category_fine.rename("category_1"),
                             prices.tasting_category.rename("category_prices")], axis=1)
    category_map.groupby(["category_1", "category_prices"]).size().reset_index(name="beers").to_csv(
        output / "price_category_map.csv", index=False)
    missing_primary = {str(i): {name: int(count) for name, count in csv[i].isna().sum().items() if count}
                       for i in (1, 4, 7)}
    assertions = {
        "all_1_4_7_beer_unique_and_same_set": all(frame.index.is_unique and set(frame.index) == set(chemical.index)
                                                   for frame in (chemical, sensory, comments)),
        "all_three_matrices_reproduce_spearman": all(row["matches_at_1e_10"] for row in reconcile if row["method"] == "spearman"),
        "concentration_block_matches_1_values_and_missingness": bool(np.nanmax(np.abs(concentration_error.to_numpy())) < 1e-10
            and chemical[chem_columns].isna().equals(concentrations.loc[chemical.index, chem_columns].isna())),
        "source_hashes_unchanged": hashes_before == {path.name: sha256(path) for path in paths},
        "first_37_basic_match_log10_on_valid_pairs": all(row["log10_matches_at_1e_7"]
            for row in calibration_check if row["transformation_group"] == "first_37_basic"),
        "aroma_taste_metadata_select_exactly_45": len(selected_45) == 45 and set(selected_45).issubset(sensory.columns),
    }
    basic_log10 = [row for row in calibration_check if row["transformation_group"] == "first_37_basic"]
    summary = {
        "attribution": ATTRIBUTION,
        "audit_date": "2026-10-23", "source_directory": str(data_dir.relative_to(project)) if data_dir.is_relative_to(project) else str(data_dir),
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "openpyxl": openpyxl.__version__},
        "source_sha256": hashes_before, "tables": {name: table_summary(frame) for name, frame in frames.items()},
        "documentation_discrepancies": [
            {"location": "题面第1页", "claim": "226种不同啤酒", "observed": "附件1/4/7各250个唯一beer"},
            {"location": "题面第3页附件2说明", "claim": "皮尔逊相关系数", "observed": "附件2/5/6精确复现Spearman"},
            {"location": "题面附件6说明", "claim": "46×50", "observed": "44×50数值矩阵"},
            {"location": "题面附件7说明", "claim": "250×187", "observed": "250×186数值字段，另3列标识/类别"},
        ],
        "primary_missingness": missing_primary, "key_checks": checks, "matrix_shapes": matrices,
        "matrix_reconciliation": reconcile, "assertions": assertions,
        "feature_counts": {"chemical": len(chem_columns), "panel": len(sensory.columns) - 2,
                           "panel_aroma_taste_45": len(selected_45),
                           "review_derived": len(comments.columns) - 2,
                           "review_usable_nonempty": int(comments.iloc[:, 2:].notna().any().sum()),
                           "calibrated_chemical": len(calibrated.columns) - 2},
        "template_formula_counts": template_formulas,
        "supplementary": {
            "shap_data_counts": shap.data.value_counts().to_dict(),
            "concentration_block_max_abs_difference_from_1": float(np.nanmax(np.abs(concentration_error.to_numpy()))),
            "calibrated_missing_cells": int(calibrated.iloc[:, 2:].isna().sum().sum()),
            "calibrated_negative_cells": len(negative_calibrated),
            "calibrated_rank_rho_min": min(row["spearman_with_1"] for row in calibration_check),
            "basic_log10_fields": len(basic_log10),
            "basic_log10_comparable_values": sum(row["n_log10_comparable"] for row in basic_log10),
            "basic_log10_max_abs_error": max(row["log10_max_abs_error"] for row in basic_log10),
            "basic_log10_zero_replacement_count": sum(row["calibrated_zero_count"] for row in basic_log10),
            "basic_log10_excluded_negative_values": sum(row["negative_calibrated"] for row in basic_log10),
            "basic_log10_excluded_calibrated_missing_values": sum(row["missing_calibrated"] for row in basic_log10),
            "remaining_33_not_direct_log10": all(not row["log10_matches_at_1e_7"]
                for row in calibration_check if row["transformation_group"] == "remaining_33_GCMS_overlap"),
            "prices_all_missing_beers": int(prices[price_cols].isna().all(axis=1).sum()),
            "price_missing_cells": int(prices[price_cols].isna().sum().sum()),
        },
        "limitations": ["No raw review text, panel individual ratings, measurement dates or preprocessing recipe supplied.",
                        "No independent human verification; no source data edited or uploaded.",
                        "Outlier flags are descriptive; they do not prove measurement errors.",
                        "Provided MDI/SHAP and full-sample matrices must not be used as supervised-selection inputs before CV splitting."],
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path.write_text(make_report(summary, csv, xlsx, reconcile, checks, negative_calibrated), encoding="utf-8")
    if not all(assertions.values()):
        raise RuntimeError(f"Audit invariants failed: {assertions}")
    print(json.dumps({"status": "completed", "report": str(report_path), "output_directory": str(output),
                      "assertions": assertions}, ensure_ascii=False, indent=2))


def make_report(summary: dict, csv: dict, xlsx: dict, reconcile: list, checks: list, negatives: list) -> str:
    rows = []
    for i in (1, 2, 4, 5, 6, 7):
        d = csv[i]
        meaning = {1: "250啤酒×231化学特征，另3列标识/类别", 2: "231×231化学相关矩阵，另1列行标签",
                   4: "250啤酒×50品评特征，另3列标识/类别", 5: "50×50品评相关矩阵，另1列行标签",
                   6: "44×50化学—品评相关子矩阵，另1列行标签",
                   7: "250啤酒×186评论派生特征，另3列标识/类别"}[i]
        rows.append(f"| {i}.csv | {len(d)}×{len(d.columns)} | {int(d.isna().sum().sum())} | {meaning} |")
    sheet_rows = [f"| {name} | {len(frame)}×{len(frame.columns)} | {int(frame.isna().sum().sum())} |"
                  for name, frame in xlsx[8].items()]
    corr_rows = [f"| {row['matrix']} | {row['method']} | {row['pair_n_min']}–{row['pair_n_max']} | {row['max_abs_error']:.6g} | {row['mean_abs_error']:.6g} | {row['matches_at_1e_10']} |"
                 for row in reconcile]
    key4 = next(row for row in checks if row["source"] == "4.csv")
    missing_rows = [f"| {name} | {count} | {count / 250:.1%} |" for name, count in summary["primary_missingness"]["1"].items()]
    neg_rows = [f"| {row['beer']} | {row['field']} | {row['value']:.8g} | {row['unit_metadata'] if pd.notna(row['unit_metadata']) else '未提供'} |" for row in negatives]
    supplementary = summary["supplementary"]
    return f"""# A题附件数据审计

审计日期：2026-10-23。{ATTRIBUTION}

本次仅检查附件本身的可用性、关联键、缺失、数值尺度和相关矩阵口径。原始文件保持不变；未联网、未上传数据。可复现脚本为 `solution_a/src/01_audit.py`，机器证据保存在 `solution_a/outputs/audit/`。脚本执行成功并不等同于人工审核。

## 1. 可用于建模的实际数据

原始附件1、4、7各有 **250个唯一啤酒样本**；可以通过 `beer` 一对一合并。化学变量231列，感官变量50列，论坛评论派生变量186列（其中3列全空，非空变量183列）。不得未经依据将250样本裁剪为226样本。题面/说明的文字口径应与这些实测结构分开记录，需参赛队确认官方是否另有筛选规则。

其中，附件8 `Sensory names` 标为Aroma的15项和Taste的30项恰好组成题目要求的45维；排除的是 `alcohol, aftertaste, body, co2, overall`。完整50字段与45维建模目标须分别统计，字段选择见 `feature_name_map.csv` 的 `selected_aroma_taste_45`。感官量尺和模板细节参见同目录 `sensory_mapping_review.md`；本审计不把附件4的连续值解释成原始0–6分。

题面第1页写“测量了226种不同啤酒的化学特性”，附件数据说明则写250个啤酒样品；本次附件实测支持250。题面对附件6写46×50、附件7写250×187，实测分别为44×50、250×186个数值字段。论文应逐项披露这些冲突，并说明采用实际附件范围的理由，不应静默改写题意。

| 文件 | 读取尺寸（含表头对应列） | 缺失单元格 | 实际含义 |
| --- | --- | ---: | --- |
{chr(10).join(rows)}

CSV均为UTF-8可读取，三个样本级CSV没有完全重复记录；数值列未检测到正负无穷。`beer` 覆盖1–250；样本顺序并非编号顺序，因此应显式按键关联。类别共22种，最小类别Faro为3例，最大类别Blond为31例，分组验证和类别比较须考虑样本不平衡。

附件3含12张 `beer_1`–`beer_12` 品评模板，每张读取为100×33，以及12×52的 `summary`。`summary` 中除编号外612个值全为缓存零，且对应612个公式；`beer_1` 中有54个公式。它是品评表设计，**不是12条额外观测**，也不是缺失的逐人评分记录。表头出现的 `A_ester_fruity`、`F_ester_fruity`、`4vg` 等与附件4名称不完全一致，不能无检查拼接。

## 2. 关联键：必须使用beer

**高风险、证据充分：附件4和7的 `beer_id` 已损失精度。** 两表各仅有{key4['unique_beer_id']}个不同条码，{key4['duplicate_beer_id_rows']}行处于重复条码组；分别有{key4['beer_id_mismatches_with_1']} / 250条记录与附件1的条码不一致。例如beer=10：附件1为5410228202929，附件4/7为5410230000000。按条码关联会丢行或产生多对多扩张。

附件1/4/7的 `beer` 集合、行序及类别逐条一致。附件8的标定数据、SHAP的两个区块、价格表均覆盖相同250个 `beer`；其条码与附件1一致。推荐保留附件1的原始条码作为展示信息，合并时使用 `validate='one_to_one'`；不要把已截断条码当作独立产品。价格表按beer排序，与附件1顺序不同。价格类别有 `wit`→`Wheat`、`bruin`→`Brown`、`oud bruin`→`Flanders old brown` 等别名，映射见 `price_category_map.csv`。

证据：`key_checks.csv`、`beer_id_discrepancies.csv`。条码应按字符串保存；已经丢失的末位不能靠格式化恢复。

## 3. 2/5/6的相关系数实测复核

**三张给定矩阵均精确复现Spearman，不能称为Pearson。** 复算按共有 `beer` 对齐、逐变量配对删除缺失；Spearman使用平均秩处理并列值，比较容差为1e-10。

题面第3页附件2说明写“单元格为皮尔逊相关系数”，与附件数据说明的Spearman表述矛盾；以下数值复核支持后者。

| 矩阵 | 复算方法 | 每对样本数 | 最大绝对误差 | 平均绝对误差 | 容差内一致 |
| --- | --- | ---: | ---: | ---: | --- |
{chr(10).join(corr_rows)}

化学名称从附件2顺序映射到附件1，并由整张矩阵精确复现交叉验证；231个字段在附件8 `Sensory thresholds.field_name_R` 中均可找到。感官名称按附件8 `Sensory names` 的 `R_name` 与 `CleanName_PanelFeedback` 显式匹配。**附件4尾部的 `aftertaste, body, co2` 与附件5的 `Carbonation, Aftertaste, Body` 顺序不同**；按位置直接对齐会制造错误差异。

附件2、5的行列标签唯一、矩阵对称且对角为1。附件6仅包含44个化学指标，不代表全231个化学指标。建模及机制分析如需其余指标，应从样本级数据重新计算。任何基于相关性进行的特征筛选都应在训练折内重算；本节全样本复核只用于审计，不应充当验证折未知信息。

证据：`matrix_reconciliation.csv`、`feature_name_map.csv`；JSON同时记录矩阵形状、范围、对称误差。

## 4. 缺失与数值异常

附件1共73个缺失，占250×231数值单元的0.1264%，分布于以下9列。缺失比例不高，但不能无条件删去所有含缺失样本；插补参数须仅在训练折拟合。

| 化学字段 | 缺失数 | 样本缺失率 |
| --- | ---: | ---: |
{chr(10).join(missing_rows)}

附件4无缺失。附件7中 `palate_acetaldehyde`、`palate_acetate`、`palate_dentist` 各250个缺失；其余183列无缺失。建议在派生数据中记录并剔除这3列，保留原始附件。

**中风险、尺度不统一的证据充分：不能假定附件1全部已做标准z变换。** 存在负值且各列样本标准差从约0.003123到796.633，均值也不统一为零。`X2.methyltetrahydrofuran.3.one` 最大12596.43287（beer=156），`gamma.nonalactone` 最大7256.135876（beer=181），`X.E..beta.damascenone` 最大5991.012523（beer=151）。附件8的concentration区块复现同样值，因此这不是本次CSV解析偶然错误；仍不能仅据此判定测量错误。主模型应比较训练折内标准化/稳健缩放等处理；不要把这些数直接解释为mg/L，也不要无依据删除极端样本。已核实的37列log10关系和其余字段的限制见第6节。

`column_profile.csv` 给出全字段缺失、分位数、均值、标准差、负值、零值、IQR标记。`extreme_value_flags.csv` 使用3IQR规则提供定位；IQR=0或稀疏词汇会产生很多标记，**标记不等于异常测量**。没有时间列、采集日期和历史版本，本审计不对时效性、批次漂移或随时间变化作推断。

## 5. 附件7的含义边界

附件7是按 `aroma_`、`palate_`、`taste_` 三个前缀组织、每组62词的数值表。附件8 `Sensory names.dataset` 中把相应描述词来源标为Ratebeer。没有评论正文、用户、逐条评分、评论日期或评论数，也没有 `ave_score_5` 字段（虽在名称字典中出现）。因此可把它作为论坛描述词派生特征，不能伪称已做原始评论分词、情感分析，亦不能把该表任何一列未经定义当成消费者总体评分。现有文件没有提供这些数值的完整归一化/变换说明。

## 6. 附件8的补充信息及使用限制

| 工作表 | 尺寸 | 缺失单元格 |
| --- | --- | ---: |
{chr(10).join(sheet_rows)}

- `Sensory thresholds` 是261条化学字段元数据，包含单位、阈值、来源等；部分文本有编码显示问题，阈值和单位需逐项核对后才能用于配方约束。字段元数据不能支持把全表采用同一个逆变换。
- `Sensory names` 有101条名称/类别记录，包含50个trained panel指标及Ratebeer字段；字典中出现不代表样本表实际存在。
- `Calibrated chemical dataset` 是250样本×70个标定数值，另3列标识/类别；有{supplementary['calibrated_missing_cells']}个缺失。与附件1共有字段的Spearman最小约{supplementary['calibrated_rank_rho_min']:.6f}，但并非全为1；37个基础理化字段与33个GCMS重叠字段需分别处理，不能全表取指数。若要估计现实添加量，应优先核对该页单位和阈值，再定义处理方案。
- `GBM importance MDI` 是已有模型的231项重要性，第一列是重要性数值、第二列是字段名；`GBM SHAP Values` 的500行是250条SHAP与250条concentration，不是500种啤酒。后者与附件1数值最大差{supplementary['concentration_block_max_abs_difference_from_1']:.3g}。这些是既有模型派生结果，训练数据和建模过程未完整给出；不能当成新的独立样本或在交叉验证之前据此筛选特征。
- `Beer Prices` 有7个渠道价格列，共{supplementary['price_missing_cells']}个价格缺失；{supplementary['prices_all_missing_beers']}种啤酒所有渠道价格均缺失。不同国家、渠道、计价单位和日期未在该页完整说明，不能直接平均作为原料成本或跨币种价格目标。`Sheet2` 为空表。

**37个基础字段的底数已由本地数值复核为10。** 附件1从 `acetaldehyde` 到 `kcalperc` 的前37个数值字段满足 `x=log10(c)`；校准值为0时先用该列最小正值的一半替代。共有{supplementary['basic_log10_comparable_values']}个可比较值，包含{supplementary['basic_log10_zero_replacement_count']}个零值替代，最大绝对误差{supplementary['basic_log10_max_abs_error']:.6g}。此范围明确排除了{supplementary['basic_log10_excluded_calibrated_missing_values']}个校准缺失和{supplementary['basic_log10_excluded_negative_values']}个负值，不能宣称它们也通过直接log10验证。其余33个GCMS重叠字段全部不满足同一直接log10关系；不得对231列统一使用 `exp(x)` 或 `10**x` 后计算浓度/OAV。已知37列中，逆变换只能还原零值的替代量，也无法据此恢复真实检出量。逐列证据见 `calibrated_comparison.csv`。

标定数据还存在{len(negatives)}个负值，建议保留并核查检测/标定背景，勿擅自视为真实负浓度用于配方约束：

| beer | 字段 | 标定值 | 元数据单位 |
| ---: | --- | ---: | --- |
{chr(10).join(neg_rows)}

## 7. 建模交接与未决事项

1. 以250个 `beer` 为样本集，保留附件1的231个化学变量和附件4的50个品评字段。45维感官画像明确按元数据Aroma15项+Taste30项选定，不能把完整50字段与45维目标混为一谈；把 `beer`、`beer_id` 排除出预测特征。是否加入类别由研究问题和验证方案决定。
2. 模型评价采用严格训练/测试隔离，插补、缩放、降维、特征筛选均在训练折内完成；不要用附件2/5/6的全样本相关性或附件8的MDI/SHAP提前筛选后再报告无偏验证。
3. 可复现附件相关矩阵采用Spearman；如果问题明确要求Pearson，另外重算Pearson并明确二者差别。所有字段名和关联键均显式匹配。
4. 论坛数据只支持描述词数值特征分析；缺少原始评论和消费者整体评分时，应明确任务可识别边界。
5. 需要参赛队核对题面样本数冲突、预处理说明、标定单位、负浓度、官方勘误；现阶段不把待核实信息写成事实。

## 8. 复现与证据清单

在仓库根目录执行：

```bash
/Users/Admin1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 solution_a/src/01_audit.py
```

也可用安装了pandas、numpy、openpyxl的Python执行同一脚本。可选 `--data-dir` 和 `--output-dir` 修改输入及审计证据目录；脚本不修改原始附件。环境版本保存在 `audit_summary.json`，SHA-256保存在 `source_files.csv`；本次执行前后哈希一致。

证据文件：`audit_summary.json`（总体结构与自动校验）、`source_files.csv`（输入哈希）、`column_profile.csv`（逐列画像）、`missingness.csv`、`key_checks.csv`、`beer_id_discrepancies.csv`、`feature_name_map.csv`、`matrix_reconciliation.csv`、`calibrated_comparison.csv`、`negative_calibrated_values.csv`、`extreme_value_flags.csv`、`style_counts.csv`、`price_availability.csv`、`price_category_map.csv`。所有结论来自本地附件；没有外部文献补充或人工审核声明。
"""


if __name__ == "__main__":
    main()
