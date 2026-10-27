#!/usr/bin/env python3
"""Q2 descriptive association baseline.

Run from any directory with the repository .venv Python. Data transformations,
selection rules, and validation scope are recorded in outputs/q2/README.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题" / "2026年MathorCup大数据挑战赛-赛道A初赛" / "附件"
OUTPUT = ROOT / "solution_a" / "outputs" / "q2"
EXCLUDE_X = {
    "kcalperc", "esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum",
    "bitterness", "color.EBC.", "pH", "protein.g.L.", "betaglucan.mg.L.",
    "GFS", "unfermentables", "CO2.PSI.",
}
EXCLUDE_Y = {"alcohol", "aftertaste", "body", "co2", "overall"}


def unit_columns(a: np.ndarray) -> np.ndarray:
    centered = a - a.mean(axis=0)
    denominator = np.sqrt(np.sum(centered * centered, axis=0))
    return np.divide(centered, denominator, out=np.zeros_like(centered), where=denominator > 0)


def cross_correlation(x: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    """Pairwise-complete correlation, reranking y when an x column is missing.

    Every target must be observed. Ranking each x column independently is not
    sufficient for pairwise Spearman: y must also be reranked on the same rows.
    Constant columns are returned as NaN and cannot enter a Top-3.
    """
    if not np.isfinite(y).all():
        raise ValueError("Targets contain missing/nonfinite values; revise pairwise handling.")
    result = np.full((x.shape[1], y.shape[1]), np.nan)
    complete = np.isfinite(x).all(axis=0)
    xx, yy = x[:, complete], y
    if method == "spearman":
        xx, yy = rankdata(xx, axis=0), rankdata(yy, axis=0)
    result[complete] = unit_columns(xx).T @ unit_columns(yy)
    result[np.flatnonzero(complete)[np.ptp(xx, axis=0) == 0], :] = np.nan
    result[:, np.ptp(yy, axis=0) == 0] = np.nan
    for j in np.flatnonzero(~complete):
        keep = np.isfinite(x[:, j])
        if keep.sum() < 3:
            continue
        xx, yy = x[keep, j:j + 1], y[keep]
        if method == "spearman":
            xx, yy = rankdata(xx, axis=0), rankdata(yy, axis=0)
        if np.ptp(xx) == 0:
            continue
        result[j] = (unit_columns(xx).T @ unit_columns(yy))[0]
        result[j, np.ptp(yy, axis=0) == 0] = np.nan
    return np.clip(result, -1, 1)


def top3(correlation: np.ndarray) -> np.ndarray:
    """Stable input-order tie break; missing/constant correlations rank last."""
    if np.any(np.isfinite(correlation).sum(axis=0) < 3):
        raise ValueError("Fewer than three eligible candidates for a target.")
    return np.argsort(-np.nan_to_num(np.abs(correlation), nan=-1), axis=0, kind="stable")[:3]


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = set(a.tolist()), set(b.tolist())
    return len(aa & bb) / len(aa | bb)


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUTPUT / name, index=False, encoding="utf-8-sig", float_format="%.10g", lineterminator="\n")


def main(bootstraps: int, seed: int) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    chemical = pd.read_csv(DATA / "1.csv")
    sensory = pd.read_csv(DATA / "4.csv")
    supplied = pd.read_csv(DATA / "2.csv", index_col=0)
    metadata = pd.read_excel(DATA / "8.xlsx", sheet_name="Sensory thresholds")
    metadata = metadata.dropna(subset=["field_name_R"]).drop_duplicates("field_name_R").set_index("field_name_R")
    features = chemical.columns[3:].tolist()
    targets = [c for c in sensory.columns[3:] if c not in EXCLUDE_Y]
    assert len(features) == 231 and len(targets) == 45
    assert chemical.beer.is_unique and sensory.beer.is_unique
    assert supplied.index.tolist() == supplied.columns.tolist()
    merged = chemical.merge(sensory[["beer", *targets]], on="beer", validate="one_to_one", how="inner")
    assert len(merged) == len(chemical) == len(sensory) == 250
    # Attachments retain the same numeric feature order; verify using every pair.
    actual_spearman = chemical[features].corr(method="spearman").to_numpy()
    actual_pearson = chemical[features].corr(method="pearson").to_numpy()
    r = supplied.to_numpy(dtype=float)
    upper = np.triu_indices(len(features), 1)
    delta_s = np.abs(actual_spearman[upper] - r[upper])
    delta_p = np.abs(actual_pearson[upper] - r[upper])
    assert float(np.nanmax(delta_s)) < 1e-10, "Attachment order/definition changed."
    pretty = dict(zip(features, supplied.index.astype(str)))
    candidates = [c for c in features if c not in EXCLUDE_X]
    x, y = merged[candidates].to_numpy(float), merged[targets].to_numpy(float)
    rho = cross_correlation(x, y, "spearman")
    pearson = cross_correlation(x, y, "pearson")
    # Check the optimized pairwise routine against scipy, including missingness.
    check_columns = np.unique(np.r_[np.arange(5), np.flatnonzero(~np.isfinite(x).all(axis=0))])
    max_check_error = 0.0
    for j in check_columns:
        keep = np.isfinite(x[:, j])
        for t in [0, 20, 44]:
            for matrix, function in [(rho, spearmanr), (pearson, pearsonr)]:
                reference = function(x[keep, j], y[keep, t]).statistic
                max_check_error = max(max_check_error, abs(float(reference - matrix[j, t])))
    assert max_check_error < 1e-12

    node_table = pd.DataFrame({"field": features, "chemical_name": [pretty[f] for f in features], "eligible_top3": [f in candidates for f in features]})
    for col in ["origin_paper", "unit", "calibrated", "off_flavors", "threshold_mean", "threshold_median"]:
        node_table[col] = metadata[col].reindex(features).to_numpy()
    network_summary = []
    for threshold in [0.6, 0.7, 0.8]:
        adjacency = np.abs(r) >= threshold
        np.fill_diagonal(adjacency, False)
        left, right = np.where(np.triu(adjacency, 1))
        weights = r[left, right]
        save_csv(pd.DataFrame({
            "source": np.asarray(features)[left], "target": np.asarray(features)[right],
            "source_name": supplied.index.to_numpy()[left], "target_name": supplied.index.to_numpy()[right],
            "spearman_rho": weights, "absolute_weight": np.abs(weights), "sign": np.sign(weights).astype(int),
        }), f"network_edges_rho_{threshold:.1f}.csv")
        node_table[f"degree_{threshold:.1f}"] = adjacency.sum(axis=1)
        node_table[f"strength_{threshold:.1f}"] = (np.abs(r) * adjacency).sum(axis=1)
        network_summary.append({"absolute_rho_threshold": threshold, "edges": len(weights), "positive_edges": int((weights > 0).sum()), "negative_edges": int((weights < 0).sum()), "isolated_nodes": int((adjacency.sum(axis=1) == 0).sum())})
    save_csv(node_table, "network_nodes.csv")

    initial_top = top3(rho)
    pearson_top = top3(pearson)
    # This is a sensitivity check, not a causal adjustment or formal partial rho.
    within_x = merged[candidates] - merged.groupby("tasting_category_fine")[candidates].transform("mean")
    within_y = merged[targets] - merged.groupby("tasting_category_fine")[targets].transform("mean")
    within_rho = cross_correlation(within_x.to_numpy(float), within_y.to_numpy(float), "spearman")
    within_top = top3(within_rho)
    rng = np.random.default_rng(seed)
    selections = np.zeros_like(rho, dtype=int)
    draws = np.empty((bootstraps, *rho.shape), dtype=float)
    for b in range(bootstraps):
        rows = rng.integers(0, len(merged), len(merged))
        draws[b] = cross_correlation(x[rows], y[rows], "spearman")
        winners = top3(draws[b])
        selections[winners, np.arange(len(targets))] += 1
        if (b + 1) % 50 == 0:
            print(f"bootstrap {b + 1}/{bootstraps}", flush=True)
    frequency = selections / bootstraps
    assert np.all(selections.sum(axis=0) == 3 * bootstraps)
    lower, upper_ci = np.nanquantile(draws, [0.025, 0.975], axis=0)
    valid_draws = np.isfinite(draws).sum(axis=0)
    sign_agreement = ((np.sign(draws) == np.sign(rho)[None]) & np.isfinite(draws)).sum(axis=0) / valid_draws
    global_ranks = np.empty_like(initial_top, shape=rho.shape)
    within_ranks = np.empty_like(global_ranks)
    for t in range(len(targets)):
        global_ranks[np.argsort(-np.abs(rho[:, t]), kind="stable"), t] = np.arange(1, len(candidates) + 1)
        within_ranks[np.argsort(-np.abs(within_rho[:, t]), kind="stable"), t] = np.arange(1, len(candidates) + 1)
    rows = []
    method_rows = []
    for t, target in enumerate(targets):
        method_rows.append({
            "sensory_target": target,
            "spearman_top3": " | ".join(candidates[j] for j in initial_top[:, t]),
            "pearson_top3": " | ".join(candidates[j] for j in pearson_top[:, t]),
            "within_style_top3": " | ".join(candidates[j] for j in within_top[:, t]),
            "jaccard_spearman_pearson": jaccard(initial_top[:, t], pearson_top[:, t]),
            "jaccard_global_within_style": jaccard(initial_top[:, t], within_top[:, t]),
        })
        for j, candidate in enumerate(candidates):
            rows.append({
                "sensory_target": target, "rank": int(global_ranks[j, t]), "compound": candidate,
                "chemical_name": pretty[candidate], "n_observed_pairs": int(np.isfinite(x[:, j]).sum()),
                "spearman_rho": rho[j, t], "pearson_r": pearson[j, t],
                "bootstrap_top3_count": int(selections[j, t]), "bootstrap_top3_frequency": frequency[j, t],
                "bootstrap_rho_lower_95": lower[j, t], "bootstrap_rho_upper_95": upper_ci[j, t],
                "bootstrap_sign_agreement": sign_agreement[j, t], "bootstrap_valid_draws": int(valid_draws[j, t]),
                "within_style_spearman_rho": within_rho[j, t], "within_style_rank": int(within_ranks[j, t]),
                "is_pearson_top3": bool(j in pearson_top[:, t]),
                "is_within_style_top3": bool(j in within_top[:, t]),
            })
    scores = pd.DataFrame(rows).sort_values(["sensory_target", "rank"])
    top_table = scores[scores["rank"] <= 3].copy()
    top_table = top_table.merge(node_table, left_on="compound", right_on="field", how="left", validate="many_to_one", suffixes=("", "_network")).drop(columns=["field", "chemical_name_network"])
    methods = pd.DataFrame(method_rows)
    save_csv(scores, "all_candidate_stability.csv")
    save_csv(top_table, "top3_compounds.csv")
    save_csv(methods, "method_sensitivity.csv")
    summary = {
        "scope": "Full-sample descriptive associations; neither causal evidence nor held-out predictive validation.",
        "n_beers": len(merged), "n_features": len(features), "n_candidate_compounds": len(candidates),
        "n_sensory_targets": len(targets), "bootstrap_draws": bootstraps, "seed": seed,
        "excluded_compounds": sorted(EXCLUDE_X), "excluded_sensory_targets": sorted(EXCLUDE_Y),
        "correlation_audit": {"supplied_vs_spearman_max_abs_error": float(np.nanmax(delta_s)), "supplied_vs_spearman_mean_abs_error": float(np.nanmean(delta_s)), "supplied_vs_pearson_mean_abs_error": float(np.nanmean(delta_p)), "supplied_vs_pearson_max_abs_error": float(np.nanmax(delta_p)), "optimized_routine_vs_scipy_max_abs_error": max_check_error},
        "networks": network_summary,
        "top3_stability": {"mean_selection_frequency": float(top_table.bootstrap_top3_frequency.mean()), "median_selection_frequency": float(top_table.bootstrap_top3_frequency.median()), "number_frequency_at_least_0_8": int((top_table.bootstrap_top3_frequency >= .8).sum()), "total_selected": len(top_table), "max_frequency_monte_carlo_standard_error": float(np.sqrt(.25 / bootstraps))},
        "mean_jaccard_spearman_pearson": float(methods.jaccard_spearman_pearson.mean()),
        "mean_jaccard_global_within_style": float(methods.jaccard_global_within_style.mean()),
        "targets_with_no_global_within_overlap": int((methods.jaccard_global_within_style == 0).sum()),
        "style_counts": {str(k): int(v) for k, v in merged.tasting_category_fine.value_counts().items()},
        "limitations": ["No causal interpretation", "No per-candidate mechanism literature verification yet", "All-sample network is not an external independent prior", "Global row bootstrap does not fix style composition", "Mean-centered within-style rank correlation is a sensitivity analysis, not full confounding control", "Continuous standardized targets do not supply raw 0-6 calibration", "Graph provides structure/context; this baseline ranks marginal correlations and does not fit a graph-regularized predictive model"],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (OUTPUT / "README.md").write_text(f"""# 第二问首轮关联分析

本目录由 `solution_a/src/03_q2_network.py` 生成，计算结果与 scipy 成对相关计算进行数值核对。重跑：

```bash
.venv/bin/python solution_a/src/03_q2_network.py --bootstraps {bootstraps} --seed {seed}
```

数据为按 beer 一对一连接的 250 款啤酒；全图 231 节点、Top-3 候选 218 项、感官目标 45 项。所有计算在附件既有数值表示上进行，没有统一取指数或把连续标签伪还原为 0–6 分。

| 文件 | 用途 |
|---|---|
| network_nodes.csv | 全节点名映射、来源/单位/阈值及各阈值度数、加权度数 |
| network_edges_rho_*.csv | 绝对相关阈值 0.6/0.7/0.8 的有符号网络 |
| top3_compounds.csv | 每个感官维度按绝对 Spearman 相关选择的 3 个候选 |
| all_candidate_stability.csv | 全部 218×45 候选的相关、bootstrap 入选频率、分位区间和风格敏感性 |
| method_sensitivity.csv | Pearson 替代和去风格均值后的 Top-3/Jaccard |
| summary.json | 样本、阈值、网络、稳定性、数值校验及限制摘要 |

2.csv 与全样本 Spearman 矩阵最大差 {np.nanmax(delta_s):.3g}，故它是当前样本导出的描述性网络；预测交叉验证时必须仅在训练折重建。全样本图可解释 Top-3 的邻域和相关冗余，但本轮未拟合图正则化模型。阈值 0.6/0.7/0.8 是稀疏化参数，不是显著性阈值。

Bootstrap 为 {bootstraps} 次全样本行重采样，重新排名，固定种子 {seed}，频率最大蒙特卡洛标准误 {np.sqrt(.25 / bootstraps):.4f}。并列时按附件特征顺序打破。区间为相关系数的描述性 bootstrap 分位数，不是选择后推断保证。去风格均值对 X、Y 都在当前表示上按风格减均值后计算 Spearman，只检查风格间均值混杂。

本目录结论是**关联候选，不是因果因素**；稳定性不等于化学合理性。`8.xlsx` 元信息仅作附件证据，逐条机制文献验证尚未完成。`5.csv/6.csv` 和预计算 GBM/SHAP 未用于筛选。Top-3 的所有 135 项中，入选频率 ≥0.8 的有 {summary['top3_stability']['number_frequency_at_least_0_8']} 项；Spearman/Pearson 平均 Jaccard 为 {summary['mean_jaccard_spearman_pearson']:.3f}，去风格均值前后为 {summary['mean_jaccard_global_within_style']:.3f}。这些是全样本解释稳定性，不是未见啤酒预测性能。
""", encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstraps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    if args.bootstraps < 1:
        parser.error("--bootstraps must be positive")
    main(args.bootstraps, args.seed)
