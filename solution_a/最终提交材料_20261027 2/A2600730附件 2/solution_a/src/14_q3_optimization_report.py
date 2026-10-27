"""Summarize nested generation, weak-proxy experiments and robust transfer."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import SOLUTION
from q3_data import load_q3_data

OUT = SOLUTION / "outputs/q3_optimized"
OLD = SOLUTION / "outputs/q3"
STYLES = ["Tripel", "Kriek", "Lambic", "Pils/Lager"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    meta, x, y, manifest = load_q3_data()
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    generation = read_json(OUT / "generation/summary.json")
    search = read_json(OUT / "stale/summary.json")
    fixed = read_json(OUT / "stale_fixed/summary.json")
    transfer = read_json(OUT / "transfer/summary.json")
    gen_cv = pd.read_csv(OUT / "generation/cv_summary.csv")
    transfer_table = pd.read_csv(OUT / "transfer/transfer_summary.csv")
    default = transfer_table[transfer_table.is_default_policy].sort_values("source_beer")
    interval = fixed["paired_cluster_bootstrap"]
    fixed_metrics = pd.DataFrame(fixed["pooled_repeat_metrics"]).set_index("method")
    search_metrics = pd.DataFrame(search["pooled_repeat_metrics"]).set_index("method")
    source_holdout = pd.DataFrame(fixed["pils_holdout_metrics"]).set_index("model")
    profile_model = joblib.load(OLD / "sensory/sensory_model.joblib")
    stale_model = joblib.load(OUT / "stale_fixed/transfer_stale_model.joblib")
    profiles, profile_rows = [], []
    for style in STYLES:
        generated = pd.read_csv(OUT / f"generation/generated_{style.replace('/', '-')}.csv")
        pred = profile_model["pipeline"].predict(generated[x.columns])
        pred[:, y.columns.get_loc("stale_hops")] = stale_model["pipeline"].predict(generated[x.columns])
        profile = pd.DataFrame(pred, columns=y.columns)
        profile.insert(0, "target_style", style)
        profile.insert(0, "sample_id", generated.sample_id)
        profiles.append(profile)
        for j, target in enumerate(y.columns):
            profile_rows.append({"style": style, "target": target,
                                 "predicted_mean": float(pred[:, j].mean()),
                                 "predicted_p05": float(np.quantile(pred[:, j], .05)),
                                 "predicted_p95": float(np.quantile(pred[:, j], .95)),
                                 "real_observed_mean": float(y.loc[meta.tasting_category_fine.eq(style), target].mean()),
                                 "scale": "attachment_4_standardized",
                                 "prediction_model": "fixed_stale_forest_239_non_pils" if target == "stale_hops"
                                                     else "original_45_output_forest_250",
                                 "interval_meaning": "generated_spectrum_spread_not_confidence_interval"})
    pd.concat(profiles, ignore_index=True).to_csv(OUT / "generated_sensory_profiles.csv", index=False, lineterminator="\n")
    new_profiles = pd.DataFrame(profile_rows)
    new_profiles.to_csv(OUT / "generated_sensory_summary.csv", index=False, lineterminator="\n")
    old_profiles = pd.read_csv(OLD / "generated_sensory_summary.csv")

    # Aggregate chemical rechecks independently of the proxy model counts.
    old_grid = pd.read_csv(OUT / "transfer/archived_threshold_recheck.csv")
    new_grid = pd.read_csv(OUT / "transfer/threshold_sensitivity.csv")
    scenario_rows = []
    for scenario in transfer["threshold_scenarios"]:
        name = scenario["scenario"]
        archived = old_grid[old_grid.scenario.eq(name)]
        current = new_grid[new_grid.scenario.eq(name)].groupby("transfer_id").passed.all()
        current_checks = default.transfer_id.map(current).fillna(False)
        scenario_rows.append({"scenario": name, "statistic": scenario["statistic"], "factor": scenario["factor"],
                              "old_model_checks_and_chemical": int((archived.old_model_checks_passed & archived.chemical_passed).sum()),
                              "new_chemical_only": int(current_checks.sum()),
                              "new_model_checks_and_chemical": int((current_checks & default.all_model_checks_passed).sum()),
                              "cross_proxy_checks_and_chemical": int((current_checks & default.cross_proxy_checks_passed).sum())})
    scenario_counts = pd.DataFrame(scenario_rows)
    scenario_counts.to_csv(OUT / "transfer/threshold_pass_counts.csv", index=False, lineterminator="\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for offset, method, label, color in [(-.24, "legacy_copula", "Old copula", "#9CA1A8"),
                                        (0, "nested_selected", "Optimized selection", "#2877A6"),
                                        (.24, "style_bootstrap", "Style bootstrap", "#C68B2B")]:
        part = gen_cv[gen_cv.method.eq(method)].set_index("style").loc[["Kriek", "Lambic"]]
        axes[0].bar(np.arange(2) + offset, part.energy_pca5_mean, .22, color=color,
                    yerr=part.energy_pca5_std, capsize=3, label=label,
                    error_kw={"elinewidth": 1, "ecolor": "#555B62"})
    axes[0].set_xticks([0, 1], ["Kriek", "Lambic"])
    axes[0].set_ylabel("Energy discrepancy (V-statistic)")
    axes[0].set_title("Style generation on matched outer splits", loc="left", pad=34)
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.005))
    method_specs = [("mean", "Mean", "#C4C7CC"),
                    ("old_multioutput_extra_trees", "Old 45-output", "#9CA1A8"),
                    ("fixed_single_extra_trees", "Dedicated tree", "#2877A6")]
    for offset, (method, label, color) in zip([-.24, 0, .24], method_specs):
        values = [search_metrics.loc[method, "rmse"], fixed_metrics.loc[method, "rmse"]]
        axes[1].bar(np.arange(2) + offset, values, .22, color=color, label=label)
        for i, value in enumerate(values):
            axes[1].text(i + offset, value + .001, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks([0, 1], ["Search-stage splits", "Follow-up splits"])
    axes[1].set_ylabel("Stale hops pooled repeated OOF RMSE")
    axes[1].set_ylim(0, .52)
    axes[1].set_title("Prediction gains remain uncertain", loc="left", pad=34)
    axes[1].legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.005))
    for ax in axes:
        ax.grid(axis="y", color="#E5E8EC", linewidth=.7)
        ax.set_axisbelow(True)
    fig.text(.08, .025, "Generation: 2 x 3 outer folds, training-only method selection; bars show mean and descriptive fold SD.\n"
             "Prediction: two repeated five-fold splits per stage. Dedicated-tree advantage has a paired interval crossing zero.",
             fontsize=9, color="#555B62")
    fig.subplots_adjust(left=.08, right=.98, top=.79, bottom=.23, wspace=.35)
    fig.savefig(figures / "generation_and_prediction.png", dpi=300)
    fig.savefig(figures / "generation_and_prediction.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    positions = np.arange(len(scenario_counts))
    for field, label, color, marker in [
        ("old_model_checks_and_chemical", "Archived selection + original proxy", "#969DA5", "o"),
        ("new_model_checks_and_chemical", "Robust chemical + new proxy checks", "#2877A6", "s"),
        ("cross_proxy_checks_and_chemical", "Robust chemical + both proxy committees", "#C68B2B", "D")]:
        ax.plot(positions, scenario_counts[field], marker=marker, color=color, linewidth=1.6, label=label)
    ax.set_xticks(positions, [f"{r.statistic}\nx{r.factor:g}" for r in scenario_counts.itertuples()])
    ax.set_ylim(-.4, 11.5)
    ax.set_yticks(range(0, 12))
    ax.set_ylabel("Sources passing the specified checks / 11")
    ax.set_title("One selected spectrum per source, checked across ten threshold scenarios", loc="left", fontsize=13, pad=17)
    ax.axvline(4.5, color="#D8DDE3", linewidth=1)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(axis="y", color="#E5E8EC", linewidth=.7)
    ax.set_axisbelow(True)
    fig.text(.09, .025, "Same 2004-candidate budget; the robust method enforces the intersection of all chemical scenarios before selection.\n"
             "Proxy checks are uncalibrated diagnostics; weaker cross-style prediction has not been resolved. No audit-based reselection.",
             fontsize=9, color="#555B62")
    fig.subplots_adjust(left=.09, right=.98, top=.85, bottom=.24)
    fig.savefig(figures / "threshold_stability.png", dpi=300)
    fig.savefig(figures / "threshold_stability.svg")
    plt.close(fig)

    attributes = ["A_malt_all", "F_malt_all", "A_hops_all", "F_hops_all", "A_esters_all",
                  "F_esters_all", "acidity", "bitternes", "sweetness"]
    values, labels = [], []
    for style in ["Kriek", "Lambic"]:
        old_part = old_profiles[old_profiles["style"].eq(style)].set_index("target").loc[attributes]
        new_part = new_profiles[new_profiles["style"].eq(style)].set_index("target").loc[attributes]
        values.extend([old_part.predicted_mean, new_part.predicted_mean, new_part.real_observed_mean])
        labels.extend([f"{style}: old", f"{style}: optimized", f"{style}: observed"])
    values = np.asarray(values)
    bound = max(np.abs(values).max(), .5)
    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xticks(np.arange(len(attributes)), ["Malt\naroma", "Malt\nflavour", "Hops\naroma", "Hops\nflavour",
                                               "Esters\naroma", "Esters\nflavour", "Acidity", "Bitterness", "Sweetness"])
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(value) > bound * .65 else "#25282D")
    ax.set_title("Generated sensory fingerprints: old and optimized", loc="left", fontsize=14, pad=15)
    fig.colorbar(im, ax=ax, shrink=.85, label="Attachment-standardized value")
    fig.text(.19, .025, "300 spectra per style. Old and optimized rows are model predictions; observed rows summarize existing panel scores.\n"
             "This descriptive fingerprint is not independent sensory validation and is not the original 0-6 scale.",
             fontsize=9, color="#555B62")
    fig.subplots_adjust(left=.19, right=.96, top=.85, bottom=.24)
    fig.savefig(figures / "sensory_fingerprint_comparison.png", dpi=300)
    fig.savefig(figures / "sensory_fingerprint_comparison.svg")
    plt.close(fig)

    generation_rows = []
    gains = {}
    for style in STYLES:
        table = gen_cv[gen_cv["style"].eq(style)].set_index("method")
        old, new, bootstrap = table.loc["legacy_copula"], table.loc["nested_selected"], table.loc["style_bootstrap"]
        gain = 1 - new.energy_pca5_mean / old.energy_pca5_mean
        gains[style] = float(gain)
        generation_rows.append(f"| {style} | {old.energy_pca5_mean:.4f} | {new.energy_pca5_mean:.4f} | "
                               f"{gain * 100:+.1f}% | {bootstrap.energy_pca5_mean:.4f} | "
                               f"{old.marginal_w1_iqr_median_mean:.4f} → {new.marginal_w1_iqr_median_mean:.4f} |")
    stale_rows = []
    for label, table in [("模型搜索阶段", search_metrics), ("固定方案后续复核", fixed_metrics)]:
        for method, name in [("old_multioutput_extra_trees", "旧45目标森林"),
                             ("fixed_single_extra_trees", "固定单目标森林")]:
            row = table.loc[method]
            stale_rows.append(f"| {label} | {name} | {row.rmse:.4f} | {row.r2:.4f} | "
                              f"{row.spearman:.4f} | {row.prediction_std_ratio:.3f} |")
    status_names = {"model_checks_passed": "新代理审计通过", "resampling_audit_failed": "新代理审计未通过",
                    "no_candidate_found": "未找到候选", "proxy_not_qualified": "代理训练内资格不足",
                    "proxy_uninformative": "代理对候选缺少响应", "source_defect_unverifiable": "源缺陷不可核验"}
    transfer_rows = []
    for row in default.itertuples():
        improvement = f"{100 * row.target_improvement:.1f}%" if pd.notna(row.candidate_index) else "—"
        transfer_rows.append(f"| {row.source_beer} | {status_names[row.status]} | {improvement} | "
                             f"{row.audit_q95:+.4f} | {'是' if row.cross_proxy_checks_passed else '否'} |")
    counts = transfer["default_policy"]
    ci = interval["rmse_difference_percentile95"]
    delta = interval["point_rmse_difference"]
    report = f"""# 第三问优化：风格保真性、缺陷预测与阈值稳健迁移

日期：2026-10-23。原始第三问65个结果文件已冻结并逐文件校验；优化结果另存 `../outputs/q3_optimized/`。

## 1. 本轮结论

- Kriek、Lambic的嵌套选择生成算法，在匹配外层划分和共同评价坐标下，能量差异指标较旧模型分别降低 **{100 * gains['Kriek']:.1f}%**、**{100 * gains['Lambic']:.1f}%**。两者仍略逊于重采样，不能称所有质量指标全面胜出。
- Stale hops专用树在后续重复切分中，RMSE由 **{fixed_metrics.loc['old_multioutput_extra_trees', 'rmse']:.4f}** 降为 **{fixed_metrics.loc['fixed_single_extra_trees', 'rmse']:.4f}**，下降 **{fixed['rmse_reduction_vs_old_multioutput'] * 100:.2f}%**。按beer配对重采样的差异区间跨零，且Pils整风格留出更差，**尚未证实稳定的预测提升**。
- 默认迁移在10个预定阈值情景下同时约束化学保护，11款源均找到候选，**{counts['model_checks_passed_all_thresholds']}**款通过新代理检查，**{counts['cross_proxy_checks_passed_all_thresholds']}**款同时通过旧冻结代理检查。旧固定谱在同一完整阈值网格下通过全部模型与化学检查的数量为 **{transfer['archived_baseline']['model_checks_passed_all_thresholds_same_spectra']}**。

模型检查仅定义本次候选筛查的通过范围，不证明实际品评结果。阈值稳健性与感官预测准确性分别报告。

## 2. 生成模型：保留风格局部边缘

旧模型采用全局经验边缘并向全局均值收缩，会削弱小样本风格的独特分布。本轮预声明三个候选：风格局部边缘copula、风格中心平滑混合、旧copula。局部模型保留目标风格边缘支持；平滑混合使用固定0.45噪声强度。内层使用新颖性门槛，避免以近似复制训练行制造误差下降。

外层为2×3折风格分层切分，种子20261123、20262123；每个训练集内2折选择方法，所有候选统一在该训练折旧copula的高斯秩空间及5维PCA参考坐标下评价。每折每法300条、2个模拟种子。内层选择不读取外层目标数据；这些新划分仍复用同一250款酒，属于重复内部验证。

| 风格 | 旧模型能量差异 | 嵌套选择能量差异 | 相对下降 | 重采样对照 | W1跨特征中位数：旧→新 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(generation_rows)}

能量指标为未开平方的经验V统计量，越小越好。平均W1、相关差异、零频率、多样性和复制检查也完整保存；Kriek平均W1仍被极端化合物支配，不能只看某一项指标。外层折与重复并不独立，折间标准差不是置信区间。

全250款重新训练后的风格选择为 `{json.dumps(generation['full_selected_methods'], ensure_ascii=False)}`。四风格各生成300条谱，原始完整行及忽略训练缺失格后的观测行复制率均为0。

![生成与预测对照](../outputs/q3_optimized/figures/generation_and_prediction.png)

## 3. Stale hops：比较失败与有限改善均保留

18个配置包括稳健线性、PLS、SVR和树模型；特征限于226个化学指标及附件aging元数据确定的子集，不使用实际其他感官评分或评论特征。处理、筛选和调参均在训练折内部。内层预测方差比至少0.05才进入合格配置，再按RMSE选择。

搜索算法本身的嵌套结果为RMSE **{search_metrics.loc['nested_selected', 'rmse']:.4f}**，劣于同划分旧森林的 **{search_metrics.loc['old_multioutput_extra_trees', 'rmse']:.4f}**。因此没有把外层表现最好的家族直接替代为“无偏最优模型”。

固定单目标森林保留原基线的300树、min_leaf=3、max_features=0.7参数，不再调参；在额外种子20263123、20264123的5折切分复核。这个后续算法选择由前一轮结果触发，因此仍是探索性复核，并非全新独立测试。

| 阶段 | 模型 | 重复OOF RMSE | R² | Spearman | 预测SD/实际SD |
|---|---|---:|---:|---:|---:|
{chr(10).join(stale_rows)}

固定方案两次切分一好一坏。先按beer平均两次切分平方误差，再对250款酒做2000次配对重采样：`新RMSE−旧RMSE` 点差 **{delta:+.5f}**，95%分位区间 **[{ci[0]:+.5f}, {ci[1]:+.5f}]**，包含零。它只描述已保存OOF误差的抽样波动，不包含模型选择过程，也没有重拟合模型。

Pils整风格留出11款时，专用树RMSE为 **{source_holdout.loc['fixed_single_extra_trees', 'rmse']:.4f}**，劣于均值 **{source_holdout.loc['non_pils_mean', 'rmse']:.4f}** 和旧代理 **{source_holdout.loc['old_multioutput_transfer_proxy', 'rmse']:.4f}**。这说明内部小幅改善未转化为跨风格预测提升。迁移代理资格门槛仅来自239款非Pils训练内验证，不等于已经通过源风格外推验证。

## 4. 同时保护10种阈值情景

定义情景集合为中位数/均值两种口径分别乘以0.8、0.9、1.0、1.1、1.2。对每个缺陷要求所有情景的超阈程度均不增加：

`c'_d ≤ min_s max(T[d,s], c_source[d]) = max(min_s T[d,s], c_source[d])`。

相应最严格阈值为DMS **13.8 μg/L**、Diacetyl **0.072 mg/L**、4VG **0.16 mg/L**。trans-2-nonenal没有附件阈值，继续采用浓度不得增加。保护依据为源的观测校准值，不以插补值替代。

迁移仍排除全部11款Pils训练数据，使用冻结的原239款训练几何和完整226维最近邻支持规则；目标距离至少改善10%、改动RMS至少0.05的条件保持不变。新旧生成器各500条、4个插值比例加4个共同目标中位谱，匹配预算均为2004候选；合并池4004候选单独报告。

239款重新训练的Tripel生成方法仍为旧copula，相同种子的新旧基础池完全一致：**{transfer['base_draws_equal']}**。因此本轮Tripel迁移改善不能归因于生成器更换，主要改变来自同时阈值约束和专用代理筛选；Kriek、Lambic的生成改善是独立结果。

专用代理有60个固定结构bootstrap模型，前40个筛选，后20个仅审计已选谱；方差和候选差异门槛用于识别近常数代理。旧30/10委员会仅对已选谱做冻结架构一致性检查，不参与重选。所有模型均排除全部11款源；不同模型仍共享239款真实训练数据，分位数不是95%真实风味保证。

| 源beer | 默认方案状态 | 目标距离改善 | 新代理审计Q95 | 同时通过旧代理 |
|---|---|---:|---:|---|
{chr(10).join(transfer_rows)}

默认有 **{counts['model_checks_passed_all_thresholds']}/11** 通过新代理检查、**{counts['cross_proxy_checks_passed_all_thresholds']}/11** 通过新旧代理共同检查，其化学保护在全部10个阈值情景中保持通过。这里的稳定性来自预先施加的情景交集约束，不能外推为任意未知阈值下都稳定。

![阈值稳定性](../outputs/q3_optimized/figures/threshold_stability.png)

## 5. 画像与结果文件

![风格画像对照](../outputs/q3_optimized/figures/sensory_fingerprint_comparison.png)

四种风格各300条谱及1200×45画像已保存。其他44维使用原全量感官预测器，Stale hops由本轮239款非Pils专用树输出；没有把它们描述为全45维新训练模型。感官尺度仍为附件标准化数值，不能当作恢复的原始0–6分。

- 生成谱与完整分布验证：`../outputs/q3_optimized/generation/`。
- 18配置嵌套搜索：`../outputs/q3_optimized/stale/`；固定专用树复核：`../outputs/q3_optimized/stale_fixed/`。
- 逐项情景保护、候选筛选、审计及三种λ结果：`../outputs/q3_optimized/transfer/`。
- 画像：`../outputs/q3_optimized/generated_sensory_profiles.csv` 和 `generated_sensory_summary.csv`。
- 实现：`11_q3_generation_optimized.py`、`12_q3_stale_optimized.py`、`12b_q3_stale_fixed.py`、`13_q3_transfer_optimized.py`。
- 一键复现：仓库根目录执行 `.venv/bin/python solution_a/run_q3_optimized.py`。原第三问对照已冻结，此入口不重写旧结果。

方法与审查详见 `q3_generation_optimized.md`、`q3_stale_optimized.md`、`q3_optimization_review.md`。数值校验见 `../outputs/q3_optimized/verification.json`。当前结果不包含真实酿造或迁移后的品评实验，Stale hops预测可靠性仍是主要瓶颈。
"""
    (SOLUTION / "docs/q3_optimization_results.md").write_text(report, encoding="utf-8", newline="\n")
    summary = {"generation_energy_reduction_vs_matched_legacy": gains,
               "fixed_stale_rmse_reduction_vs_matched_old": fixed["rmse_reduction_vs_old_multioutput"],
               "paired_rmse_difference_interval": ci,
               "stale_improvement_stably_established": False,
               "source_style_prediction_improved": fixed["pils_holdout_improves_old"],
               "default_transfer": counts, "base_draws_equal": transfer["base_draws_equal"]}
    (OUT / "results_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
