"""Build the domain-transfer report from saved prediction and audit evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from common import SOLUTION


OUT = SOLUTION / "outputs/q3_domain"
STALE = OUT / "stale"
TRANSFER = OUT / "transfer"
PREVIOUS = SOLUTION / "outputs/q3_optimized"
TABLES = OUT / "report_tables"
FIGURES = OUT / "figures"
SOURCES = []
COLORS = {"current": "#277DA8", "previous": "#8C959E",
          "historical": "#BD8733", "fresh": "#667C42", "text": "#30383F"}
CHECKS = {"new": "all_model_checks_passed", "historical": "cross_proxy_checks_passed",
          "fresh": "fresh_cross_proxy_checks_passed", "all": "all_committees_passed"}


def record_source(path, rows=None):
    SOURCES.append({"path": path.relative_to(SOLUTION).as_posix(), "rows": rows,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


def read_csv(path):
    frame = pd.read_csv(path)
    record_source(path, len(frame))
    return frame


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    record_source(path)
    return value


def boolean(series):
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    converted = series.map({True: True, False: False, 1: True, 0: False,
                            "True": True, "False": False, "true": True, "false": False})
    if (converted.isna() & series.notna()).any():
        raise ValueError(f"Unexpected boolean values: {series.unique()}")
    return converted.fillna(False).astype(bool)


def normalize_checks(frame):
    result = frame.copy()
    for column in [*CHECKS.values(), "is_default_policy", "all_thresholds_chemical_ok"]:
        if column in result:
            result[column] = boolean(result[column])
    if "all_committees_passed" not in result and all(c in result for c in
            ["cross_proxy_checks_passed", "fresh_cross_proxy_checks_passed"]):
        result["all_committees_passed"] = result.cross_proxy_checks_passed & result.fresh_cross_proxy_checks_passed
    return result


def method_name(method):
    aliases = {"mean": "Training mean", "non_pils_mean": "Training mean", "mean_global": "Training mean",
               "mean_style_balanced": "Style-balanced mean", "old_multioutput_tree": "Old 45-output tree",
               "fixed_single_tree": "Dedicated 300-tree",
               "mean_baseline": "Training mean", "old_multioutput_extra_trees": "Old 45-output tree",
               "old_multioutput": "Old 45-output tree", "fixed_single_extra_trees": "Dedicated 300-tree",
               "old_single_tree": "Dedicated 300-tree"}
    return aliases.get(str(method), str(method).replace("_", " "))


def prediction_tables(overall, by_style, pils_metrics, pils_predictions, stale_summary):
    selected = stale_summary["selected_method"]
    if selected not in set(overall.method) or selected not in set(pils_metrics.method):
        raise ValueError("Selected method is absent from prediction evidence")
    merged = overall.merge(pils_metrics[["method", "rmse", "mae", "r2"]].rename(
        columns={"rmse": "pils_rmse", "mae": "pils_mae", "r2": "pils_r2"}), on="method", validate="one_to_one")
    merged["is_selected"] = merged.method.eq(selected)
    merged.to_csv(TABLES / "prediction_method_comparison.csv", index=False, lineterminator="\n")
    selected_predictions = pils_predictions.loc[pils_predictions.method.eq(selected)].copy()
    if not selected_predictions.beer.is_unique:
        raise ValueError("Expected one saved Pils diagnostic prediction per selected-method beer")
    paired = selected_predictions[["beer", "observed", "prediction"]].rename(
        columns={"prediction": "domain_prediction"})
    baseline_ids = {"matched_single_tree": stale_summary["old_single_tree_baseline_method"],
                    "old_multioutput": stale_summary["old_multioutput_baseline_method"],
                    "training_mean": stale_summary["mean_baseline_method"]}
    for label, method in baseline_ids.items():
        part = pils_predictions.loc[pils_predictions.method.eq(method), ["beer", "observed", "prediction"]]
        paired = paired.merge(part.rename(columns={"observed": "check_observed",
                              "prediction": label + "_prediction"}), on="beer", validate="one_to_one")
        if len(paired) != len(selected_predictions) or not np.allclose(paired.observed, paired.check_observed):
            raise ValueError("Matched diagnostic predictions cover different populations or targets")
        paired = paired.drop(columns="check_observed")
    for label, field in [("matched_single_tree", "matched_single_tree_prediction"),
                         ("old_multioutput", "old_multioutput_prediction"),
                         ("training_mean", "training_mean_prediction"),
                         ("domain_selected", "domain_prediction")]:
        paired[f"{label}_residual"] = paired[field] - paired.observed
        squared = paired[f"{label}_residual"] ** 2
        paired[f"{label}_sse_share"] = squared / squared.sum()
    paired.sort_values("beer").to_csv(TABLES / "pils_source_error_influence.csv", index=False, lineterminator="\n")
    style_table = by_style.loc[by_style.method.eq(selected)].copy()
    style_table.to_csv(TABLES / "selected_loso_by_style.csv", index=False, lineterminator="\n")
    return merged, paired, selected


def transfer_tables(transfers, thresholds, previous, scenarios, experiment):
    rows = []
    for (pool, policy), group in transfers.groupby(["pool", "policy"], sort=False):
        if not group.source_beer.is_unique:
            raise ValueError("Duplicate source within transfer policy")
        selected = group.candidate_index.notna()
        rows.append({"pool": pool, "policy": policy, "sources": len(group),
            "selected": int(selected.sum()), **{f"{key}_passed": int(group[field].sum())
                                                for key, field in CHECKS.items()},
            "median_target_improvement_selected": float(group.loc[selected, "target_improvement"].median()),
            "median_change_rms_selected": float(group.loc[selected, "change_latent_rms"].median()),
            "median_target_improvement_fresh_pass": float(group.loc[group[CHECKS["fresh"]], "target_improvement"].median())})
    policies = pd.DataFrame(rows)
    policies.to_csv(TABLES / f"{experiment}_policy_comparison.csv", index=False, lineterminator="\n")
    default = transfers.loc[transfers.is_default_policy].sort_values("source_beer")
    if not default.source_beer.is_unique:
        raise ValueError("Default policy must contain one row per source")
    if set(default.source_beer) != set(previous.source_beer):
        raise ValueError("Current and rechecked historical spectra cover different sources")
    current_chemical = thresholds.groupby(["transfer_id", "scenario"]).passed.all()
    scenario_rows = []
    for scenario in scenarios:
        name = scenario["scenario"]
        mask = np.array([bool(current_chemical.get((t, name), False)) for t in default.transfer_id])
        scenario_rows.append({"scenario": name, "statistic": scenario.get("statistic"),
            "factor": scenario.get("factor"), "sources": len(default),
            "chemical_only": int(mask.sum()),
            **{f"{key}_passed": int(np.sum(mask & default[field].to_numpy(bool)))
               for key, field in CHECKS.items()}})
    scenario_table = pd.DataFrame(scenario_rows)
    scenario_table.to_csv(TABLES / f"{experiment}_default_threshold_pass_counts.csv", index=False, lineterminator="\n")
    source = default.merge(previous.rename(columns={field: "previous_" + field for field in CHECKS.values()}),
                           on="source_beer", suffixes=("", "_previous"), validate="one_to_one")
    source.to_csv(TABLES / f"{experiment}_default_source_comparison.csv", index=False, lineterminator="\n")
    return policies, default, scenario_table, source


def configure_figures():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": COLORS["text"], "text.color": COLORS["text"],
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "svg.fonttype": "none"})


def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    fig.savefig(FIGURES / f"{name}.svg")
    plt.close(fig)


def prediction_figure(comparison, selected, n_styles, n_pils):
    positions = np.arange(len(comparison))
    labels = [textwrap.fill(method_name(m), 30) + (" *" if m == selected else "") for m in comparison.method]
    colors = [COLORS["current"] if m == selected else COLORS["previous"] for m in comparison.method]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, max(7.0, len(comparison) * .45)), sharey=True)
    for axis, field, title in [(axes[0], "macro_style_rmse", f"Selection evidence: LOSO macro RMSE ({n_styles} styles)"),
                                (axes[1], "pils_rmse", f"Repeatedly viewed Pils diagnostic RMSE (n={n_pils})")]:
        values = comparison[field].to_numpy(float)
        axis.barh(positions, values, height=.6, color=colors)
        axis.set_xlim(0, float(values.max()) * 1.19)
        for y, value in zip(positions, values):
            axis.text(value + values.max() * .015, y, f"{value:.3f}", va="center", fontsize=9)
        axis.set_title(title, loc="left", fontsize=11, pad=15)
        axis.set_xlabel("Stale hops RMSE (attachment-standardized units)")
        axis.grid(axis="x", color="#E5E8EC", linewidth=.7)
        axis.set_axisbelow(True)
    axes[0].set_yticks(positions, labels, fontsize=9)
    axes[0].invert_yaxis()
    fig.text(.24, .025, "* Selected by non-Pils LOSO score, not by the Pils column.\n"
             "Candidate LOSO scores were used for selection; they are not unbiased post-selection test estimates.",
             fontsize=9, color="#525C65")
    fig.subplots_adjust(left=.24, right=.98, top=.9, bottom=.14, wspace=.23)
    save_figure(fig, "prediction_domain_validation")


def transfer_figure(experiments):
    rows, matrix_rows, matrix_labels = [], [], []
    for name, evidence in experiments.items():
        for kind, part in [("Previous spectra\nrechecked", evidence["previous"]),
                           ("Fixed default", evidence["default"])]:
            rows.append({"label": f"{evidence['label']}\n{kind}", "sources": len(part),
                         **{f"{key}_passed": int(part[field].sum()) for key, field in CHECKS.items()}})
        for field, label in [("chemical_only", "chemical"), ("fresh_passed", "fresh checks")]:
            matrix_rows.append(evidence["scenario_table"][field].to_numpy())
            matrix_labels.append(evidence["label"] + ": " + label)
    table = pd.DataFrame(rows)
    n = int(table.sources.max())
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.2), gridspec_kw={"width_ratios": [1.0, 1.45]})
    y = np.arange(len(table))
    specs = [(-.27, "new", "Primary committee checks", COLORS["current"]),
             (-.09, "historical", "+ original / historical", COLORS["historical"]),
             (.09, "fresh", "+ original / fresh", COLORS["fresh"]),
             (.27, "all", "Historical and fresh intersection", COLORS["text"])]
    for offset, key, label, color in specs:
        values = table[f"{key}_passed"]
        axes[0].barh(y + offset, values, .16, label=label, color=color)
        for yy, value, denominator in zip(y + offset, values, table.sources):
            axes[0].text(value + .08, yy, f"{value}/{denominator}", va="center", fontsize=8)
    axes[0].set_yticks(y, table.label, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, n + 1.3)
    axes[0].set_xlabel("Sources passing the stated checks")
    axes[0].set_title("Each default and its matched historical recheck", loc="left", fontsize=11, pad=14)
    axes[0].grid(axis="x", color="#E5E8EC", linewidth=.7)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -.12), ncol=2)
    matrix = np.asarray(matrix_rows)
    scenario_table = next(iter(experiments.values()))["scenario_table"]
    axes[1].imshow(matrix, vmin=0, vmax=n, cmap="Blues", aspect="auto")
    axes[1].set_yticks(np.arange(len(matrix)), matrix_labels, fontsize=9)
    scenario_labels = [f"{r.statistic}\nx{r.factor:g}" if pd.notna(r.factor) else r.scenario
                       for r in scenario_table.itertuples()]
    axes[1].set_xticks(np.arange(len(scenario_table)), scenario_labels, fontsize=8)
    for (row, column), value in np.ndenumerate(matrix):
        axes[1].text(column, row, f"{value}/{n}", ha="center", va="center", fontsize=8,
                     color="white" if value > n * .6 else COLORS["text"])
    axes[1].set_title("Fixed spectra across the saved threshold scenarios", loc="left", fontsize=11, pad=14)
    axes[1].spines[["left", "bottom"]].set_visible(False)
    axes[1].tick_params(length=0)
    fig.text(.16, .025, "Domain and follow-up use different primary committees; each historical recheck uses its own matching audit draws.\n"
             "Fresh checks do not require the reused historical 10 models. All groups still share the same observed training population.",
             fontsize=9, color="#525C65")
    fig.subplots_adjust(left=.16, right=.99, top=.89, bottom=.25, wspace=.56)
    save_figure(fig, "transfer_policy_and_thresholds")


def target_figure(experiments, archived):
    frame = archived[["source_beer", "target_improvement", "change_latent_rms"]].rename(
        columns={"target_improvement": "previous_target_improvement", "change_latent_rms": "previous_change_rms"})
    specifications = [("domain", "Domain default", experiments["domain"]["default"], COLORS["current"], "D", -.06),
                      ("consensus", "Consensus matched default", experiments["consensus"]["default"], COLORS["fresh"], "s", .07)]
    consensus = experiments["consensus"]
    expanded_pool = max(consensus["manifest"]["pools"], key=consensus["manifest"]["pools"].get)
    expanded = consensus["transfers"].loc[consensus["transfers"].pool.eq(expanded_pool)
                & consensus["transfers"].policy.eq(consensus["manifest"]["default_policy"])]
    specifications.append(("expanded", f"Consensus expanded ({consensus['manifest']['pools'][expanded_pool]})",
                           expanded, COLORS["historical"], "^", .2))
    for name, label, part, color, marker, offset in specifications:
        keep = part[["source_beer", "target_improvement", "change_latent_rms", "fresh_cross_proxy_checks_passed"]]
        frame = frame.merge(keep.rename(columns={c: f"{name}_{c}" for c in keep if c != "source_beer"}),
                            on="source_beer", validate="one_to_one")
    frame = frame.sort_values("source_beer")
    frame.to_csv(TABLES / "all_default_target_comparison.csv", index=False, lineterminator="\n")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.0), sharey=True)
    positions = np.arange(len(frame))
    for ax, field, label, scale in [(axes[0], "target_improvement", "Reduction in target-style distance (%)", 100),
                                    (axes[1], "change_latent_rms", "Chemical change (Gaussian-space RMS)", 1)]:
        previous_field = "previous_target_improvement" if field == "target_improvement" else "previous_change_rms"
        ax.scatter(frame[previous_field] * scale, positions - .2, s=35,
                   color=COLORS["previous"], marker="o", zorder=2)
        for name, _, part, color, marker, offset in specifications:
            for position, (_, row) in zip(positions, frame.iterrows()):
                value = row[f"{name}_{field}"]
                if pd.isna(value):
                    continue
                passed = bool(row[f"{name}_fresh_cross_proxy_checks_passed"])
                ax.scatter(value * scale, position + offset, s=45, marker=marker,
                           facecolor=color if passed else "white", edgecolor=color, linewidth=1.15, zorder=3)
        ax.set_xlabel(label)
        ax.grid(axis="x", color="#E5E8EC", linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_yticks(positions, [str(int(v)) for v in frame.source_beer])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Pils/Lager source beer")
    axes[0].set_xlim(0, 103)
    axes[0].axvline(10, color="#747E87", linewidth=1, linestyle="--")
    axes[0].set_title("Target-style proximity", loc="left", fontsize=12, pad=15)
    axes[1].set_xlim(left=0)
    axes[1].set_title("Amount of chemical change", loc="left", fontsize=12, pad=15)
    handles = [Line2D([], [], color=COLORS["previous"], marker="o", linestyle="None", label="Historical fixed spectra (reference)"),
               *[Line2D([], [], color=color, marker=marker, linestyle="None", label=label)
                 for _, label, _, color, marker, _ in specifications],
               Line2D([], [], color=COLORS["text"], marker="o", linestyle="None", label="Filled: respective fresh checks pass"),
               Line2D([], [], color=COLORS["text"], marker="o", markerfacecolor="white", linestyle="None", label="Open: at least one fresh check fails")]
    fig.legend(handles=handles, frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(.52, .085), ncol=3)
    fig.text(.09, .025, "Same source beers and frozen geometry. Missing markers mean no candidate; all missing cases stay in the 11-source denominator.\n"
             "Expanded search is a separate budget. Fresh audit draws differ between domain and consensus experiments.",
             fontsize=9, color="#525C65")
    fig.subplots_adjust(left=.09, right=.98, top=.9, bottom=.25, wspace=.23)
    save_figure(fig, "target_distance_and_consistency")


def write_report(comparison, influence, selected, by_style, stale_summary, stale_manifest,
                 experiments, neighborhood):
    selected_row = comparison.set_index("method").loc[selected]
    matched_tree = comparison.set_index("method").loc[stale_summary["old_single_tree_baseline_method"]]
    matched_multi = comparison.set_index("method").loc[stale_summary["old_multioutput_baseline_method"]]
    matched_mean = comparison.set_index("method").loc[stale_summary["mean_baseline_method"]]
    domain, consensus = experiments["domain"], experiments["consensus"]
    n_sources, n_styles = len(domain["default"]), by_style.heldout_style.nunique()
    n_train = int(stale_summary["training_samples"])
    selected_pils = float(selected_row.pils_rmse)
    prediction_failed = selected_pils >= min(matched_tree.pils_rmse, matched_multi.pils_rmse, matched_mean.pils_rmse)
    heading = "Pils预测未改善，联合筛查只有有限收益" if prediction_failed else "Pils诊断与联合筛查的探索性比较"
    d_default, c_default = domain["default"], consensus["default"]
    c_previous = consensus["previous"]
    expanded_pool = max(consensus["manifest"]["pools"], key=consensus["manifest"]["pools"].get)
    expanded_budget = consensus["manifest"]["pools"][expanded_pool]
    matched_budget = consensus["manifest"]["pools"][consensus["manifest"]["default_pool"]]
    expanded = consensus["transfers"].loc[consensus["transfers"].pool.eq(expanded_pool)
                & consensus["transfers"].policy.eq(consensus["manifest"]["default_policy"])]
    fresh_d = int(d_default.fresh_cross_proxy_checks_passed.sum())
    fresh_c = int(c_default.fresh_cross_proxy_checks_passed.sum())
    fresh_c_previous = int(c_previous.fresh_cross_proxy_checks_passed.sum())
    fresh_expanded = int(expanded.fresh_cross_proxy_checks_passed.sum())
    influence_row = influence.loc[influence.matched_single_tree_sse_share.idxmax()]
    original_rules = domain["summary"]["previous_original_rules"]
    lines = [f"# 第三问：{heading}", "",
        f"在 {n_train} 款非 Pils 啤酒的 {n_styles} 风格留一验证中，保存的主模型为 `{selected}`。"
        f"其宏 RMSE 为 **{selected_row.macro_style_rmse:.4f}**，较优均值基线为 "
        f"**{stale_summary['best_mean_macro_rmse']:.4f}**；但 Pils 的诊断 RMSE 为 "
        f"**{selected_pils:.4f}**，同轮旧多输出结构重训为 **{matched_multi.pils_rmse:.4f}**，"
        f"固定专用树为 **{matched_tree.pils_rmse:.4f}**。**没有实现 Pils 跨风格预测改善。**", "",
        f"局部域模型的固定默认迁移仅选出 **{d_default.candidate_index.notna().sum()}/{n_sources}** 款，"
        f"其中 **{fresh_d}/{n_sources}** 通过 fresh 一致检查。随后沿旧固定树协议做联合筛查消融："
        f"同 **{matched_budget}** 候选预算默认策略通过 **{fresh_c}/{n_sources}**，历史固定谱在"
        f"同一追加实验规则下重检通过 **{fresh_c_previous}/{n_sources}**；"
        f"预算扩为 **{expanded_budget}** 后为 **{fresh_expanded}/{n_sources}**。"
        "追加实验由前一轮失败触发，属于探索性策略比较，没有据此替换默认或证明预测精度提高。", "",
        "## 风格留出分数与Pils诊断", "",
        f"完整 LOSO 每次排除一种非 Pils 风格，在 {n_styles} 折分别拟合预处理和预测器。"
        "选择量为风格宏 RMSE 加上预声明的风格绝对偏差惩罚；风格宏平均给每种风格相同权重，"
        "与按啤酒数量加权的 pooled RMSE 不同。",
        f"本轮惩罚系数为 {stale_manifest['bias_penalty']:g}，选中方法的宏绝对偏差为 "
        f"{selected_row.macro_style_abs_bias:.4f}，选择分数为 {selected_row.selection_score:.4f}，"
        f"样本加权 LOSO RMSE 为 {selected_row.rmse:.4f}。", "",
        "LOSO 得分参与了方法选择，最低分存在选择乐观；未另设外层来检验整个选择器。"
        "Pils 标签此前已反复查看，本轮明确视为诊断集。配置选择在保存 LOSO 结果后完成，"
        "没有把 Pils 列中最低误差的配置替换为主方法，也没有按 Pils 标签校正截距。", "",
        "|方法|LOSO宏RMSE|宏绝对偏差|选择分数|Pils诊断RMSE|主方法|", 
        "|---|---:|---:|---:|---:|---|"]
    for _, row in comparison.iterrows():
        lines.append(f"|`{row['method']}`|{row.macro_style_rmse:.4f}|{row.macro_style_abs_bias:.4f}|"
                     f"{row.selection_score:.4f}|{row.pils_rmse:.4f}|{'是' if row.is_selected else ''}|")
    lines.extend(["", "![跨风格预测比较](../outputs/q3_domain/figures/prediction_domain_validation.png)", "",
        f"图1：全部 {len(comparison)} 个预声明配置的非 Pils 风格宏 RMSE 与全 {n_sources} 款 Pils 诊断 RMSE。"
        "蓝色只标记按 LOSO 规则选中的配置，未构造独立置信区间。", "",
        f"同轮固定专用树的 Pils 平方误差中，**{influence_row.matched_single_tree_sse_share:.1%}** 来自 beer "
        f"**{int(influence_row.beer)}**，其真实值为 {influence_row.observed:.4f}。"
        f"主局部模型对同一源预测 {influence_row.domain_prediction:.4f}，该源贡献主模型平方误差的 "
        f"**{influence_row.domain_selected_sse_share:.1%}**。所有主指标保留全部源，"
        "没有删除影响最大的样本或以该源单独调整方法。", "",
        f"局部模型的 bootstrap 邻域诊断中，每次名义上的 {stale_summary['config']['n_neighbors']} 个近邻"
        f"平均仅含 {neighborhood.nearest_unique_beers.mean():.2f} 款不同啤酒；"
        f"相对全量拟合邻域的 Jaccard 中位数为 {neighborhood.neighbor_jaccard_vs_final.median():.2f}。"
        "有放回抽样同时改变邻居、重复权重和拟合空间，说明训练内方差门槛不足以保证迁移配对差稳定。"
        "域模型因此保留为失败探索，不据其训练分数替代原有预测基准。", "",
        "## 两项迁移实验与检查口径", "",
        "第一项使用 LOSO 选出的局部域模型；第二项在看到其候选覆盖不足后，沿此前固定专用树协议追加联合筛查消融。"
        "追加实验重新固定默认策略和审计种子，仍然复用同一批源和训练总体，不能视为独立确认实验。", "",
        "每项实验都保存三种策略与两种预算：`single_geometry` 只用主筛选委员会并按几何目标排序；"
        "`joint_geometry` 加原筛选委员会约束，仍按几何目标排序；`joint_margin` 优先最小化两组筛选 Q95 "
        "的较差值，再按几何目标排序。默认都是保存协议中的 matched 预算和 joint_margin；"
        "扩池结果单列，未依据审计通过数重选默认。", "",
        "检查定义如下：", "",
        "- **主代理**：主筛选委员会与该实验的主审计检查通过。",
        "- **历史一致**：再要求原筛选委员会及已在先前实验中使用的原后10模型通过。",
        "- **fresh一致（主口径）**：主代理加原筛选委员会、fresh原协议20模型；**不要求历史10模型通过**。",
        "- **全部交集**：历史一致和fresh一致同时成立。", "",
        "原中心预测器是45输出模型，但原40个bootstrap模型本身是单目标120树协议。"
        "追加实验的两组新审计也沿单目标120树协议，只是种子不同，不能称为独立架构验证。"
        "所有组共享同一真实训练总体；historical与fresh采用并列定义，非逐层包含。", "",
        "|实验|预算|策略|源数|选出|主代理|历史一致|fresh一致|全部交集|选出谱改善中位数|选出谱改动RMS中位数|", 
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for name, evidence in experiments.items():
        for row in evidence["policies"].itertuples():
            lines.append(f"|{evidence['label_zh']}|{row.pool}|{row.policy}|{row.sources}|{row.selected}|"
                f"{row.new_passed}|{row.historical_passed}|{row.fresh_passed}|{row.all_passed}|"
                f"{row.median_target_improvement_selected:.1%}|{row.median_change_rms_selected:.3f}|")
    lines.extend(["", "改善与改动中位数只在各行选出谱中计算；选出数量不同，不可忽略分母直接比较。"
        "未找到候选的源计为未通过，保留在全部源分母。", "",
        f"历史报告在原规则下为 {original_rules['model_checks_passed_all_thresholds']}/{n_sources} 主代理通过、"
        f"{original_rules['cross_proxy_checks_passed_all_thresholds']}/{n_sources} 新旧一致通过。"
        "这些旧口径不含本轮主代理和fresh审计，不能和本轮数字直接计算提升。"
        "每项实验都对同一历史固定谱按该实验相同模型和规则重新计算：", "",
        "|重检所用实验|历史固定谱主代理|历史一致|fresh一致|全部交集|", "|---|---:|---:|---:|---:|"])
    for evidence in experiments.values():
        previous = evidence["previous"]
        lines.append(f"|{evidence['label_zh']}|{previous.all_model_checks_passed.sum()}/{len(previous)}|"
                     f"{previous.cross_proxy_checks_passed.sum()}/{len(previous)}|"
                     f"{previous.fresh_cross_proxy_checks_passed.sum()}/{len(previous)}|"
                     f"{previous.all_committees_passed.sum()}/{len(previous)}|")
    lines.extend(["", "![策略及阈值比较](../outputs/q3_domain/figures/transfer_policy_and_thresholds.png)", "",
        "图2：每项固定默认策略及其历史固定谱重检；右侧为默认固定谱跨全部阈值情景的化学保护及fresh一致通过数。"
        "两项实验的主代理和审计抽样不同，跨实验数字是探索性对照；同项重检才使用完全匹配规则。", "",
        "## 阈值情景、风格接近与改动代价", "",
        "化学投影同时保护保存的中位数/均值及各乘数情景；每项缺陷超阈程度均不得增加。"
        "trans-2-nonenal 无附件阈值时采用浓度不得增加。所有情景检查同一个已选谱，没有逐情景重新挑候选。"
        "下表化学保护计数也要求已选出谱，未选出的源没有可检查迁移谱，计为未通过而非伪造浓度。", "",
        "|阈值情景|域默认化学保护|域默认fresh|固定树默认化学保护|固定树默认fresh|", 
        "|---|---:|---:|---:|---:|"])
    ds, cs = domain["scenario_table"].set_index("scenario"), consensus["scenario_table"].set_index("scenario")
    for scenario in domain["scenario_table"].scenario:
        dr, cr = ds.loc[scenario], cs.loc[scenario]
        lines.append(f"|{scenario}|{dr.chemical_only}/{dr.sources}|{dr.fresh_passed}/{dr.sources}|"
                     f"{cr.chemical_only}/{cr.sources}|{cr.fresh_passed}/{cr.sources}|")
    lines.extend(["", "![目标距离与改动代价](../outputs/q3_domain/figures/target_distance_and_consistency.png)", "",
        "图3：全部源的历史固定谱、域模型默认、固定树matched默认及同策略expanded谱。左为冻结旧几何下的目标距离改善，"
        "右为同一高斯空间的改动RMS；历史灰点只作几何参考。其他实心点通过各自fresh检查，空心点失败，"
        "缺点表示无候选。expanded使用更多搜索预算，未替换matched默认。", "",
        f"追加实验 expanded joint_margin 的 {fresh_expanded}/{n_sources} fresh通过伴随目标接近度代价。"
        f"历史固定谱的目标改善中位数为 {consensus['previous'].previous_target_improvement.median():.1%}，"
        f"expanded选出谱为 {expanded.target_improvement.median():.1%}；改动RMS的对应差别与全部策略见上表。"
        "边际安全余量与几何最优并不一致，不能只用较高通过数量概括结果。", "",
        "|expanded joint_margin源|状态|目标改善|改动RMS|较差筛选Q95|主audit Q95|fresh原协议Q95|fresh通过|", 
        "|---|---|---:|---:|---:|---:|---:|---|"])
    def number(value):
        return f"{value:+.4f}" if pd.notna(value) else "—"
    for row in expanded.sort_values("source_beer").itertuples():
        improvement = f"{row.target_improvement:.1%}" if pd.notna(row.target_improvement) else "—"
        change = f"{row.change_latent_rms:.3f}" if pd.notna(row.change_latent_rms) else "—"
        lines.append(f"|{row.source_beer}|{row.status}|{improvement}|{change}|"
            f"{number(row.worst_selection_q95)}|{number(row.new_audit_q95)}|{number(row.fresh_original_audit_q95)}|"
            f"{'是' if row.fresh_cross_proxy_checks_passed else '否'}|")
    passing = expanded.loc[expanded.fresh_cross_proxy_checks_passed]
    if len(passing):
        thin = passing.loc[passing.worst_selection_q95.idxmax()]
        lines.extend(["", f"通过谱中最接近筛选边界的是源 {int(thin.source_beer)}，"
            f"较差筛选Q95为 {thin.worst_selection_q95:+.4f}。这类临界结果应保留连续数值，"
            "不把微小负值解释为具有品评置信保证。"])
    lines.extend(["", "## 范围与复现", "",
        "本轮仍未证明跨风格Stale hops准确预测改善。LOSO模型选择、候选搜索及多次追加实验复用现有数据，"
        "fresh只指新重采样模型，未带来新的独立啤酒或真实品评。固定结构bootstrap也未包含算法选择、"
        "测量误差和生成谱分布偏移；配对差Q95不是覆盖率校准的置信上界。", "",
        "所有感官值保持附件标准化量表，未恢复0–6分。当前输出是化学目标谱和模型内筛查，"
        "需要新增真实风格样本及迁移后的理化、品评实验，才能判断真实Stale hops是否不增加。", "",
        "- LOSO与Pils诊断：`../outputs/q3_domain/stale/`。",
        "- 域模型迁移失败探索：`../outputs/q3_domain/transfer/`。",
        "- 固定树联合筛查追加实验：`../outputs/q3_domain/consensus/`。",
        "- 本报告计算表：`../outputs/q3_domain/report_tables/`。",
        "- 三张图的PNG/SVG：`../outputs/q3_domain/figures/`。",
        "- 输入逐文件哈希与行数：`../outputs/q3_domain/report_source_manifest.json`。", "",
        "```bash", ".venv/bin/python solution_a/src/18_q3_domain_report.py", "```", ""])
    (SOLUTION / "docs/q3_domain_results.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    result = {"selected_method": selected, "non_pils_training_n": n_train, "loso_styles": n_styles,
        "pils_diagnostic_n": n_sources, "selected_loso_macro_rmse": float(selected_row.macro_style_rmse),
        "selected_pils_diagnostic_rmse": selected_pils, "matched_fixed_tree_pils_rmse": float(matched_tree.pils_rmse),
        "matched_old_multioutput_pils_rmse": float(matched_multi.pils_rmse), "pils_prediction_improved": not prediction_failed,
        "experiments": {name: {"default_pool": evidence["manifest"]["default_pool"],
            "default_policy": evidence["manifest"]["default_policy"],
            "default": {"sources": len(evidence["default"]), "selected": int(evidence["default"].candidate_index.notna().sum()),
                        **{key: int(evidence["default"][field].sum()) for key, field in CHECKS.items()}},
            "previous_same_spectra": {"sources": len(evidence["previous"]),
                        **{key: int(evidence["previous"][field].sum()) for key, field in CHECKS.items()}}}
            for name, evidence in experiments.items()},
        "consensus_expanded_same_policy": {"budget": expanded_budget, "sources": len(expanded),
            "selected": int(expanded.candidate_index.notna().sum()), "fresh": fresh_expanded,
            "all": int(expanded.all_committees_passed.sum()),
            "median_target_improvement_selected": float(expanded.target_improvement.median()),
            "median_change_rms_selected": float(expanded.change_latent_rms.median())},
        "threshold_scenarios": len(ds), "scientific_figure_count": 3,
        "scope": "Failed Pils prediction exploration and limited consensus-search gains, no sensory guarantee"}
    (OUT / "report_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    return result


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stale_summary = read_json(STALE / "summary.json")
    stale_manifest = read_json(STALE / "run_manifest.json")
    overall = read_csv(STALE / "loso_metrics_overall.csv")
    by_style = read_csv(STALE / "loso_metrics_by_style.csv")
    pils_metrics = read_csv(STALE / "pils_diagnostic_metrics.csv")
    pils_predictions = read_csv(STALE / "pils_diagnostic_predictions.csv")
    neighborhood = read_csv(STALE / "neighborhood_bootstrap_diagnostics.csv")
    comparison, influence, selected = prediction_tables(overall, by_style, pils_metrics, pils_predictions, stale_summary)
    experiments = {}
    for name, directory, label, label_zh in [("domain", TRANSFER, "Domain", "局部域模型"),
            ("consensus", OUT / "consensus", "Consensus follow-up", "固定树追加实验")]:
        summary = read_json(directory / "summary.json")
        manifest = read_json(directory / "run_manifest.json")
        transfers = normalize_checks(read_csv(directory / "transfer_summary.csv"))
        thresholds = read_csv(directory / "threshold_sensitivity.csv")
        thresholds["passed"] = boolean(thresholds.passed)
        previous = normalize_checks(read_csv(directory / "previous_same_spectra_recheck.csv"))
        policies, default, scenario_table, source = transfer_tables(transfers, thresholds, previous,
            manifest["threshold_scenarios"], name)
        experiments[name] = {"summary": summary, "manifest": manifest, "transfers": transfers,
            "policies": policies, "default": default, "scenario_table": scenario_table,
            "source": source, "previous": previous, "label": label, "label_zh": label_zh}
    if experiments["domain"]["scenario_table"].scenario.tolist() != experiments["consensus"]["scenario_table"].scenario.tolist():
        raise ValueError("Domain and follow-up threshold scenarios differ")
    archived = normalize_checks(read_csv(PREVIOUS / "transfer/transfer_summary.csv"))
    archived = archived.loc[archived.is_default_policy]
    configure_figures()
    prediction_figure(comparison, selected, by_style.heldout_style.nunique(), len(experiments["domain"]["default"]))
    transfer_figure(experiments)
    target_figure(experiments, archived)
    result = write_report(comparison, influence, selected, by_style, stale_summary, stale_manifest,
                          experiments, neighborhood)
    (OUT / "report_source_manifest.json").write_text(json.dumps(SOURCES, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
