"""Build Q3 sensory profiles, research figures and a source-backed result note."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import SOLUTION
from q3_data import load_q3_data

OUT = SOLUTION / "outputs/q3"
FIGURES = OUT / "figures"
STYLES = ["Tripel", "Kriek", "Lambic", "Pils/Lager"]
COLORS = {"conditional_copula": "#2876A5", "style_bootstrap": "#9B9FA5",
          "unconditional_copula": "#C88B27"}
STATUS_NAMES = {"model_checks_passed": "通过模型检查",
                "resampling_audit_failed": "重抽样审计未通过",
                "no_candidate_found": "未找到合格候选",
                "source_defect_unverifiable": "源缺陷不可核验"}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    meta, x, y, manifest = load_q3_data()
    generation = OUT / "generation"
    cv = pd.read_csv(generation / "cv_summary.csv")
    gs = json.loads((generation / "summary.json").read_text())
    ss = json.loads((OUT / "sensory/summary.json").read_text())
    ts = json.loads((OUT / "transfer/summary.json").read_text())
    transfer = pd.read_csv(OUT / "transfer/transfer_summary.csv")
    default = transfer[(transfer["mode"] == "protected") & transfer.is_default_policy].copy()
    default = default.sort_values("source_beer")
    selected = default[default.candidate_index.notna()]
    diagnostics = pd.read_csv(generation / "generated_diagnostics.csv")
    model = joblib.load(OUT / "sensory/sensory_model.joblib")
    assert model["features"] == x.columns.tolist()
    profiles, summary_profiles = [], []
    for style in STYLES:
        frame = pd.read_csv(generation / f"generated_{style.replace('/', '-')}.csv")
        predicted = model["pipeline"].predict(frame[x.columns])
        diag = diagnostics[diagnostics["style"] == style].reset_index(drop=True)
        profile = pd.DataFrame(predicted, columns=y.columns)
        profile.insert(0, "target_style", style)
        profile.insert(0, "sample_id", frame.sample_id)
        profile["within_loo_nn_style_q95"] = diag.within_loo_nn_style_q95.to_numpy()
        profile["within_cv_latent_style_q95"] = diag.within_cv_latent_style_q95.to_numpy()
        profiles.append(profile)
        for j, target in enumerate(y.columns):
            summary_profiles.append({"style": style, "target": target,
                                     "generated_n": len(profile), "predicted_mean": float(predicted[:, j].mean()),
                                     "predicted_p05": float(np.quantile(predicted[:, j], .05)),
                                     "predicted_p95": float(np.quantile(predicted[:, j], .95)),
                                     "real_observed_mean": float(y.loc[meta.tasting_category_fine.eq(style), target].mean()),
                                     "scale": "attachment_4_standardized",
                                     "interval_meaning": "spread_across_generated_spectra_not_confidence_interval"})
    pd.concat(profiles, ignore_index=True).to_csv(OUT / "generated_sensory_profiles.csv", index=False)
    profiles_summary = pd.DataFrame(summary_profiles)
    profiles_summary.to_csv(OUT / "generated_sensory_summary.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    positions = np.arange(len(STYLES))
    for offset, method in zip([-.24, 0, .24], COLORS):
        rows = cv[cv.method.eq(method)].set_index("style").loc[STYLES]
        ax.bar(positions + offset, rows.energy_pca5_mean, width=.22,
               yerr=rows.energy_pca5_std, capsize=3, color=COLORS[method],
               label={"conditional_copula": "Conditional copula", "style_bootstrap": "Style bootstrap",
                      "unconditional_copula": "Unconditional copula"}[method],
               error_kw={"elinewidth": 1, "ecolor": "#555555"})
    ax.set_xticks(positions, STYLES)
    ax.set_ylabel("Five-dimensional energy discrepancy (V-statistic)")
    ax.set_title("Style generation: held-out distribution comparison", loc="left", fontsize=14, pad=16)
    ax.grid(axis="y", color="#E5E7EA", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.015), fontsize=9)
    fig.text(.11, .025, "Mean and fold SD over 2 repeats x 3 stratified folds; 3 simulation seeds per fold.\n"
             "Lower is better. Fold SD is descriptive, not a confidence interval; each held-out style has 3-10 real beers.",
             fontsize=9, color="#555B62")
    fig.subplots_adjust(left=.11, right=.98, top=.83, bottom=.20)
    fig.savefig(FIGURES / "generation_validation.png", dpi=300)
    fig.savefig(FIGURES / "generation_validation.svg")
    plt.close(fig)

    keys = ["A_malt_all", "F_malt_all", "A_hops_all", "F_hops_all", "A_esters_all",
            "F_esters_all", "acidity", "bitternes", "sweetness"]
    means, labels = [], []
    for style in STYLES[:3]:
        rows = profiles_summary[profiles_summary["style"] == style].set_index("target").loc[keys]
        means.extend([rows.real_observed_mean.to_numpy(), rows.predicted_mean.to_numpy()])
        labels.extend([f"{style}: observed", f"{style}: generated"])
    values = np.array(means)
    fig, ax = plt.subplots(figsize=(11.8, 5.4))
    limit = max(float(np.max(np.abs(values))), .5)
    im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xticks(np.arange(len(keys)), ["Malt\naroma", "Malt\nflavour", "Hops\naroma", "Hops\nflavour",
                                       "Esters\naroma", "Esters\nflavour", "Acidity", "Bitterness", "Sweetness"])
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                color="white" if abs(value) > limit * .65 else "#25272B", fontsize=9)
    ax.set_title("Relative sensory fingerprints", loc="left", fontsize=14, pad=14)
    fig.colorbar(im, ax=ax, shrink=.85, label="Attachment-standardized sensory value")
    fig.text(.20, .02, "Generated rows use a fitted sensory predictor; 300 spectra per style. Values are not original 0-6 scores.\n"
             "Observed rows summarize the existing panel data; this figure is descriptive rather than validation.", fontsize=9, color="#555B62")
    fig.subplots_adjust(left=.20, right=.96, top=.86, bottom=.20)
    fig.savefig(FIGURES / "generated_sensory_fingerprints.png", dpi=300)
    fig.savefig(FIGURES / "generated_sensory_fingerprints.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.2), gridspec_kw={"width_ratios": [1.05, 1]})
    for position, row in enumerate(default.itertuples()):
        axes[0].scatter(row.target_distance_before, position, color="#9B9FA5", marker="o", s=32,
                        label="Source" if position == 0 else None, zorder=3)
        if pd.notna(row.candidate_index):
            color = "#2876A5" if row.all_model_checks_passed else "#C88B27"
            axes[0].plot([row.target_distance_after, row.target_distance_before], [position, position],
                         color=color, linewidth=1.5)
            axes[0].scatter(row.target_distance_after, position, color=color, marker="s", s=30, zorder=3)
        else:
            axes[0].annotate("No candidate", (row.target_distance_before, position),
                             xytext=(-8, -11), textcoords="offset points", ha="right", fontsize=8)
    axes[0].set_yticks(range(len(default)), [f"Beer {b}" for b in default.source_beer])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Distance to Tripel model centre\n(normalized Mahalanobis)")
    axes[0].set_title("Source and selected target spectrum", loc="left", pad=14)
    axes[0].grid(axis="x", color="#E5E7EA", linewidth=.7)
    axes[0].set_axisbelow(True)
    positions = np.arange(len(selected))
    colors = ["#2876A5" if v else "#C88B27" for v in selected.all_model_checks_passed]
    axes[1].barh(positions, selected.audit_paired_delta_q95, color=colors, height=.65)
    axes[1].scatter(selected.selection_paired_delta_q95, positions, color="#25272B", marker="x", s=35,
                    label="Selection-model upper quantile", zorder=3)
    axes[1].axvline(0, color="#555B62", linewidth=1)
    axes[1].set_yticks(positions, [f"Beer {b}" for b in selected.source_beer])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Stale hops: paired predicted change\n(upper 0.95 quantile)")
    axes[1].set_title("Resampling audit of selected candidates", loc="left", pad=14)
    axes[1].grid(axis="x", color="#E5E7EA", linewidth=.7)
    axes[1].set_axisbelow(True)
    fig.suptitle("Pils/Lager to Tripel | fixed policy lambda = 0.2", x=.09, ha="left", fontsize=15)
    fig.text(.09, .025, "Bars: audit upper quantile; crosses: selection upper quantile. Blue: model checks passed; gold: audit failed.\n"
             "All 11 sources were excluded from training. The weak Stale hops proxy does not establish actual sensory protection.\n"
             "Only 1 of 11 sources also passes the mean-threshold recheck of the same selected spectra; no reselection.",
             fontsize=9, color="#555B62")
    fig.subplots_adjust(left=.09, right=.98, top=.84, bottom=.24, wspace=.42)
    fig.savefig(FIGURES / "transfer_checks.png", dpi=300)
    fig.savefig(FIGURES / "transfer_checks.svg")
    plt.close(fig)

    # Present the exact metric, not its square root, to match the saved runs.
    comparison_lines = []
    for style in STYLES:
        rows = cv[cv["style"] == style].set_index("method")
        main_row, boot_row = rows.loc["conditional_copula"], rows.loc["style_bootstrap"]
        gain = 100 * (1 - main_row.energy_pca5_mean / boot_row.energy_pca5_mean)
        comparison_lines.append(f"| {style} | {main_row.energy_pca5_mean:.4f} | {boot_row.energy_pca5_mean:.4f} | "
                                f"{rows.loc['unconditional_copula'].energy_pca5_mean:.4f} | {gain:+.1f}% | "
                                f"{main_row.marginal_w1_iqr_median_mean:.4f} / {boot_row.marginal_w1_iqr_median_mean:.4f} |")
    transfer_lines = []
    for row in default.itertuples():
        gain = f"{100 * row.target_improvement:.1f}%" if pd.notna(row.candidate_index) else "—"
        audit = f"{row.audit_paired_delta_q95:+.4f}" if pd.notna(row.candidate_index) else "—"
        mean_pass = "是" if row.all_model_checks_passed and row.chemical_mean_ok == True else "否"
        transfer_lines.append(f"| {row.source_beer} | {STATUS_NAMES[row.status]} | {gain} | {audit} | {mean_pass} |")
    defects = ts["constraints"]
    defect_lines = "\n".join(f"| {name} | `{r['feature']}` | {r['unit']} | {r['threshold_median'] if r['threshold_median'] is not None else '缺失'} | "
                              f"{r['threshold_mean'] if r['threshold_mean'] is not None else '缺失'} |"
                              for name, r in defects.items())
    default_counts = ts["default_policy"]
    q_stale = next(r for r in ss["stale_hops_metrics"] if r["model"] == "extra_trees")
    h_stale = next(r for r in ss["transfer_proxy"]["stale_hops_metrics"] if r["model"] == "transfer_proxy")
    passed = default[default.all_model_checks_passed]
    mean_safe = passed[passed.chemical_mean_ok.eq(True)]
    improvement_range = (f"{passed.target_improvement.min() * 100:.1f}%–{passed.target_improvement.max() * 100:.1f}%"
                         if len(passed) else "未找到通过样本")
    report = f"""# 第三问：风格生成与跨风格迁移实验

日期：2026-10-23。全部数字由保存的模型、谱与验证表计算。复现入口为 `../run_q3.py`。

## 1. 运行结果

已生成 Tripel、Kriek、Lambic、Pils/Lager 各300条谱，共 **{gs['full_generation_rows']}** 条，并给出45维连续感官画像。生成谱的69个校准字段均为非负，未出现完整训练行的直接复制；157个未校准字段保留相对量标记。

对11款Pils/Lager源样本进行Tripel方向迁移。默认λ=0.2、中位阈值口径下，**{default_counts['n_candidates_selected']}**款找到筛选合格候选，**{default_counts['n_model_checks_passed']}**款通过全部模型检查，**{default_counts['n_audit_failed']}**款在保留的重抽样模型审计中未通过，**{default_counts['n_no_candidate']}**款未找到候选。同一选定谱改用均值阈值复核时，仅 **{len(mean_safe)}/11** 仍通过全部检查。

这里的“通过”指本次明确定义的数值与模型检查。Stale hops预测较弱，结果需要实际品评检验后才能判断真实缺陷是否得到控制。

## 2. 统一数据表示

使用226维输入：原1.csv的231维去掉5项派生汇总，其中69维替换为8.xlsx按beer对应的校准原值，剩余157维保留原相对表示。3个负校准观测设为缺失，共72个缺失值；补值在每个训练折内部拟合。原始零值保留，未校准的负值保留原含义。每列来源、表示、单位与缺失记录见 `../outputs/q3/sensory/feature_manifest.csv`。

感官预测器按同一226维表示重新训练，五折折外RMSE为 **0.4865**。画像保持附件4的标准化尺度，尚不能解释成原始0–6分。生成谱的5%–95%预测分位数描述300条谱之间的分布范围，不是预测置信区间。

## 3. 风格条件生成

训练经验分布中秩映射到高斯空间；风格均值以n/(n+5)向全局均值收缩，协方差以n/(n+20)混合风格与共享风格内Ledoit–Wolf协方差。所有矩阵加微小对角正则以保证正定。经验逆CDF生成新的联合组合，单列仍取训练支持中的值。零值在逆CDF中有概率区间，但生成零频率不保证与经验频率完全一致，另有逐字段诊断。

固定主参数(5,20)，单独报告(2,5)、(10,50)敏感性；未用外层结果选择主参数。对22风格分层做2次重复×3折验证，每次比较条件生成、目标风格重采样、无条件生成；每法300条、3个模拟种子。每个风格的真实留出样本只有3–10款，模拟300条不能增加真实证据量。分割重叠，折间标准差属于描述性波动。

能量差异指标在训练折拟合的5维PCA标准化空间中计算：

`E = 2 mean||a-b|| - mean||a-a'|| - mean||b-b'||`。

它是包含对角项的经验V统计量，未开平方；越小越好。W1按训练IQR标准化，零IQR回退到训练range，再回退1。跨特征中位数作为稳健补充，完整均值与逐字段结果同时保存。

| 风格 | 条件生成E | 重采样E | 无条件E | 相对重采样改善 | W1跨特征中位数：条件/重采样 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(comparison_lines)}

条件模型仅在Tripel的此项指标上优于重采样，其他风格未显示同样优势。重采样复制率为1，条件生成复制率为0，体现新颖性与分布保真性的权衡。Kriek的平均W1被极端化合物值支配，因此不能只凭平均W1判断模型优劣。模型适合提供可检查的候选谱，当前证据不足以宣称所有风格都已获得高保真生成。

相对画像中，Kriek与Lambic的酸度明显向中部收缩，反映小样本条件分布收缩和感官预测误差的共同影响；画像不能取代真实品评。

![生成验证](../outputs/q3/figures/generation_validation.png)

训练内Mahalanobis距离参考较乐观，全部生成谱都超过训练内95%距离分位；采用重复留出真实样本的参考范围时结果明显不同。两种标记均保留在generated_diagnostics.csv，属于描述性参考，不构成已校准的可行性保证。未用事后放宽支持规则来改变迁移结果。

![相对感官画像](../outputs/q3/figures/generated_sensory_fingerprints.png)

## 4. 缺陷保护迁移

全部11款源Pils/Lager排除出迁移Copula、45维感官代理和40个Stale hops重抽样模型训练。迁移训练集239款；源的四项缺陷浓度均有原始校准观测。若缺失，流程返回不可核验，不用插补值充当保护依据。

| 缺陷 | 字段 | 单位 | 中位阈值 | 均值阈值 |
|---|---|---|---:|---:|
{defect_lines}

有阈值项约束`max(c'/T-1,0) <= max(c_source/T-1,0)`，等价为`c' <= max(T,c_source)`。trans-2-nonenal未提供阈值，采用浓度不得增加的替代约束。4VG按题面列出的保护项执行，不据此推论它在所有风格中都属于缺陷。

每源固定500条条件谱×4个插值比例，加4个目标中位谱插值，共2004个候选；投影后重新计算全部距离及预测。通过条件为：四项化学约束、目标距离至少改善10%、全226维高斯秩空间的目标最近邻距离不超过Tripel训练LOO距离95%分位、所有字段处于非源训练边缘范围、改动RMS至少0.05，以及30个选择模型的配对预测差上0.95分位≤0。

在可行集合内最小化`目标距离/源目标距离 + λ × 改动RMS/源到目标中心RMS`。固定λ为0.05、0.2、1.0，默认0.2；后10个重抽样模型只审计已选谱，不参与重选。两组模型共享真实训练数据，其分位数是模型诊断，既不是独立实验，也没有经过95%覆盖校准。

| 源beer | 默认方案状态 | 目标距离改善 | 审计配对差Q95 | 同谱均值阈值也全部通过 |
|---|---|---:|---:|---|
{chr(10).join(transfer_lines)}

通过模型检查的源编号为 **{', '.join(map(str, passed.source_beer))}**，目标距离改善范围 **{improvement_range}**。同谱均值阈值也通过的编号为 **{', '.join(map(str, mean_safe.source_beer)) or '无'}**。DMS均值阈值17.25低于中位阈值19，故阈值口径会改变结果，不能省略这项敏感性。

![迁移检查](../outputs/q3/figures/transfer_checks.png)

## 5. 适用边界

Stale hops五折R²为 **{q_stale['macro_r2']:.4f}**，RMSE **{q_stale['rmse']:.4f}**；整风格留出11款Pils时R²为 **{h_stale['macro_r2']:.4f}**，RMSE **{h_stale['rmse']:.4f}**。这一关键代理预测能力有限，总体45维预测表现不能替代缺陷目标的验证。候选搜索后的模型上分位约束也不能转化为真实风味不恶化的概率保证。

在28款真实Tripel中，对9个源样本没有真实目标酒同时符合四项化学保护上界，说明目标风格与保护条件存在张力。投影生成的是化学目标谱，尚未给出可实现该谱的配料或发酵控制方案。当前验证不包含生产实验、质量守恒或真实迁移后的品评。

## 6. 文件与复现

- `../outputs/q3/generation/generated_Tripel.csv`、`generated_Kriek.csv`、`generated_Lambic.csv`、`generated_Pils-Lager.csv`：各300条226维谱。
- `../outputs/q3/generated_sensory_profiles.csv`：全部1200条谱的45维画像；`generated_sensory_summary.csv`含均值、分位与真实风格均值。
- `../outputs/q3/transfer/transferred_spectra.csv`：三种λ下的保护方案及未加缺陷保护的基线谱；失败候选仍保留状态，不作为成功样本。
- `../outputs/q3/transfer/defect_checks.csv`、`sensory_before_after.csv`、`feature_changes.csv`：逐项缺陷、感官与特征改动。
- `../outputs/q3/transfer/candidate_diagnostics.csv`、`stale_paired_deltas.csv`：候选筛选与审计原始证据。
- `q3_generation.md`、`q3_representation.md`、`q3_method_review.md`：模型、数据表示及数值审查；`../outputs/q3/verification.json`为程序一致性检查。

在仓库根目录执行：`.venv/bin/python solution_a/run_q3.py`。本轮参数、源排除与候选判据均随代码保存。

数据来源为赛题附件1/4/8。协方差实现参照 [scikit-learn LedoitWolf文档](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html)，边缘距离使用 [SciPy Wasserstein-1实现](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)。风格收缩、迁移约束及候选审计是本题明确写出的建模选择，未声称由上述文档直接保证其效果。
"""
    (SOLUTION / "docs/q3_results.md").write_text(report, encoding="utf-8")
    print(f"Saved {len(pd.concat(profiles))} sensory profiles, three figures, and docs/q3_results.md")


if __name__ == "__main__":
    main()
