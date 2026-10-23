"""Run Q3 generation using fixed hyperparameters and repeated held-out validation."""
import os

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import rankdata, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from common import SEED, SOLUTION
from q3_copula import ConditionalGaussianCopula
from q3_data import load_q3_data


OUT = SOLUTION / "outputs" / "q3" / "generation"
STYLES = ["Tripel", "Kriek", "Lambic", "Pils/Lager"]
MAIN_PARAMS = (5.0, 20.0)
SENSITIVITY_PARAMS = [(2.0, 5.0), MAIN_PARAMS, (10.0, 50.0)]
REPEAT_SEEDS = [SEED, SEED + 1000]
N_GENERATED = 300
MC_REPEATS = 3
METRICS = ["marginal_w1_iqr", "marginal_w1_iqr_median", "energy_pca5", "spearman_mae",
           "generated_nn_style_median", "holdout_nn_style_median", "nn_ratio",
           "copy_rate", "unique_fraction", "variance_ratio_median",
           "variance_coverage_0p5_2", "style_training_range_fraction"]


def safe_style(style):
    return style.replace("/", "-")


def training_scale(values):
    iqr = np.subtract(*np.percentile(values, [75, 25], axis=0))
    span = np.ptp(values, axis=0)
    scale = np.where(iqr > 1e-12, iqr, np.where(span > 1e-12, span, 1.0))
    return scale, np.where(iqr > 1e-12, "iqr", np.where(span > 1e-12, "range", "one"))


def pca_coordinates(model, x_train):
    z = model.transform(x_train)
    pca = PCA(n_components=min(5, len(z) - 1, z.shape[1]), svd_solver="full").fit(z)
    # Each component is measured in its training standard deviation.
    scale = np.maximum(np.sqrt(pca.explained_variance_), 1e-8)
    return pca, scale


def energy_distance(a, b):
    # Empirical V-statistic; biased upward at tiny held-out n, but same real
    # rows and same generated n are used for every comparison.
    return float(2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean())


def rank_correlation(values):
    ranks = rankdata(values, axis=0)
    ranks -= ranks.mean(axis=0)
    norms = np.sqrt(np.sum(ranks ** 2, axis=0))
    standardized = np.divide(ranks, norms, out=np.zeros_like(ranks), where=norms > 0)
    return standardized.T @ standardized


def evaluate(model, x_train, style_train, heldout, generated, pca, pc_scale):
    training = model.impute(x_train)
    reference = model.impute(style_train)
    real = model.impute(heldout)
    fake = model.impute(generated)
    scale, _ = training_scale(training)
    marginal = np.array([wasserstein_distance(real[:, j], fake[:, j]) / scale[j]
                         for j in range(real.shape[1])])
    real_z, fake_z = model.transform(heldout), model.transform(generated)
    reference_z = model.transform(style_train)
    real_pc = pca.transform(real_z) / pc_scale
    fake_pc = pca.transform(fake_z) / pc_scale
    nn_generated = cdist(fake_z, reference_z).min(axis=1) / np.sqrt(real.shape[1])
    nn_real = cdist(real_z, reference_z).min(axis=1) / np.sqrt(real.shape[1])
    valid_corr = (np.ptp(real, axis=0) > 1e-12) & (np.ptp(reference, axis=0) > 1e-12)
    if valid_corr.sum() > 1:
        real_corr = rank_correlation(real[:, valid_corr])
        fake_corr = rank_correlation(fake[:, valid_corr])
        upper = np.triu_indices(len(real_corr), k=1)
        corr_error = float(np.mean(np.abs(real_corr[upper] - fake_corr[upper])))
    else:
        corr_error = float("nan")
    reference_var = np.var(reference, axis=0, ddof=1)
    fake_var = np.var(fake, axis=0, ddof=1)
    variable = reference_var > 1e-20
    ratios = fake_var[variable] / reference_var[variable]
    return {
        "marginal_w1_iqr": float(marginal.mean()),
        "marginal_w1_iqr_median": float(np.median(marginal)),
        "energy_pca5": energy_distance(real_pc, fake_pc),
        "spearman_mae": corr_error,
        "correlation_features": int(valid_corr.sum()),
        "generated_nn_style_median": float(np.median(nn_generated)),
        "holdout_nn_style_median": float(np.median(nn_real)),
        "nn_ratio": float(np.median(nn_generated) / max(np.median(nn_real), 1e-12)),
        "copy_rate": float(np.mean(nn_generated < 1e-12)),
        "unique_fraction": float(len(np.unique(fake, axis=0)) / len(fake)),
        "variance_ratio_median": float(np.median(ratios)),
        "variance_coverage_0p5_2": float(np.mean((ratios >= 0.5) & (ratios <= 2))),
        "variance_reference_features": int(variable.sum()),
        "style_training_range_fraction": float(np.mean(
            (fake >= reference.min(axis=0)) & (fake <= reference.max(axis=0)))),
    }, marginal


def summarize_validation(records):
    # First average Monte Carlo replicates so one data split is one unit in
    # the descriptive fold SD. Repeated folds are dependent: this is not a CI.
    fold = records.groupby(["repeat", "fold", "style", "method"], as_index=False)[METRICS].mean()
    summary = fold.groupby(["style", "method"])[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return fold, summary.reset_index()


def run_validation(x, styles):
    records, sensitivity, assignments, marginal_records, support_records = [], [], [], [], []
    for repeat, split_seed in enumerate(REPEAT_SEEDS, start=1):
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=split_seed)
        for fold, (train, test) in enumerate(splitter.split(x, styles), start=1):
            print(f"Generation validation repeat={repeat}, fold={fold}", flush=True)
            x_train = x.iloc[train]
            train_styles = styles.iloc[train]
            x_test = x.iloc[test]
            test_styles = styles.iloc[test]
            assignments.extend({"repeat": repeat, "fold": fold, "row_index": int(i),
                                "role": "test", "style": str(styles.iloc[i])} for i in test)
            for mean_strength, cov_strength in SENSITIVITY_PARAMS:
                model = ConditionalGaussianCopula(mean_strength, cov_strength).fit(x_train, train_styles)
                pca, pc_scale = pca_coordinates(model, x_train)
                main = (mean_strength, cov_strength) == MAIN_PARAMS
                for style_index, style in enumerate(STYLES):
                    style_train = x_train.loc[train_styles == style]
                    heldout = x_test.loc[test_styles == style]
                    if len(heldout) == 0:
                        raise AssertionError(f"Missing evaluation style {style}")
                    if main:
                        heldout_latent = model.transform(heldout)
                        train_latent = model.transform(style_train)
                        near = cdist(heldout_latent, train_latent).min(axis=1) / np.sqrt(x.shape[1])
                        latent_distance = model.distance(heldout, style)
                        support_records.extend({"repeat": repeat, "fold": fold,
                            "row_index": int(i), "style": style,
                            "latent_distance": float(distance), "nn_train_style": float(nn)}
                            for i, distance, nn in zip(heldout.index, latent_distance, near))
                    mc_count = MC_REPEATS if main else 1
                    for mc in range(mc_count):
                        sample_seed = split_seed + fold * 10000 + style_index * 100 + mc
                        draws = {"conditional_copula": model.sample(style, N_GENERATED, sample_seed)}
                        if main:
                            indices = np.random.default_rng(sample_seed).integers(0, len(style_train), N_GENERATED)
                            draws["style_bootstrap"] = pd.DataFrame(
                                model.impute(style_train)[indices], columns=x.columns)
                            draws["unconditional_copula"] = model.sample(None, N_GENERATED, sample_seed)
                        for method, generated in draws.items():
                            metrics, marginal = evaluate(model, x_train, style_train, heldout,
                                                         generated, pca, pc_scale)
                            record = {"repeat": repeat, "fold": fold, "mc": mc,
                                      "split_seed": split_seed, "sample_seed": sample_seed,
                                      "style": style, "method": method,
                                      "train_n": len(x_train), "style_train_n": len(style_train),
                                      "holdout_style_n": len(heldout), "generated_n": len(generated),
                                      "mean_strength": mean_strength, "cov_strength": cov_strength,
                                      **metrics}
                            if main:
                                records.append(record)
                                if mc == 0:
                                    marginal_records.extend({"repeat": repeat, "fold": fold,
                                        "style": style, "method": method, "feature": feature,
                                        "w1_iqr": float(value)}
                                        for feature, value in zip(x.columns, marginal))
                            else:
                                sensitivity.append(record)
    return (pd.DataFrame(records), pd.DataFrame(sensitivity),
            pd.DataFrame(assignments), pd.DataFrame(marginal_records), pd.DataFrame(support_records))


def support_diagnostics(model, x, labels):
    z = model.transform(x)
    rows, thresholds = [], []
    for style in sorted(labels.unique()):
        indices = np.flatnonzero(np.asarray(labels) == style)
        style_x = x.iloc[indices]
        distances = model.distance(style_x, style)
        pairwise = cdist(z[indices], z[indices]) / np.sqrt(z.shape[1])
        np.fill_diagonal(pairwise, np.inf)
        nn = pairwise.min(axis=1)
        rows.extend({"row_index": int(i), "style": style,
                     "latent_distance": float(distance),
                     "leave_one_out_nn_style": float(near)}
                    for i, distance, near in zip(indices, distances, nn))
        thresholds.append({"style": style, "n": len(indices),
            "latent_distance_q50": float(np.quantile(distances, 0.5)),
            "latent_distance_q95": float(np.quantile(distances, 0.95)),
            "latent_distance_max": float(distances.max()),
            "loo_nn_style_q50": float(np.quantile(nn, 0.5)),
            "loo_nn_style_q95": float(np.quantile(nn, 0.95)),
            "loo_nn_style_max": float(nn.max()),
            "calibration": "full_fit_in_sample_style_distance_and_loo_nn_descriptive"})
    return pd.DataFrame(rows), pd.DataFrame(thresholds)


def generated_diagnostics(model, generated, style, x, labels, support, cv_support):
    fake_z, all_z = model.transform(generated), model.transform(x)
    target_z = all_z[np.asarray(labels) == style]
    all_nn = cdist(fake_z, all_z).min(axis=1) / np.sqrt(fake_z.shape[1])
    target_nn = cdist(fake_z, target_z).min(axis=1) / np.sqrt(fake_z.shape[1])
    distances = model.distance(generated, style)
    thresholds = support.set_index("style").loc[style]
    cv_thresholds = cv_support.set_index("style").loc[style]
    return pd.DataFrame({"sample_id": np.arange(1, len(generated) + 1), "style": style,
        "latent_distance": distances, "nn_all": all_nn, "nn_target_style": target_nn,
        "exact_training_copy": all_nn < 1e-12,
        "within_latent_style_q95": distances <= thresholds.latent_distance_q95,
        "within_loo_nn_style_q95": target_nn <= thresholds.loo_nn_style_q95,
        "within_cv_latent_style_q95": distances <= cv_thresholds.latent_distance_q95,
        "within_cv_nn_style_q95": target_nn <= cv_thresholds.nn_train_style_q95})


def write_report(summary, sensitivity, full_diagnostics, support, elapsed, x, labels):
    lines = [
        "# 第三问：风格条件生成运行报告", "",
        "运行日期：2026-10-23。", "",
        "## 数据、模型与固定参数", "",
        f"使用 {len(x)} 款啤酒、{x.shape[1]} 个混合表示特征及 {labels.nunique()} 个风格。"
        "其中 69 个字段使用附件校准值，157 个使用原文件相对表示；去除 5 个派生汇总列。"
        "负校准值视为缺失，保留真实零，按训练折中位数补缺。原始附件不作修改。", "",
        "逐列用训练经验分布中秩映射到正态潜在空间。风格均值以 n/(n+5) 向全局均值收缩；"
        "协方差以 n/(n+20) 混合风格 Ledoit–Wolf 协方差与全体训练样本去风格均值后的 "
        "Ledoit–Wolf 共享协方差，另加微小对角项以保证正定。固定参数 (5,20) 在查看验证结果前声明，"
        "不以外层验证选择参数。用 (2,5) 与 (10,50) 单独做敏感性比较。", "",
        "生成逆映射采用离散经验逆分布，单列取值来自补缺后的训练支持；零及检出限在逆分布中拥有"
        "非零宽度的概率区间。条件潜在分布及高斯近似会改变实际生成的零频率，"
        "不保证它精确等于全局或目标风格的经验零频率；差异另存 `generated_zero_diagnostics.csv`。"
        "新颖性来自不同列的新组合。连续潜在高斯模型仍只是对点质量依赖关系的近似。"
        "该方法输出化学目标谱，未包含配料、温度、时间或发酵控制，因此不构成可执行酿造配方。", "",
        "## 无泄漏验证与结果", "",
        "对全部 22 个风格做 2 次独立随机种子的分层 3 折划分，逐折重拟合补缺、经验分布、"
        "风格中心和协方差；评价 Tripel、Kriek、Lambic、Pils/Lager。每法每折 300 个生成样本，"
        "3 个 Monte Carlo 种子。比较条件 copula、目标风格训练样本 bootstrap、无条件 copula。"
        "下表先平均同折 Monte Carlo 次数，再对 6 个划分取均值；折间标准差见 CSV，"
        "由于折及重复间数据有重叠，该标准差不是独立实验置信区间。", "",
        "|风格|方法|平均边缘 W1↓|边缘W1中位数↓|五维能量距离↓|Spearman差↓|复制率|方差覆盖率|", "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    method_names = {"conditional_copula": "条件copula", "style_bootstrap": "风格bootstrap",
                    "unconditional_copula": "无条件copula"}
    for _, row in summary.iterrows():
        lines.append(f"|{row['style']}|{method_names[row['method']]}|"
            f"{row['marginal_w1_iqr_mean']:.4f}|{row['marginal_w1_iqr_median_mean']:.4f}|{row['energy_pca5_mean']:.4f}|"
            f"{row['spearman_mae_mean']:.4f}|{row['copy_rate_mean']:.1%}|"
            f"{row['variance_coverage_0p5_2_mean']:.1%}|")
    lines.extend(["", "边缘 W1 以全部训练行各列 IQR 标准化；零 IQR 依次回退训练极差、常数 1。"
        "均值保留了所有极端值的影响，另报跨特征中位数。Kriek 的均值尤其受相对特征 "
        "`X2.methyltetrahydrofuran.3.one` 的极端值支配，因此不能据平均 W1 的单项改善宣称全面优越。"
        "五维能量距离在训练 Gaussian 空间拟合的 PCA 中计算，各轴以训练标准差缩放，"
        "使用经验 V 统计量。Spearman 差为测试及风格训练中均非恒定的列之间相关系数的平均绝对差，"
        "生成恒定列的相关记 0。方差覆盖率为生成方差/目标风格训练方差落在 [0.5,2] 的特征比例，"
        "排除目标训练中恒定列；它衡量风格训练方差再现程度，不代表包含真实总体方差的置信覆盖。", ""])
    for style in STYLES:
        table = summary.loc[summary["style"] == style].set_index("method")
        cond = table.loc["conditional_copula"]
        boot = table.loc["style_bootstrap"]
        uncond = table.loc["unconditional_copula"]
        w1change = (cond.marginal_w1_iqr_mean / boot.marginal_w1_iqr_mean - 1) * 100
        energychange = (cond.energy_pca5_mean / boot.energy_pca5_mean - 1) * 100
        lines.append(f"- {style}：条件模型相对风格 bootstrap 的边缘 W1 变化 {w1change:+.1f}%，"
                     f"能量距离变化 {energychange:+.1f}%；相对无条件模型能量距离变化 "
                     f"{(cond.energy_pca5_mean / uncond.energy_pca5_mean - 1) * 100:+.1f}%。")
    lines.extend(["", "bootstrap 直接复制风格训练行，复制率 100%，作为高保真但无新颖性的基线；"
        "不得将条件模型在部分指标劣于 bootstrap 的结果删除。300 是每次模拟样本数，"
        "并没有将 Kriek 的 10 款真实样本扩充成 300 个独立证据；Kriek/Lambic/Pils-Lager "
        "每折测试仅约 3–4 款，Tripel 每折约 9–10 款，边缘尾部、相关性与能量距离估计均很不稳定。"
        "尤其相关矩阵有 226 维，其留出比较只作探索性诊断。需要新增真实啤酒及实测品评验证。", "",
        "## 全量拟合生成及支持域", "",
        "在全部数据重新拟合后，为 Tripel、Kriek、Lambic、Pils/Lager 各保存 300 行生成谱；"
        "输出不是折外性能估计。每行另存相对风格中心的正则 Mahalanobis 距离、"
        "到所有/目标风格训练啤酒的 Gaussian 均方根近邻距离、完全复制标记。", "",
        "|风格|完全复制率|距离处于全量训练95%参考内|近邻处于LOO95%参考内|距离处于CV留出95%参考内|", "|---|---:|---:|---:|---:|"])
    for style, rows in full_diagnostics.groupby("style"):
        lines.append(f"|{style}|{rows.exact_training_copy.mean():.1%}|"
                     f"{rows.within_latent_style_q95.mean():.1%}|{rows.within_loo_nn_style_q95.mean():.1%}|"
                     f"{rows.within_cv_latent_style_q95.mean():.1%}|")
    lines.extend(["", "支持阈值由全量风格样本的拟合距离 95% 分位数及风格内留一近邻距离 95% 分位数构成，"
        "用于跨风格候选的描述性筛选。它们不是覆盖率保证或独立校准阈值；中心和距离使用全量拟合，"
        "小样本阈值有明显不确定性，应同时报告候选的连续距离与是否越界。"
        "本轮全部生成谱都超过全量训练距离95%阈值，反映高维小样本模型的训练内距离明显乐观。"
        "另存折外真实样本距离和 `cv_support_thresholds.csv`，用作较宽的对照参考；"
        "它来自不同折模型，将其用于全量模型候选也没有严格覆盖率保证。"
        "所有阈值版本均保留，避免在看到通过率后隐藏不利诊断。", "",
        "## 参数敏感性与复现", "",
        "敏感性仅用每折相同的第一个 Monte Carlo 种子，比对已预先声明的三对固定参数。"
        "不得把敏感性最好的外层结果重新报告为无偏的调参后精度。", "",
        "|均值收缩强度|协方差收缩强度|平均边缘 W1|平均能量距离|", "|---:|---:|---:|---:|"])
    for _, row in sensitivity.iterrows():
        lines.append(f"|{row.mean_strength:g}|{row.cov_strength:g}|{row.marginal_w1_iqr:.4f}|{row.energy_pca5:.4f}|")
    lines.extend(["", f"本次运行耗时约 {elapsed:.1f} 秒，线性代数线程数固定为 1。",
        "", "```bash", ".venv/bin/python solution_a/src/07_q3_generation.py", "```", "",
        "模型类接口：`ConditionalGaussianCopula(5,20).fit(X, styles)`；"
        "`sample(style,n,seed)` 返回同列顺序 DataFrame；`transform(X)` 返回 Gaussian 潜在矩阵；"
        "`inverse_transform(Z)` 返回离散经验逆映射；`distance(X,style)` 返回按特征数归一的正则 "
        "Mahalanobis 距离。全量模型保存为 `outputs/q3/generation/fitted_model.joblib`。", "",
        "主要输出：`cv_metrics.csv`、`cv_summary.csv`、`cv_fold_means.csv`、`cv_mc_variability.csv`、"
        "`cv_assignments.csv`、`cv_marginal_w1_by_feature.csv`、`sensitivity_metrics.csv`、"
        "`sensitivity_summary.csv`、`generated_*.csv`、`generated_diagnostics.csv`、"
        "`full_style_support_metrics.csv`、`full_style_support_thresholds.csv`、"
        "`cv_holdout_support_metrics.csv`、`cv_support_thresholds.csv`、`generated_zero_diagnostics.csv` 与 `summary.json`。", ""])
    mc_variability = pd.read_csv(OUT / "cv_mc_variability.csv")
    conditional_mc = mc_variability.loc[mc_variability.method == "conditional_copula", "energy_pca5_std"]
    lines.extend(["条件生成能量距离在固定划分内的 3 次 Monte Carlo 标准差："
                  f"中位数 {conditional_mc.median():.4f}，最大值 {conditional_mc.max():.4f}。"
                  "它仅反映有限生成次数的随机波动，不覆盖真实风格样本少带来的估计误差。", ""])
    (SOLUTION / "docs" / "q3_generation.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    meta, x, _, _ = load_q3_data()
    labels = meta.tasting_category_fine.reset_index(drop=True)
    x = x.reset_index(drop=True)
    if min(labels.value_counts()) < 3:
        raise ValueError("Repeated 3-fold validation requires at least 3 rows per style")
    records, sensitivity, assignments, marginal, cv_support_rows = run_validation(x, labels)
    records.to_csv(OUT / "cv_metrics.csv", index=False)
    sensitivity.to_csv(OUT / "sensitivity_metrics.csv", index=False)
    assignments.insert(3, "beer", assignments.row_index.map(meta.beer))
    assignments.to_csv(OUT / "cv_assignments.csv", index=False)
    marginal.to_csv(OUT / "cv_marginal_w1_by_feature.csv", index=False)
    cv_support_rows.insert(3, "beer", cv_support_rows.row_index.map(meta.beer))
    cv_support_rows.to_csv(OUT / "cv_holdout_support_metrics.csv", index=False)
    cv_support = cv_support_rows.groupby("style").agg(
        reference_rows=("row_index", "size"), unique_beers=("beer", "nunique"),
        latent_distance_q95=("latent_distance", lambda v: v.quantile(0.95)),
        nn_train_style_q95=("nn_train_style", lambda v: v.quantile(0.95))).reset_index()
    cv_support["calibration"] = "repeated_3fold_heldout_descriptive_not_formal_fullfit_coverage"
    cv_support.to_csv(OUT / "cv_support_thresholds.csv", index=False)
    mc_variability = records.groupby(["repeat", "fold", "style", "method"])[METRICS].agg(["mean", "std"])
    mc_variability.columns = [f"{metric}_{stat}" for metric, stat in mc_variability.columns]
    mc_variability.reset_index().to_csv(OUT / "cv_mc_variability.csv", index=False)
    folds, summary = summarize_validation(records)
    folds.to_csv(OUT / "cv_fold_means.csv", index=False)
    summary.to_csv(OUT / "cv_summary.csv", index=False)
    sensitivity_all = pd.concat([sensitivity, records.loc[
        (records.method == "conditional_copula") & (records.mc == 0)]], ignore_index=True)
    sensitivity_summary = sensitivity_all.groupby(["mean_strength", "cov_strength"], as_index=False)[
        ["marginal_w1_iqr", "marginal_w1_iqr_median", "energy_pca5", "spearman_mae", "copy_rate",
         "variance_coverage_0p5_2"]].mean()
    sensitivity_summary.to_csv(OUT / "sensitivity_summary.csv", index=False)
    model = ConditionalGaussianCopula(*MAIN_PARAMS).fit(x, labels)
    joblib.dump(model, OUT / "fitted_model.joblib")
    pd.DataFrame(model.fit_diagnostics_).to_csv(OUT / "fit_diagnostics.csv", index=False)
    _, scale_source = training_scale(model.impute(x))
    pd.DataFrame({"feature": x.columns, "scaling_source": scale_source,
                  "imputation_median": model.median_}).to_csv(OUT / "full_preprocessing.csv", index=False)
    support_rows, support = support_diagnostics(model, x, labels)
    support_rows.insert(1, "beer", support_rows.row_index.map(meta.beer))
    support_rows.to_csv(OUT / "full_style_support_metrics.csv", index=False)
    support.to_csv(OUT / "full_style_support_thresholds.csv", index=False)
    full_diagnostics, zero_diagnostics = [], []
    for index, style in enumerate(STYLES):
        generated = model.sample(style, N_GENERATED, SEED + 100000 + index)
        generated.insert(0, "sample_id", [f"{safe_style(style)}_{i:03d}" for i in range(1, N_GENERATED + 1)])
        generated.insert(1, "target_style", style)
        generated.to_csv(OUT / f"generated_{safe_style(style)}.csv", index=False)
        diag = generated_diagnostics(model, generated[x.columns], style, x, labels, support, cv_support)
        diag["sample_id"] = generated.sample_id
        full_diagnostics.append(diag)
        style_x = x.loc[labels == style]
        imputed_full = model.impute(x)
        imputed_style = model.impute(style_x)
        for j, feature in enumerate(x.columns):
            zero_diagnostics.append({"style": style, "feature": feature,
                "full_observed_zero_fraction": float(x[feature].eq(0).sum() / x[feature].notna().sum()),
                "style_observed_zero_fraction": float(style_x[feature].eq(0).sum() / style_x[feature].notna().sum()),
                "full_imputed_zero_fraction": float(np.mean(imputed_full[:, j] == 0)),
                "style_imputed_zero_fraction": float(np.mean(imputed_style[:, j] == 0)),
                "generated_zero_fraction": float(generated[feature].eq(0).mean())})
    full_diagnostics = pd.concat(full_diagnostics, ignore_index=True)
    full_diagnostics.to_csv(OUT / "generated_diagnostics.csv", index=False)
    pd.DataFrame(zero_diagnostics).to_csv(OUT / "generated_zero_diagnostics.csv", index=False)
    elapsed = time.perf_counter() - start
    summary_json = {
        "date": "2026-10-23",
        "n_real": len(x), "n_features": x.shape[1], "n_styles": labels.nunique(),
        "mean_strength": MAIN_PARAMS[0], "cov_strength": MAIN_PARAMS[1],
        "parameter_policy": "prespecified_fixed_main_and_separate_sensitivity_no_outer_selection",
        "cv_repeat_seeds": REPEAT_SEEDS, "cv_folds": 3, "mc_repeats": MC_REPEATS,
        "generated_n_each_draw": N_GENERATED, "validation_records": len(records),
        "validation_test_style_n_min": int(records.holdout_style_n.min()),
        "validation_test_style_n_max": int(records.holdout_style_n.max()),
        "full_generation_rows": len(full_diagnostics),
        "full_generation_copy_rate": float(full_diagnostics.exact_training_copy.mean()),
        "full_generation_within_insample_latent_q95": float(full_diagnostics.within_latent_style_q95.mean()),
        "full_generation_within_cv_latent_q95": float(full_diagnostics.within_cv_latent_style_q95.mean()),
        "minimum_style_cov_eigenvalue": float(min(d["minimum_cov_eigenvalue"] for d in model.fit_diagnostics_)),
        "elapsed_seconds": elapsed,
        "limits": ["Tiny style samples; 300 generated rows are Monte Carlo, not independent real evidence",
                   "Shared empirical margins and shrinkage may dilute style-specific distributions",
                   "Full-fit support thresholds are descriptive, not independently calibrated",
                   "Chemical target spectra are not brewing recipes or sensory panel validation"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, sensitivity_summary, full_diagnostics, support, elapsed, x, labels)
    print(json.dumps(summary_json, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
