"""Evaluate nested, style-margin generation against frozen method baselines."""
import os

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import importlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from common import SOLUTION
from q3_data import load_q3_data
from q3_generator_optimized import (CANDIDATE_METHODS, EVALUATION_STYLES,
                                    MAX_INNER_MATCH_FRACTION, MIN_INNER_NN_RATIO,
                                    OptimizedStyleGenerator, SMOOTHING_NOISE)


REFERENCE = importlib.import_module("07_q3_generation")
OUT = SOLUTION / "outputs" / "q3_optimized" / "generation"
OUTER_SEEDS = [20261123, 20262123]
N_GENERATED = 300
MC_REPEATS = 2
METHODS = ["nested_selected", *CANDIDATE_METHODS, "style_bootstrap"]
METRICS = [*REFERENCE.METRICS, "zero_rate_mae", "zero_rate_mae_training_zero_features",
           "zero_rate_mae_calibrated", "nearest_equal_feature_fraction",
           "nearest_equal_feature_fraction_p95", "generated_pair_nn_median",
           "observed_training_copy_rate"]


def extra_metrics(reference, train, heldout, generated, calibrated_columns):
    training = reference.impute(train)
    real = reference.impute(heldout)
    fake = reference.impute(generated)
    zero_error = np.abs(np.mean(real == 0, axis=0) - np.mean(fake == 0, axis=0))
    has_training_zero = np.any(training == 0, axis=0)
    calibrated_indices = [generated.columns.get_loc(c) for c in calibrated_columns]
    fake_z, training_z = reference.transform(generated), reference.transform(train)
    distances = cdist(fake_z, training_z) / np.sqrt(fake_z.shape[1])
    closest = distances.argmin(axis=1)
    equal_fraction = np.isclose(fake, training[closest], rtol=1e-10, atol=1e-12).mean(axis=1)
    pairwise = cdist(fake_z, fake_z) / np.sqrt(fake_z.shape[1])
    np.fill_diagonal(pairwise, np.inf)
    observed = train.to_numpy(dtype=float)
    copied_observed = (np.isclose(fake[:, None, :], observed[None, :, :],
                                  rtol=1e-10, atol=1e-12)
                       | np.isnan(observed)[None, :, :]).all(axis=2).any(axis=1)
    return {"zero_rate_mae": float(zero_error.mean()),
            "zero_rate_mae_training_zero_features": float(zero_error[has_training_zero].mean())
            if has_training_zero.any() else 0.0,
            "training_zero_features": int(has_training_zero.sum()),
            "zero_rate_mae_calibrated": float(zero_error[calibrated_indices].mean()),
            "nearest_equal_feature_fraction": float(np.median(equal_fraction)),
            "nearest_equal_feature_fraction_p95": float(np.quantile(equal_fraction, 0.95)),
            "observed_training_copy_rate": float(copied_observed.mean()),
            "generated_pair_nn_median": float(np.median(pairwise.min(axis=1)))}


def fold_summary(records):
    folds = records.groupby(["repeat", "fold", "style", "method"], as_index=False)[METRICS].mean()
    summary = folds.groupby(["style", "method"])[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return folds, summary.reset_index()


def nested_validation(meta, x, styles, calibrated):
    records, assignments, selections, inner_details, margins = [], [], [], [], []
    for repeat, seed in enumerate(OUTER_SEEDS, start=1):
        outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(outer.split(x, styles), start=1):
            print(f"Optimized generation repeat={repeat}, fold={fold}", flush=True)
            train_x, train_labels = x.iloc[train], styles.iloc[train]
            selection_seed = seed + fold * 100000
            generator = OptimizedStyleGenerator(selection_seed=selection_seed).fit(train_x, train_labels)
            reference = generator.fitted_candidates_["legacy_copula"]
            pca, pc_scale = REFERENCE.pca_coordinates(reference, train_x)
            for row in generator.selection_table_:
                selections.append({"repeat": repeat, "fold": fold, "selection_seed": selection_seed, **row})
            details = generator.inner_scores_.copy()
            details.insert(0, "outer_repeat", repeat)
            details.insert(1, "outer_fold", fold)
            inner_details.append(details)
            assignments.extend({"repeat": repeat, "fold": fold, "row_index": int(i),
                "beer": meta.beer.iloc[i], "style": styles.iloc[i], "role": "test"} for i in test)
            for style_index, style in enumerate(EVALUATION_STYLES):
                style_train = train_x.loc[train_labels == style]
                heldout = x.iloc[test].loc[styles.iloc[test] == style]
                for mc in range(MC_REPEATS):
                    sample_seed = seed + fold * 10000 + style_index * 100 + mc
                    draws = {method: model.sample(style, N_GENERATED, sample_seed)
                             for method, model in generator.fitted_candidates_.items()}
                    draws["nested_selected"] = draws[generator.selected_methods_[style]]
                    bootstrap_indices = np.random.default_rng(sample_seed).integers(0, len(style_train), N_GENERATED)
                    draws["style_bootstrap"] = pd.DataFrame(reference.impute(style_train)[bootstrap_indices], columns=x.columns)
                    for method in METHODS:
                        generated = draws[method]
                        metrics, marginal = REFERENCE.evaluate(reference, train_x, style_train, heldout,
                                                                 generated, pca, pc_scale)
                        records.append({"repeat": repeat, "fold": fold, "style": style,
                            "method": method, "selected_method": generator.selected_methods_[style]
                            if method == "nested_selected" else method,
                            "outer_seed": seed, "selection_seed": selection_seed,
                            "mc": mc, "sample_seed": sample_seed,
                            "train_n": len(train_x), "style_train_n": len(style_train),
                            "holdout_style_n": len(heldout), "generated_n": len(generated),
                            **metrics, **extra_metrics(reference, style_train, heldout, generated, calibrated)})
                        if mc == 0 and method in ("nested_selected", "legacy_copula", "style_bootstrap"):
                            margins.extend({"repeat": repeat, "fold": fold, "style": style,
                                "method": method, "feature": feature, "w1_iqr": float(value)}
                                for feature, value in zip(x.columns, marginal))
    return (pd.DataFrame(records), pd.DataFrame(assignments), pd.DataFrame(selections),
            pd.concat(inner_details, ignore_index=True), pd.DataFrame(margins))


def build_report(summary, selections, full_model, full_diag, elapsed):
    lines = ["# 第三问生成模型优化：局部风格边缘与嵌套选择", "",
        "运行日期：2026-10-23。使用 250 款啤酒、226 个混合特征（69 个校准字段与 157 个相对字段）。"
        "旧第三问结果完整保留，本轮文件单独位于 `outputs/q3_optimized/generation/`。", "",
        "## 优化针对的问题", "",
        "旧模型对全部风格共用单列经验边缘，并把小风格均值向全局收缩。Kriek、Lambic "
        "分别只有 10、11 款真实样本，风格差异可能在这两步被稀释。本轮把风格边缘与相关结构分开处理，"
        "保持小风格的观测边缘，仍用收缩相关矩阵共享信息。", "",
        "在查看本轮外层结果前固定三个候选：", "",
        "1. `style_margin_copula`：各风格分别拟合经验边缘和中秩正态分数；局部 Ledoit–Wolf "
        "相关以 n/(n+10) 的权重混合全部训练风格正态分数的共享相关。相关矩阵对角归一到 1，"
        "从零均值 Gaussian 抽样后用风格局部离散经验逆分布输出。", 
        "2. `style_smoothed_mixture`：使用相同边缘和相关，从风格训练行的正态分数中抽一个中心，"
        "加标准差固定为 0.45 的相关 Gaussian 噪声；该噪声强度固定，不搜索近零扰动。", 
        "3. `legacy_copula`：旧条件 copula 固定参数 (5,20)，作为可被内层选择的保守退路。", "",
        "各风格中位数补缺，风格某列全缺失时回退全部训练行中位数；所有统计量均只用当前训练折。"
        "局部逆分布中的零拥有概率区间，Gaussian 候选在标准正态边缘下按期望再现风格训练零频率，"
        "平滑混合候选不保证相同频率。相对字段中的数值零也可能只是原表示的基准，不能全解释为物质不存在。", "",
        "## 训练内选择与外层检验", "",
        "外层采用预先固定的 20261123、20262123 两个种子的 22 风格分层 3 折划分。每个外层训练集"
        "再按全部训练风格做分层 2 折，分别为四个评价风格按内层平均五维能量距离选方法。"
        f"候选必须在每个内折都无整行复制、生成/真实留出近邻距离中位数比至少 {MIN_INNER_NN_RATIO:g}，"
        f"且与最近训练行相等的特征比例中位数不超过 {MAX_INNER_MATCH_FRACTION:.0%}；"
        "bootstrap 只作基准，不参与选择。复制检查同时比较完整补缺向量，以及仅比较训练行原本有观测"
        "的各列，避免不同中位数补缺掩盖原观测的整行复制。若无方法通过则退回旧模型，并保留失败诊断。", "",
        "内层和外层中，同一数据划分的所有候选、被选方法及 bootstrap 均使用同一旧 copula 在该训练折"
        "拟合的全局 rank 变换、五维 PCA 和 PCA 标准差作评价坐标。没有用各候选自己的局部 CDF 或 PCA "
        "计算可比较能量距离。每个外层方法每风格生成 300 行，使用两个 Monte Carlo 种子；"
        "先平均同折的 Monte Carlo，再汇总 6 个划分。", "",
        "本轮重新划分仍使用同一批已经分析过的 250 款啤酒。嵌套选择避免把本轮外层行直接用于选方法，"
        "但新种子不等于新增独立验证数据；不能把探索后的新划分称为独立确认性检验。", "",
        "|风格|方法|五维能量距离↓|平均边缘W1↓|边缘W1中位数↓|校准零频差↓|NN距离比|复制率|方差覆盖|", 
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    names = {"nested_selected": "训练内选择", "legacy_copula": "旧copula", "style_bootstrap": "风格bootstrap",
             "style_margin_copula": "局部边缘copula", "style_smoothed_mixture": "平滑混合"}
    for _, row in summary.iterrows():
        lines.append(f"|{row['style']}|{names[row['method']]}|{row.energy_pca5_mean:.4f}|"
            f"{row.marginal_w1_iqr_mean:.4f}|{row.marginal_w1_iqr_median_mean:.4f}|"
            f"{row.zero_rate_mae_calibrated_mean:.4f}|{row.nn_ratio_mean:.3f}|"
            f"{row.copy_rate_mean:.1%}|{row.variance_coverage_0p5_2_mean:.1%}|")
    lines.extend(["", "边缘 W1 各列按外层全部训练行 IQR 标准化，零 IQR 回退训练极差、再回退 1。"
        "原始极端值仍会支配跨列均值，因此同时保留跨列中位数。能量距离使用经验 V 统计量，"
        "小留出集下偏高；所有方法的留出行和生成样本数相同。方差覆盖为生成/风格训练方差比处于"
        "[0.5,2] 的列占比，排除训练恒定列，并非总体方差置信覆盖。Spearman 相关差、"
        "最近训练行特征匹配比例与生成内近邻距离另存 CSV。", ""])
    for style in EVALUATION_STYLES:
        rows = summary.loc[summary['style'] == style].set_index('method')
        chosen, old, bootstrap = rows.loc['nested_selected'], rows.loc['legacy_copula'], rows.loc['style_bootstrap']
        change = (chosen.energy_pca5_mean / old.energy_pca5_mean - 1) * 100
        boot_change = (chosen.energy_pca5_mean / bootstrap.energy_pca5_mean - 1) * 100
        lines.append(f"- {style}：训练内选择相对旧 copula 的能量距离变化 {change:+.1f}%，"
                     f"相对 bootstrap 变化 {boot_change:+.1f}%；"
                     f"边缘 W1 中位数由 {old.marginal_w1_iqr_median_mean:.4f} 变为 {chosen.marginal_w1_iqr_median_mean:.4f}。")
    lines.extend(["", "## 选择稳定性与新颖性", "",
        "|风格|各外层训练集所选方法次数|全量训练内选择|", "|---|---|---|"])
    selected_rows = selections.loc[selections.selected]
    for style in EVALUATION_STYLES:
        counts = selected_rows.loc[selected_rows['style'] == style, 'method'].value_counts()
        count_text = "；".join(f"{names[k]} {v}/6" for k, v in counts.items())
        lines.append(f"|{style}|{count_text}|{names[full_model.selected_methods_[style]]}|")
    lines.extend(["", "300 行只增加模拟精度，不增加真实风格证据。低复制率也不证明没有记忆训练分布，"
        "因此同时报告最近训练行距离、相等特征比例、生成内近邻距离和方差覆盖。"
        "内层新颖性门槛不能保证外层或全量抽样也通过；下面保留实测结果。", "",
        "|风格|全量所选方法|完全复制率|NN距离中位数|最近行匹配比例中位数|生成内NN中位数|", 
        "|---|---|---:|---:|---:|---:|"])
    for style, group in full_diag.groupby('style'):
        lines.append(f"|{style}|{names[full_model.selected_methods_[style]]}|{group.exact_training_copy.mean():.1%}|"
                     f"{group.nn_target_style.median():.4f}|{group.equal_fraction_nearest_train.median():.3f}|"
                     f"{group.nn_generated.median():.4f}|")
    lines.extend(["", "## 使用与限制", "",
        "全量模型使用同样的训练内选择，保存为 `fullmodel.joblib`，接口为 "
        "`OptimizedStyleGenerator().fit(X,styles)` 及 `sample(style,n,seed)`，返回原 226 列顺序。"
        "另存四风格各 300 行生成谱（含 sample_id、target_style 元信息）、完整内层与外层记录、"
        "所有候选的外层比较，以及生成零频率诊断。", "",
        "全量 250 行模型只用于生成画像。迁移验证应在去除源风格之后重新拟合，不能复用这个全量"
        "生成器；迁移风格距离仍由旧模型定义，保证新旧比较的量尺一致。", "",
        "风格局部经验边缘不外推，也无法补足未观测的尾部；小风格中每列只有数个取值，"
        "相关与零频率估计不稳定。生成谱是化学目标组合，仍需酿造可行性、理化检测和感官实测验证。", "",
        "```bash", ".venv/bin/python solution_a/src/11_q3_generation_optimized.py", "```", "",
        f"运行耗时约 {elapsed:.1f} 秒，线性代数线程固定为 1。", ""])
    (SOLUTION / 'docs' / 'q3_generation_optimized.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    meta, x, _, manifest = load_q3_data()
    styles = meta.tasting_category_fine
    calibrated = manifest.loc[manifest.representation == 'calibrated', 'feature'].tolist()
    records, assignments, selections, inner_details, margins = nested_validation(meta, x, styles, calibrated)
    records.to_csv(OUT / 'cv_metrics.csv', index=False)
    assignments.to_csv(OUT / 'cv_assignments.csv', index=False)
    selections.to_csv(OUT / 'inner_selection_outer.csv', index=False)
    inner_details.to_csv(OUT / 'inner_scores_outer.csv', index=False)
    margins.to_csv(OUT / 'cv_marginal_w1_by_feature.csv', index=False)
    folds, summary = fold_summary(records)
    folds.to_csv(OUT / 'cv_fold_means.csv', index=False)
    summary.to_csv(OUT / 'cv_summary.csv', index=False)
    mc = records.groupby(['repeat', 'fold', 'style', 'method'])[METRICS].std().reset_index()
    mc.to_csv(OUT / 'cv_monte_carlo_std.csv', index=False)
    print('Fitting final optimized generator on all 250 rows', flush=True)
    full_model = OptimizedStyleGenerator(selection_seed=OUTER_SEEDS[0]).fit(x, styles)
    joblib.dump(full_model, OUT / 'fullmodel.joblib', compress=3)
    full_model.inner_scores_.to_csv(OUT / 'inner_scores_full.csv', index=False)
    pd.DataFrame(full_model.selection_table_).to_csv(OUT / 'inner_selection_full.csv', index=False)
    pd.DataFrame([{'style': style, 'method': method} for style, method in full_model.selected_methods_.items()]).to_csv(
        OUT / 'full_selected_methods.csv', index=False)
    reference = full_model.fitted_candidates_['legacy_copula']
    full_diags, zero_rows = [], []
    for i, style in enumerate(EVALUATION_STYLES):
        generated = full_model.sample(style, N_GENERATED, OUTER_SEEDS[0] + 900000 + i)
        generated.insert(0, 'sample_id', [f"optimized_{REFERENCE.safe_style(style)}_{j:03d}" for j in range(1, N_GENERATED+1)])
        generated.insert(1, 'target_style', style)
        generated.to_csv(OUT / f'generated_{REFERENCE.safe_style(style)}.csv', index=False)
        style_x = x.loc[styles == style]
        generated_x = generated[x.columns]
        fake_z, style_z = reference.transform(generated_x), reference.transform(style_x)
        pairwise = cdist(fake_z, fake_z) / np.sqrt(x.shape[1])
        np.fill_diagonal(pairwise, np.inf)
        target_distances = cdist(fake_z, style_z) / np.sqrt(x.shape[1])
        nn = target_distances.argmin(axis=1)
        equal = np.isclose(reference.impute(generated_x), reference.impute(style_x)[nn],
                           rtol=1e-10, atol=1e-12).mean(axis=1)
        raw = style_x.to_numpy(dtype=float)
        observed_copy = (np.isclose(generated_x.to_numpy()[:, None, :], raw[None, :, :],
                                   rtol=1e-10, atol=1e-12)
                         | np.isnan(raw)[None, :, :]).all(axis=2).any(axis=1)
        full_diags.append(pd.DataFrame({'sample_id': generated.sample_id, 'style': style,
            'method': full_model.selected_methods_[style], 'nn_target_style': target_distances.min(axis=1),
            'nn_generated': pairwise.min(axis=1), 'equal_fraction_nearest_train': equal,
            'latent_style_distance': reference.distance(generated_x, style),
            'observed_training_copy': observed_copy,
            'exact_training_copy': target_distances.min(axis=1) < 1e-12}))
        for feature in x:
            zero_rows.append({'style': style, 'feature': feature,
                'style_observed_zero_fraction': float(style_x[feature].eq(0).sum() / style_x[feature].notna().sum()),
                'generated_zero_fraction': float(generated[feature].eq(0).mean())})
    full_diag = pd.concat(full_diags, ignore_index=True)
    full_diag.to_csv(OUT / 'generated_diagnostics.csv', index=False)
    pd.DataFrame(zero_rows).to_csv(OUT / 'generated_zero_diagnostics.csv', index=False)
    elapsed = time.perf_counter() - started
    result = {'date': '2026-10-23', 'n_real': len(x), 'n_features': x.shape[1],
        'outer_seeds': OUTER_SEEDS, 'outer_folds': 3, 'inner_folds': 2,
        'mc_repeats': MC_REPEATS, 'generated_n_each_draw': N_GENERATED,
        'candidate_methods': list(CANDIDATE_METHODS), 'smoothing_noise': SMOOTHING_NOISE,
        'selection_uses_outer_test': False, 'selection_metric': 'shared_training_reference_pca5_energy',
        'novelty_gates': {'minimum_inner_nn_ratio': MIN_INNER_NN_RATIO,
                         'maximum_inner_nearest_equal_fraction': MAX_INNER_MATCH_FRACTION,
                         'maximum_inner_copy_rate': 0.0,
                         'maximum_inner_observed_training_copy_rate': 0.0},
        'cv_records': len(records), 'full_generation_rows': len(full_diag),
        'full_selected_methods': {s: full_model.selected_methods_[s] for s in EVALUATION_STYLES},
        'full_generation_copy_rate': float(full_diag.exact_training_copy.mean()),
        'full_generation_observed_copy_rate': float(full_diag.observed_training_copy.mean()),
        'elapsed_seconds': elapsed,
        'limits': ['New outer seeds still reuse the previously explored real dataset',
                   'Small style samples and unstable inner selection',
                   'Local-margin methods use style-imputed support; legacy uses global training support',
                   'Chemical spectra are not validated brewing recipes']}
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    build_report(summary, selections, full_model, full_diag, elapsed)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
