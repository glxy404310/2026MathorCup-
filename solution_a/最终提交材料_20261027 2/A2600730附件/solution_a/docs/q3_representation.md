# 第三问混合化学表示与感官代理

运行日期：2026-10-23。本文记录附件表示、实际训练、折外验证及迁移代理的适用范围。

## 1. 226 维表示的可复现数据契约

入口为 `src/q3_data.py::load_q3_data()`，依次返回 `(meta, X, y, manifest)`。样本沿用第一问按 `beer` 精确连接的顺序：250 行，`meta` 三列为 `beer, beer_id, tasting_category_fine`；`y` 为附件 `4.csv` 中保留的 45 个感官目标，目标尺度保持附件标准化表示。身份字段和风格不进入感官预测特征。

`1.csv` 的 231 个数值列去掉五个派生/汇总指标 `kcalperc, esters_sum, aroma_hops_sum, acids_sum, sulfur_sum`，得到 226 维。对其中 69 个有校准原值的指标，以 `8.xlsx / Calibrated chemical dataset` 中按 `beer` 对应的值替换；70 个校准指标中的 `kcalperc` 已被排除。剩余 157 列保留 `1.csv` 的相对量或尚未识别变换的表示。两类特征均保持原列顺序。

每一个校准字段的单位仅以 `Sensory thresholds.field_name_R` 精确匹配附件中的 `unit`。`unfermentables` 和 `ethyl.pentanoate` 的附件单位缺失，标为 `not_provided_in_attachment`；其余 157 维标为 `supplied_relative_or_transformed_units`。这 226 维不能统一称为 mg/L 浓度，也不能对 `1.csv` 统一取指数得到原值。

校准域负值有三处：`betaglucan.mg.L.` 两处、`unfermentables` 一处，均视为无效测量并转为 NaN。清洗前选定表示有 69 个缺失，处理负值后共有 72 个缺失。校准原值的真实零保留；未识别的相对表示出现负值时也保留，不以浓度的非负规则误删。数据加载函数不补值、不标准化、不写输出；全部拟合预处理由各个训练子集负责。

`feature_manifest.csv` 每列记录位置、表示、单位及单位来源、数据来源、选定表示的原始缺失数、原 `1.csv` 缺失数、无效负值数、最终缺失数、零值数和缺陷浓度标记。`original_missing_count` 指当前选定来源在清洗前的缺失数，`original_1csv_missing_count` 另行保存原文件对照。

## 2. 四项化学缺陷的字段与阈值

所有阈值均由代码在运行时读取 `8.xlsx / Sensory thresholds`，没有外部补填。导出的 `DEFECTS` 字典按 DMS、Diacetyl、4VG、trans_2_nonenal 索引，包含 `feature, unit, threshold_median, threshold_mean, threshold_source`。

| 缺陷 | 精确校准字段 | 单位 | 中位阈值 | 均值阈值 |
|---|---|---|---:|---:|
| DMS | `DMS` | ug/L | 19 | 17.25 |
| Diacetyl | `X2.3.butanedione` | mg/L | 0.09 | 0.29925 |
| 4VG | `X4VG` | mg/L | 0.20 | 0.206333333 |
| trans-2-nonenal | `X.E..2.nonenal` | mg/L | 缺失 | 缺失 |

11 个 Pils/Lager 源样本的这四项校准浓度全部完整且非负，源 `beer` 为 124、81、218、96、230、185、139、209、119、90、28。全数据中的 trans-2-nonenal 有一处缺失，发生于其他风格。阈值缺失时应保留缺失；迁移可用源浓度不增加约束，不能把不存在的阈值写为零。

## 3. 混合表示感官代理的折外验证

固定算法为 ExtraTreesRegressor：300 棵树、`min_samples_leaf=3`、`max_features=0.7`、`n_jobs=1`。流水线先在训练子集拟合中位数补值，保留缺失指示及全空列，再训练 45 维多输出森林；不额外标准化化学量。随机种子为 20261023，外层沿用第一问相同的 5 折 shuffled KFold。代码已核对第一问保存的样本 `beer` 与折号完全相同。每折中位数只使用该折训练样本，不在全数据上提前补值。

| 5 折折外模型 | 总体 MAE | 总体 RMSE | 45 维宏平均 R² |
|---|---:|---:|---:|
| 训练折均值 | 0.409634 | 0.574456 | -0.007574 |
| 226 维混合表示 ExtraTrees | 0.341765 | 0.486517 | 0.214345 |

总体 RMSE 相对均值下降 15.31%。该结果描述当前附件风格混合中的新样本预测；没有证明未见风格或人工合成谱的预测精度。附件创建时使用的全局预处理或校准无法重新在折内拟合，故此处验证仍条件于提供的数据。

`stale_hops` 的折外表现明显较弱：森林 RMSE 为 **0.435584**，均值基线为 0.441609，仅降低约 1.36%；R² 为 **0.022878**，Spearman 为 **0.142246**。因此后续的陈旧酒花感约束只能是该模型代理指标的约束，不能据此承诺真实品评不恶化。附件没有可识别的原始 0–6 监督标签，本流程也未生成伪造的 0–6 分数。

全量拟合的 `sensory_model.joblib` 保存 `pipeline, features, targets` 等元数据，可为三种风格生成谱输出相对感官画像。其训练内拟合预测单独命名为 `fitted_standardized_NOT_validation.csv`，不当作折外精度。

## 4. 迁移源排除与配对 bootstrap

为避免 Pils/Lager 源样本在自身预测器中的训练记忆，迁移另用 `transfer_proxy.joblib`：从训练中排除全部 11 个 Pils/Lager，仅在其余 239 行上训练相同的 45 维流水线。模型文件保存全局训练行号和排除行号；11 源样本的预测与观察值分别落盘。

| 11 个 Pils/Lager 整风格留出 | 总体 MAE | 总体 RMSE | 宏平均 R² |
|---|---:|---:|---:|
| 非 Pils 训练均值 | 0.385682 | 0.471361 | -6.624450 |
| 非 Pils 训练森林 | 0.274179 | 0.361027 | -2.683408 |

宏平均 R² 对此小风格内的低方差目标很敏感，虽然总体 RMSE 优于全局均值对照，逐目标 R² 仍可能很差。11 源上的 `stale_hops` RMSE 为 **0.526276**，均值基线 0.546600；R² 为 **-0.042038**。不能只凭整体误差下降认定该特定缺陷的预测已可靠。

`stale_bootstrap.joblib` 共存 40 个仅预测 `stale_hops` 的流水线，每个以同样的非 Pils 239 行为抽样总体、配对有放回抽样 239 次，分别重新拟合中位数及 120 棵树的森林。随机抽样种子为 `20261023+2000`，第 b 个模型的森林种子为 `20261023+3000+b`。文件同时保存每次抽样在 250 行总体中的全局下标；`stale_bootstrap_indices.npz` 另存 40×239 下标和源 beer 供审计，已核验没有任何 Pils 源行进入任一抽样。

同一个模型必须分别预测同一源样本和候选，计算配对差：

\[
\Delta_b(x',x_{src})=f_b(x')-f_b(x_{src}).
\]

使用前 30 个模型进行候选筛选；后 10 个模型只在候选确定后审计一次，审计失败不得再以审计结果重选候选。二者是同一训练总体上的不同 bootstrap 模型抽样，**不是独立实测验证数据**。输出 95% 分位数只表示有限 bootstrap 集合内的差值上分位数，既非经覆盖率校准的置信上界，也不包含全部测量误差、工艺变化或模型失配。后 10 个模型尤其不能提供精确的尾概率推断。

对候选施加 `quantile_0.95(Δ_b)≤0`，仅表示在所使用模型抽样和数值口径下代理指标不增加。由于该感官项折外精度很弱、合成谱可能偏离训练分布、候选选择还会带来选择偏差，真实陈旧酒花感仍须通过新的独立化学检测和感官实验验证。

## 5. 运行与文件

从仓库根目录执行：

```bash
.venv/bin/python solution_a/src/06_q3_sensory.py
```

所有本步骤结果均在 `outputs/q3/sensory/`：

- `meta.csv, hybrid_features.csv, y45.csv, feature_manifest.csv`：带缺失标记的固定数据契约。
- `defect_thresholds.json`：附件精确字段映射及阈值，缺失保存为 JSON null。
- `oof_extra_trees.csv, oof_mean.csv, metrics_overall.csv, metrics_by_target.csv, metrics_by_fold.csv`：相同五折的预测及验证；45 维指标完整保留。
- `sensory_model.joblib`：全 250 行训练的生成画像代理。
- `transfer_proxy.joblib, transfer_pils_heldout_*.csv`：排除全部 Pils 的 45 维迁移画像代理及源样本留出检查。
- `stale_bootstrap.joblib, stale_bootstrap_indices.npz`：40 个源排除的配对差代理、前 30/后 10 分工及抽样审计。
- `summary.json`：参数、版本、上述指标、边界与运行时间。

已核验：226 列顺序与三个模型工件完全一致；250×45 与 11×45 预测均有限；40×239 bootstrap 下标全部排除源；同一候选等于源时所有配对差为零。模型文件以 joblib 压缩保存，三个文件合计约 43.2 MiB；加载应保持所记录的 scikit-learn 版本一致。
