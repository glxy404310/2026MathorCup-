# A题感官量尺、术语映射与原始来源核查

核查日期：2026-10-23。核查范围：附件 3.xlsx、8.xlsx、1.csv、4.csv，以及作者公开论文和官方数据仓库；没有训练模型，没有把分位等级当成真实评分。配套字段表见 `../outputs/audit/sensory_dimensions_review.csv`。

## 结论

1. **附件没有识别出可直接监督学习的 250 款啤酒原始 0–6 分标签。** `4.csv` 是按品评员标准化后再聚合的连续感官结果，可以作为连续预测目标，不能解释成原始量尺。
2. **`3.xlsx` 是空白表单模板。** 它含 `beer_1` 至 `beer_12` 共 12 张表单和 `summary`，并非 10 个已评分真实样本。
3. **题目要求的 Aroma/Flavour 45 维可以明确定位。** 按 `8.xlsx / Sensory names` 的 Trained panel 元数据，Aroma 15 维、Taste 30 维；排除 `alcohol, aftertaste, body, co2, overall` 后恰好 45 维。这里表单 Flavour 对应元数据 Taste，不应按表单左右位置简单分类。
4. **不能用 `4.csv` 的列均值、列标准差逆标准化到 0–6。** 未识别出每位品评员的原始评分、标准化均值/标准差及完整品评关系，转换不可辨识。按分位数生成 0–6 或拉伸到 [0,6] 只能称人为等级/展示刻度。
5. **化学变换必须逐字段判断。** `1.csv` 的前 37 个参数（`acetaldehyde` 至 `kcalperc`）与 `8.xlsx / Calibrated chemical dataset` 的对应值符合 log10 变换；余下 33 个有校准值的 GCMS 重叠参数不直接符合，不能全表取 `10**x` 当作浓度。

## 评分卡证据与可执行提取规则

对 `3.xlsx` 的公式、缓存值、所有评分输入位及底层文件结构进行只读核查：

- 12 张 `beer_n` 表单除 `S3` 的模板编号外，所有单元格值完全相同；各表单有 54 个公式，`summary` 有 612 个公式，其公式缓存值全部为 0。
- 0–6 的数字位于 `C8,E8,G8,I8,K8,M8,O8` 等量尺说明行；实际输入在上一行，例如麦芽香气输入 `C7,E7,G7,I7,K7,M7,O7`。右侧对应 `S7,U7,W7,Y7,AA7,AC7,AE7`。全部评分输入位均为空，12 张表单也没有可承载图形勾选的 drawing/VML/control/comment 对象。
- 12 张表单的子描述符勾选位也为空；不应将公式对空白输入默认返回的 0 解释为品评员选择了“未检出”。
- `summary!B1:AY1` 提供 50 个目标名称，`B2:AY13` 对应模板槽位 1–12；`AZ` 是评论。`beer_n` 中的 `Beer#` 只是会话模板编号，未提供与 `1.csv/4.csv` 的 250 款酒 `beer` 主键连接关系。

若后续获得实际填写表单，可按以下规则提取：

| 项目 | 表单坐标/规则 |
|---|---|
| 麦芽、啤酒花、酯类总体 Aroma | `Q7, Q20, Q33`；输入分别为同一行 `C,E,G,I,K,M,O` 七个量尺位 |
| 对应 Flavour/Taste 总体 | `AG7, AG20, AG33`；输入为同一行 `S,U,W,Y,AA,AC,AE` |
| Aroma 子描述符 | 输出 `Q10,Q12,Q14,Q16; Q23,Q25,Q27,Q29; Q36,Q38,Q40,Q42`；勾选位是同一行的 `C` 列 |
| Flavour 子描述符 | 与上行同样行号，输出 `AG`，勾选位 `S` |
| 子描述符含义 | 被勾选时继承该类总体强度，未勾选时为 0；必须先确认该类总体已完成评分。不是独立 0–6 强度输入 |
| 其他数值维度 | 详见配套 CSV 的 `correct_form_output_cell`；所在行的七个量尺位与左右区域规则一致 |

原表公式只检查 1–6 对应输入位，多个勾选会被解释为最高已选值，完全空白则返回 0。因此可靠解析必须先核验七个输入位：无选择为缺失、多选择为冲突、恰好一个选择才读出 0–6，不应直接读取缓存公式值。若原表只填非空标记，标记本身无需为数字。

**模板中发现的实质错误：** `summary!AX2:AX13` 的 `barnyard` 公式引用各 `beer_n!AG76`，这是 Stale hops；表单 Barnyard/horsey 的正确输出应为 `AG90`。这是模板引用错误，不能据此宣称 `4.csv` 的 barnyard 被污染；现有 `barnyard` 与 `stale_hops` 两列并不相同。不得为修复模板而覆盖原始附件。

另有 Acetaldehyde 两个量尺（`Q45,AG45`），但未进入 `summary` 的 50 个感官目标，也未进入 `4.csv`，不能自行补成第 51/52 个标签。`AG62` 和 `AG94` 是无相应属性标题的遗留公式，也不构成目标。

## 45 维名称与量尺含义

`8.xlsx / Sensory names` 中 `dataset=Trained panel` 恰有 50 条：Aroma 15、Taste 30、Palate 4、Overall 1。前两类为本题 45 维。50 维中 26 条元数据类型为 numeric、24 条为 categorical；45 维中为 21 条 numeric、24 条 categorical。这里 categorical 记录原表单“选中描述符”机制，经过总体强度加权、品评员标准化和聚合后，`4.csv` 中这些字段仍是连续值，不应当作已经给定的整数分类标签。

对每个 Malt/Hops/Esters 家族，Aroma 与 Flavour 各包含 1 个 overall 和 4 个子描述符，总计 30 维；余下 15 个 Taste 属性为 `acidity, bitternes, sweetness, X4vg, diacetyl, dms, metallic, stale_hops, t2n, orange, coriander, clove, lactic, acetic, barnyard`。

术语映射应保留原始主键并单独添加显示名称：

- `A_` = Aroma；`F_` = Flavour，元数据中写作 Taste。
- `A_ester_fruity/F_ester_fruity` 是 `3.xlsx / summary` 的单数拼写，`4.csv/8.xlsx` 使用 `A_esters_fruity/F_esters_fruity`。
- `4vg`（模板）对应 `X4vg`（感官列），化学表中的 `X4VG` 是化学测量，不是同一列。
- `bitternes` 为源文件拼写，应映射显示名称 Bitterness，不能在连接时默默改键。
- `hops_noble` 对应 Noble hops (dried herbs, dried grass, hay)，元数据简写 grassy；`malt_burn` 对应 Smoked, burnt toast，简写 smoked。
- `esters_isoaa` 对应 Banana, pear, candy；`esters_ethac` 对应 Ethyl acetate, solvent, nail polish；`esters_fruity` 对应 Sweet, fruity, apples。这些是感官描述符，不等于对应化合物浓度。
- `orange` 的 `CleanName_PaperLong` 错写为 Hops taste - citrus，但短名及品评反馈名均为 Orange；应保留独立 `orange` 目标，不与 `F_hops_citrus` 合并。
- `t2n` 的长名写 Aging aroma - cardboard，但 `Category_Paper=Taste`；45维筛选按明确分类和原始键执行。`8.xlsx` 中 Ratebeer 的 `aroma_fruity/taste_fruity` 被映射到 nail polish 等可疑对应不能直接视为科学等价关系，若后续做评论对齐须单独验证。

## 为什么 4.csv 无法逆回真实 0–6

原作者方法写明：**“Scores were scaled by standard deviation and mean-centered per taster.”** 随后跨品评员汇总。[原论文 Methods，PDF 第12页](https://www.nature.com/articles/s41467-024-46346-0.pdf)。该句说明标准化发生于品评员层，不是对最终 250 行逐列做一次标准化。

设原评分为 `s[i,t,j]`，品评员为 `t`，感官属性为 `j`，标准化参数为 `mu[t,j], sigma[t,j]`，聚合记录近似为：

`z[i,j] = mean_t((s[i,t,j] - mu[t,j]) / sigma[t,j])`。

一般情况下，平均后的 `z[i,j]` 不足以恢复每人的原评分，亦不足以恢复原量尺上的平均评分。并且原作者说明并非每位品评员出席每次品评，因此每款酒的参与者可能不同。即使拿到一组全体样本均值/标准差，也不能替代每位品评员的参数和参与关系。

本地数值复核：`4.csv` 的 50 列均值范围约 `[-0.049312, 0.062833]`，样本标准差范围 `[0.369659, 0.858463]`，全表最小/最大值约 `-2.150729 / 6.172383`。这与“按人标准化再平均”相容，与“这些就是0–6原始分”不相容。不能把再标准化后均值0/方差1误认为原始变换已恢复。

可继续做的工作是 45 维连续预测并在现有尺度计算 MAE/RMSE/R²。真实 0–6 评分任务仍需原始标签或经独立实际评分校准的映射；未获得前应明确留项，不填伪标签。

## 化学对数底数与零值

对 `1.csv` 和 `8.xlsx / Calibrated chemical dataset` 按 `beer` 对齐，70 个校准重叠字段中，前 37 个常规理化参数符合：

`x[i,j] = log10(c[i,j])`，若校准值为 0，则用该列最小正值的一半代替后取对数。

37 列中共 755 个零值符合此替代规则；考虑校准数据的末位精度，全部对应记录的最大绝对差约 `2.93e-9`。因此这部分对数底数已由附件数值验证，而非猜测。逆变换会还原替代值，不能自行判断成真实检出浓度；零值及检出限解释仍须保留。

其余 33 个 GCMS 校准重叠字段不符合上述直接关系，可能涉及相对信号、另行校准或不同变换。仅凭当前检查不能决定每列变换和单位。须结合 `Sensory thresholds` 的 `calibrated/unit`、原作者方法与原始测量说明逐列核验，不能将统一逆对数结果用于阈值/OAV计算。

## 原始来源核验与公开数据缺口

定位到 Schreurs 等人的 *Predicting and improving complex beer flavor through machine learning*，Nature Communications 15, 2368 (2024)，[DOI 10.1038/s41467-024-46346-0](https://doi.org/10.1038/s41467-024-46346-0)。作者官方仓库为 [Zenodo 10653704](https://zenodo.org/records/10653704)，该记录明确关联本论文。

已定向读取公开文件清单、README、ANOVA脚本和补充工作簿；补充工作簿只读入内存，没有导入训练、没有上传本地数据。该工作簿 MD5 为 `52645e38d90aafda272532e19e2fea1c`，与仓库校验值相同。本地 `1.csv` 对应其 `Supplementary File S1`，本地 `4.csv` 对应 `Supplementary File S4`：`beer` 顺序、数据形状及缺失掩码一致，数值最大绝对差分别为 `8.89e-16` 和 `1.12e-16`，确认了数据来源关系。

公开包中未识别出 250 款酒对应的原始逐品评员评分或每位品评员的标准化参数：

- [README](https://zenodo.org/api/records/10653704/files/README.txt/content) 指向补充工作簿作为公开数据入口。
- [感官重复样本 ANOVA 脚本](https://zenodo.org/api/records/10653704/files/Sensory%20data%20quality%20control%20-%20ANOVA%20on%20repeated%20samples.Rmd/content) 第22行读取 `Raw_Sensory_data_Trained_panel.csv`，但当前公开记录的文件清单没有该文件，补充工作簿也没有同名原始逐人评分页。这是原始数据曾存在的代码线索，不是可用标签。
- 补充工作簿含 `Validation Results 1/2/3` 的后续加料验证实验记录，样本编码和属性集合不同，不能拿来充当当前 250 款酒 × 45 维标签。
- `Supplementary Table S2` 是重复品评一致性检验结果，不是 12 款酒的原始0–6评分。`Supplementary File S3` 仍为评分卡。

准确状态为“**在本地附件和本次检查的官方公开包中未识别到可恢复0–6的原始评分/参数**”，不延伸断言作者没有这些数据，也不声称已经完成原始量尺预测。
