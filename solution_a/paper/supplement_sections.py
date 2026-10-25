"""Insert supplementary results without changing the primary result tables."""
import json
import pandas as pd


CONCENTRATION_HEADERS = ["beer", "化合物与单位", "源浓度", "迁移浓度", "最严上界", "剩余裕量", "检查"]
CONCENTRATION_WIDTHS = [1.1, 3.7, 2.4, 2.4, 2.4, 2.4, 1.6]


def concentration_rows(frame):
    names = {"trans_2_nonenal": "trans-2-nonenal"}
    return [[str(r.source_beer), names.get(r.defect, r.defect) + "\n" + r.unit_display,
             f"{r.source:.6g}", f"{r.transferred:.6g}", f"{r.cap:.6g}",
             f"{r.slack:.6g}", "通过" if r.passed else "未通过"] for r in frame.itertuples()]


def paragraph(text):
    return {"type": "paragraph", "text": text}


def table(caption, headers, rows, widths):
    return {"type": "table", "caption": caption, "headers": headers,
            "rows": rows, "widths": widths, "compact": True}


def add_main_sections(q1, q2, q3, root):
    results = root / "supplement/results"
    metric = pd.read_csv(results / "q1/metrics_overall.csv")
    grouped = pd.read_csv(results / "q1/balanced_group_metrics_overall.csv")
    names = {"mean": "训练折均值", "extra_trees": "ExtraTrees",
             "gbr_per_target": "逐目标GBR", "hgb_per_target": "逐目标HGB"}
    rows = []
    for r in metric.loc[metric.protocol.eq("random_3fold")].itertuples():
        rows.append(["随机三折", names[r.model], f"{r.rmse:.4f}", f"{r.macro_r2:.4f}"])
    for r in grouped.itertuples():
        rows.append(["整风格三折", names[r.model], f"{r.rmse:.4f}", f"{r.macro_r2:.4f}"])
    random_rmse = metric.loc[metric.protocol.eq("random_3fold") & metric.model.eq("extra_trees"), "rmse"].iloc[0]
    grouped_rmse = grouped.loc[grouped.model.eq("extra_trees"), "rmse"].iloc[0]
    pressure = metric.loc[metric.protocol.eq("random_3fold_test_mask")]
    stress_text = []
    for rate in [.1, .2]:
        values = pressure.loc[pressure.rate.eq(rate), "rmse"]
        stress_text.append(f"{rate:.0%}掩码的RMSE均值为{values.mean():.4f}、掩码间标准差为{values.std(ddof=1):.4f}")
    q1 += [{"type": "heading", "level": 2, "text": "5.6 补充对照与未见风格验证"},
           paragraph("主结果完成后增加固定参数三折对照，用于检查模型家族及部署范围；这是同一批250款酒上的补充分析，"
                     "不作为独立外部测试，也不替换表5-1的五折指标。四种模型使用相同随机划分，所有补值均只在训练折估计。"
                     "ExtraTrees沿用300树、叶节点至少3样本及0.7特征比例；GBR逐目标拟合100棵深度2的树，学习率0.05、"
                     "叶节点至少5样本；HGB逐目标迭代100次，学习率0.05、最多7叶、叶节点至少10样本、正则系数1。未据此调参或逐目标拼接赢家。"),
           table("表5-3 固定参数补充对照与整风格留出", ["验证", "模型", "RMSE", "宏平均R²"], rows, [4, 6, 3, 3]),
           paragraph(f"整风格划分由GroupKFold按类别样本数平衡，三个测试折为83、83、84款，训练与测试风格互斥。"
                     f"与训练样本量接近的随机三折相比，ExtraTrees的RMSE增加{grouped_rmse/random_rmse-1:.2%}。"
                     "这反映未见风格预测更困难，不能将随机划分下的精度直接推广至新风格。固定参数提升树没有整体优势，故保留ExtraTrees主方案。"),
           paragraph("对同一随机三折模型再施加20组配对测试掩码：" + "；".join(stress_text) + "。"
                     "这里的波动仅来自测试掩码，不含重新分折或重训的不确定性，也不模拟检出限删失和整组仪器失效。")]

    ablation = pd.read_csv(results / "q1_indicator_ablation/paired_metrics.csv")
    labels = {"all": "全部样本", "original_missing": "原始有缺失", "original_complete": "原始完整"}
    ablation_rows = [[labels[r.cohort], str(r.n), f"{r.rmse_with_indicator:.6f}",
                      f"{r.rmse_without_indicator:.6f}", f"{r.delta_rmse:+.6f}"]
                     for r in ablation.loc[ablation.scenario.eq("original")].itertuples()]
    hidden = ablation.loc[ablation.cohort.eq("all")].set_index("scenario")
    q1 += [paragraph("另按原五折、原模型参数及原测试掩码进行缺失指示消融，仅切换是否附加指示列，"
                     "不根据结果重新选择参数。63款原始有缺失和187款原始完整的分组在新增遮蔽前确定，结果见表5-4。"
                     "有指示分支的三组折外预测与原保存结果逐值一致。"),
           table("表5-4 原始缺失分组下的指示变量消融", ["样本组", "款数", "有指示RMSE", "无指示RMSE", "差值（有−无）"],
                 ablation_rows, [3.5, 1.2, 3.7, 3.7, 3.9]),
           paragraph(f"新增10%缺失时，整体有、无指示的RMSE为{hidden.loc['mask10', 'rmse_with_indicator']:.6f}、"
                     f"{hidden.loc['mask10', 'rmse_without_indicator']:.6f}；20%时为"
                     f"{hidden.loc['mask20', 'rmse_with_indicator']:.6f}、{hidden.loc['mask20', 'rmse_without_indicator']:.6f}。"
                     "原始有缺失组在三种情景下均略有改善，原始完整组则均略差，说明收益很小且依赖样本和遮蔽情景。"
                     "添加指示也改变有效输入维数及随机候选特征数，不能把差异解释为缺失模式的因果作用。"
                     "此处复用原五折的一组配对掩码，与上述三折20组掩码的波动含义不同。")]

    bootstrap = json.loads((results / "q2_stratified/summary.json").read_text(encoding="utf-8"))
    q2.insert(next(i for i, b in enumerate(q2) if b.get("text", "").startswith("6.3 ")),
              paragraph(f"另在22种风格内分别有放回抽取原样本数，固定混合比例重做200次Bootstrap。原135项的平均频率由"
                        f"{bootstrap['original_top3_mean_frequency_global']:.3f}变为{bootstrap['original_top3_mean_frequency_stratified']:.3f}，"
                        f"达到0.8的位置由14变为{bootstrap['original_top3_stratified_frequency_ge_0p8']}。"
                        "乙酸乙酯对应感官的频率为0.985，4VG为0.890。该补充不改全样本相关前三；"
                        "固定风格比例仍保留风格之间的差异，不能视为已消除混杂。200次频率的小幅差别亦受蒙特卡洛误差影响。"))
    permutation = json.loads((results / "q2_permutation/summary.json").read_text(encoding="utf-8"))
    importance = pd.read_csv(results / "q2_permutation/importance.csv")
    case_rows = []
    cases = [("酯味指甲油(F)", "F_esters_ethac", "ethyl_acetate", "乙酸乙酯"),
             ("4VG感官", "X4vg", "X4VG", "4VG"),
             ("麦芽总体味(F)", "F_malt_all", "ethanol..v.v.", "乙醇"),
             ("陈旧酒花", "stale_hops", "ethyl_acetate", "乙酸乙酯")]
    for label, target, feature, name in cases:
        r = importance.loc[importance.sensory_target.eq(target) & importance.feature.eq(feature)].iloc[0]
        case_rows.append([label + "—" + name, str(int(r.candidate_rank)), f"{r.delta_mse:.5f}",
                          f"{r.fold_sd:.5f}", f"{int(r.positive_folds)}/5"])
    q2 += [{"type": "heading", "level": 2, "text": "6.5 折外置换与预测依赖"},
           paragraph("为区分边际关联与模型使用的信息，按第一问原五折划分和固定ExtraTrees参数重拟合，"
                     f"本次折外RMSE为{permutation['rmse']:.6f}；主表原保存值为{permutation['frozen_oof_rmse']:.6f}。"
                     "在每个验证折中分别置换一个输入变量，连同其缺失指示一起移动，其他输入和模型固定；每折重复5次。"
                     "记f_fk为第f折对第k目标的预测器，V_f为验证样本集合、n_f为其大小，π为该折随机排列，置换引起的误差变化为"),
           {"type": "equation", "label": "6-5", "latex":
            "\\Delta_{jk}^{(f,r)}=\\frac{1}{n_f}\\sum_{i\\in V_f}[(y_{ik}-f_{fk}(x_i^{\\pi_{j,r}}))^2-(y_{ik}-f_{fk}(x_i))^2]."},
           paragraph("先对每折5次取均值，再按验证样本数合并；负值保留，不截为零。表6-3给出代表物，"
                     "标准差按五个折均值计算，正值折数只作稳定性描述，不作为显著性检验。"
                     "对全部231项输入计算后，仅在与相关分析相同的218项化学候选中排序，完整45项结果见表B-3。"),
           table("表6-3 代表候选的折外置换结果", ["感官—化合物", "置换名次", "平均ΔMSE", "折间标准差", "正值折数"],
                 case_rows, [5.6, 2, 2.8, 3.1, 2.5]),
           paragraph(f"置换前三与相关前三的平均Jaccard为{permutation['mean_jaccard_with_spearman']:.3f}，"
                     f"{permutation['targets_with_no_top3_overlap']}/45个目标没有共同候选。两种排名回答不同问题，故并列保留。"
                     "4VG在边际相关、风格内相关、气味依据及预测依赖上均有支持；乙酸乙酯亦被预测器使用。"
                     "麦芽总体—乙醇虽有预测依赖，风格内相关却近零，仍可能只是风格或共同工艺的标记。"
                     "陈旧酒花的置换首项不在原相关前三中，其误差增量也小，不能据此提出加料因果方案。"),
           paragraph("相关输入可能相互替代，逐项置换还会打破真实化学组合。为检查这一问题，"
                     "每个训练折独立重建|Spearman|≥0.7网络，并将连通分量整体置换；五折分别形成"
                     + "、".join(map(str, permutation['modules_per_fold'])) + "个分量。"
                     "成员和逐折误差随附件保存，不将成员不同的分量混作同一对象求平均。"
                     "这些分析提供模型依赖证据，均不识别因果效应。第一问使用231维附件表示，第三问改用69维校准量与157维相对量并另训代理；"
                     "本问候选没有直接作为第三问的生成约束，三问的联系应限于证据核对和缺陷变量解释。")]

    module = pd.read_csv(results / "reference_tables/ethyl_acetate_F_esters_ethac_by_fold.csv")
    q2 += [paragraph("以乙酸乙酯对酯味指甲油（F_esters_ethac）的依赖为例，表6-4列出逐折单项及模块置换。"
                     "整个模块五折的误差增量均为正，单项仅四折为正，表明相邻输入可能共同承载预测信息。"
                     "模块成员跨折变化，完整成员见表B-4，不将模块与单项增量之差视作乙酸乙酯的独立贡献。"),
           table("表6-4 乙酸乙酯及其相关模块的逐折预测依赖", ["折", "模块成员数", "单项ΔMSE", "模块ΔMSE"],
                 [[str(r.fold), str(r.size), f"{r.individual_delta_mse:+.5f}", f"{r.module_delta_mse:+.5f}"]
                  for r in module.itertuples()], [1.5, 3, 5.75, 5.75]),
           paragraph("模块联合置换保留组内组合，仍会打破模块与其余输入之间的联系，因此不同于给定邻域后的条件置换。"
                     "现有Bootstrap重算的是相关排名，不能用于声称条件重要性已经稳定。4VG在五折中均为单节点，"
                     "其模块置换不提供额外的群组证据。")]

    for b in q3:
        if "caption" in b:
            b["caption"] = b["caption"].replace("表7-2", "表7-5")
        if "text" in b:
            b["text"] = b["text"].replace("表7-2", "表7-5")
    defect_rows = [
        ["DMS", "校准量，μg/L", "均值17.25；中位19", "十情景下超阈程度不增加", "给定情景的化学保护"],
        ["Diacetyl", "校准量，mg/L", "均值0.29925；中位0.09", "同上", "给定情景的化学保护"],
        ["4VG", "校准量，mg/L", "均值0.20633；中位0.20", "同上，依题面按缺陷处理", "给定情景的化学保护"],
        ["trans-2-nonenal", "校准量，mg/L", "附件空缺", "候选浓度不大于源浓度", "不保证低于感知阈值"],
        ["Stale hops", "标准化感官预测", "无单一化学行及阈值", "中心预测与配对差Q95不增加，预留模型复核", "代理检查，需实际品评验证"]]
    q3.insert(next(i for i, b in enumerate(q3) if b.get("text", "").startswith("7.7 ")),
              table("表7-2 五项缺陷的变量及保护范围", ["缺陷", "表示与单位", "附件阈值", "约束", "解释边界"],
                    defect_rows, [2.5, 2.8, 3.2, 4.2, 3.3]))
    concentration = pd.read_csv(results / "reference_tables/transfer_defect_concentrations.csv")
    at = next(i for i, b in enumerate(q3) if b.get("text", "").startswith("7.7 "))
    q3[at:at] = [paragraph("表7-3展示余量优先主方案的三款代表源，11款完整结果见表C-3。上界取十个情景的交集，"
                           "裕量为上界减迁移浓度。beer 81的DMS从8.858685升至13.8 μg/L，仍满足超阈程度不增加，"
                           "故不能把化学通过概括为四项浓度全部下降。"),
                 table("表7-3 代表源的迁移前后校准浓度", CONCENTRATION_HEADERS,
                       concentration_rows(concentration.loc[concentration.source_beer.isin([81, 96, 124])]),
                       CONCENTRATION_WIDTHS),
                 paragraph("按绝对容差10⁻¹⁰统计，DMS、双乙酰及trans-2-nonenal分别有11、7、10款达到上界；"
                           "4VG没有达到上界，11款均处于零下界。这些是所选谱的边界分布，不能据此判断某化合物是唯一限制因素；"
                           "约束对迁移幅度的影响还需单独放宽后的反事实比较。")]
    committees = pd.read_csv(results / "reference_tables/beer96_committees.csv").set_index("stage")
    stages = [("current_selection", "当前模型筛选"), ("old_selection", "旧协议模型筛选"),
              ("current_fresh_audit", "预留当前模型复核"), ("old_fresh_audit", "预留旧协议模型复核")]
    audit_rows = []
    for stage, label in stages:
        r = committees.loc[stage]
        audit_rows.append([label, str(int(r.model_count)), str(int(r.positive_count)),
                           f"{r.q95:+.5f}", "通过" if r.q95_passed else "未通过"])
    at = next(i for i, b in enumerate(q3) if b.get("text", "").startswith("7.9 "))
    q3[at:at] = [paragraph("beer 96说明化学保护与预测复核需分别检查：6004个名义候选中有13个共同可行候选，"
                           "选中谱通过十情景化学检查，但表7-4中预留当前模型有2/20个预测增量为正，Q95变为+0.02100，"
                           "因此最终失败。旧协议复核通过不能抵消该失败；未根据预留结果重新挑选候选。"),
                 table("表7-4 beer 96的配对预测筛选与预留复核", ["阶段", "模型数", "增量为正", "配对差Q95", "Q95检查"],
                       audit_rows, [5.5, 1.6, 2.4, 3.5, 3]),
                 paragraph("正增量数按逐模型配对差大于零统计，Q95不大于零为本表检查规则。"
                           "少量正增量与Q95通过可能同时出现，不能把分位数筛选解释为所有模型均预测不增加。")]
    return q1, q2, q3


def permutation_appendix(root):
    folder = root / "supplement/results/q2_permutation"
    top = pd.read_csv(folder / "top3.csv")
    comparison = pd.read_csv(folder / "top3_comparison.csv").set_index("sensory_target")
    rows = []
    for target in comparison.index:
        selected = top.loc[top.sensory_target.eq(target)].sort_values("candidate_rank")
        rows.append([target] + [f"{r.feature}\nΔ={r.delta_mse:.3g}；{r.positive_folds}/5" for r in selected.itertuples()]
                    + [f"{comparison.loc[target, 'jaccard']:.2f}"])
    module = pd.read_csv(root / "supplement/results/reference_tables/ethyl_acetate_F_esters_ethac_by_fold.csv")
    return [paragraph("表B-3给出218项化学候选中的预测依赖前三，各项标出平均ΔMSE与正值折数。"
                      "J为与原边际相关前三的Jaccard；原表B-1保持不变。完整231项置换结果、五折相关模块及分层Bootstrap结果随支撑材料保存。"),
            table("表B-3 全部感官的折外置换候选", ["感官", "置换第1项", "第2项", "第3项", "J"], rows, [2.8, 4, 4, 4, 1.2]),
            paragraph("表B-4逐折列出表6-4所用乙酸乙酯模块的全部成员。模块只使用该折训练数据构建，"
                      "大小及成员会变化；该表不表示五折识别出了同一个固定化学组。"),
            table("表B-4 乙酸乙酯相关模块的逐折成员", ["折与成员数", "成员字段"],
                  [[f"第{r.fold}折\n{r.size}项", r.memberlist] for r in module.itertuples()], [2, 14])]
