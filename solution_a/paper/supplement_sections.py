"""Insert supplementary results without changing the primary result tables."""
import json
import pandas as pd


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

    for b in q3:
        if "caption" in b:
            b["caption"] = b["caption"].replace("表7-2", "表7-3")
        if "text" in b:
            b["text"] = b["text"].replace("表7-2", "表7-3")
    defect_rows = [
        ["DMS", "校准量，μg/L", "均值17.25；中位19", "十情景下超阈程度不增加", "给定情景的化学保护"],
        ["Diacetyl", "校准量，mg/L", "均值0.29925；中位0.09", "同上", "给定情景的化学保护"],
        ["4VG", "校准量，mg/L", "均值0.20633；中位0.20", "同上，依题面按缺陷处理", "给定情景的化学保护"],
        ["trans-2-nonenal", "校准量，mg/L", "附件空缺", "候选浓度不大于源浓度", "不保证低于感知阈值"],
        ["Stale hops", "标准化感官预测", "无单一化学行及阈值", "中心预测与配对差Q95不增加，预留模型复核", "代理检查，需实际品评验证"]]
    q3.insert(next(i for i, b in enumerate(q3) if b.get("text", "").startswith("7.8 ")),
              table("表7-2 五项缺陷的变量及保护范围", ["缺陷", "表示与单位", "附件阈值", "约束", "解释边界"],
                    defect_rows, [2.5, 2.8, 3.2, 4.2, 3.3]))
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
    return [paragraph("表B-3给出218项化学候选中的预测依赖前三，各项标出平均ΔMSE与正值折数。"
                      "J为与原边际相关前三的Jaccard；原表B-1保持不变。完整231项置换结果、五折相关模块及分层Bootstrap结果随支撑材料保存。"),
            table("表B-3 全部感官的折外置换候选", ["感官", "置换第1项", "第2项", "第3项", "J"], rows, [2.8, 4, 4, 4, 1.2])]
