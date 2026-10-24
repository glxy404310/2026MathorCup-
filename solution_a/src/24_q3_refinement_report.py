"""Report beer81 diagnosis, grouped prediction, and finite transfer trade-offs."""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import SOLUTION

OUT=SOLUTION/"outputs/q3_refined"
SOURCES=[]
COLORS={"legacy_2004":"#929AA3","legacy_6004":"#B98632","contracted_6004":"#267DA7"}


def read(path):
    raw=path.read_bytes()
    value=json.loads(raw) if path.suffix==".json" else pd.read_csv(path)
    SOURCES.append({"path":path.relative_to(SOLUTION).as_posix(),"sha256":hashlib.sha256(raw).hexdigest(),
                    "rows":len(value) if isinstance(value,pd.DataFrame) else None})
    return value


def number(value, percent=False):
    return "—" if pd.isna(value) else (f"{value:.1%}" if percent else f"{value:.4f}")


def save(fig,name):
    fig.savefig(OUT/f"figures/{name}.png",dpi=300,bbox_inches="tight")
    fig.savefig(OUT/f"figures/{name}.svg",bbox_inches="tight")
    plt.close(fig)


def main():
    (OUT/"figures").mkdir(parents=True,exist_ok=True)
    (OUT/"report_tables").mkdir(parents=True,exist_ok=True)
    audit=read(OUT/"beer81/summary.json")
    prediction=read(OUT/"stale/summary.json")
    loso=read(OUT/"stale/loso_metrics_overall.csv")
    pils=read(OUT/"stale/pils_diagnostic_metrics.csv")
    pils_rows=read(OUT/"stale/pils_diagnostic_predictions.csv")
    nested=read(OUT/"stale/nested_metrics.csv")
    response=read(OUT/"stale/bootstrap_response_comparison.csv")
    transfer=read(OUT/"transfer/summary.json")
    manifest=read(OUT/"transfer/run_manifest.json")
    frontier=read(OUT/"transfer/frontier.csv")
    outcomes=read(OUT/"transfer/transfer_summary.csv")
    references=read(OUT/"transfer/fixed_spectra_rechecks.csv")
    unique=read(OUT/"transfer/pool_uniqueness.csv")
    comparison=loso.merge(pils[["method","rmse","mean_bias"]].rename(columns={"rmse":"pils_rmse","mean_bias":"pils_bias"}),on="method",validate="one_to_one")
    comparison.to_csv(OUT/"report_tables/prediction_comparison.csv",index=False, lineterminator="\n")
    selected=prediction["selected_method"]
    selected_row=comparison.set_index("method").loc[selected]
    tail=comparison.set_index("method").loc["tail_ridge_aging100"]
    tail_r2=float(pils.set_index("method").loc["tail_ridge_aging100","r2"])
    old=comparison.set_index("method").loc["old_multioutput_tree"]
    nested_values=nested.set_index("method")
    selected81=float(pils_rows.loc[pils_rows.method.eq(selected)&pils_rows.beer.eq(81),"prediction"].iloc[0])
    tail81=float(pils_rows.loc[pils_rows.method.eq("tail_ridge_aging100")&pils_rows.beer.eq(81),"prediction"].iloc[0])
    retained=outcomes[outcomes.bound.eq("retain80")]
    retained.to_csv(OUT/"report_tables/retain80_all_policies.csv",index=False, lineterminator="\n")
    ref_counts=references.groupby(["arm","reference","bound"],as_index=False).agg(
        available=("source_beer","size"),passed=("fresh_consistency_passed","sum"),
        median_gain=("target_improvement","median"))
    ref_counts["sources_denominator"]=11
    ref_counts.to_csv(OUT/"report_tables/reference_counts.csv",index=False, lineterminator="\n")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
    fig,axes=plt.subplots(1,2,figsize=(12,5.4),gridspec_kw={"width_ratios":[1.1,1]})
    for row in comparison.itertuples():
        color="#267DA7" if row.method==selected else "#B98632" if row.method=="tail_ridge_aging100" else "#929AA3"
        axes[0].scatter(row.macro_style_rmse,row.pils_rmse,color=color,s=50)
    for method,label,offset in [(selected,"Selected stable ridge",(10,8)),("tail_ridge_aging100","Tail ridge (not eligible)",(-125,12)),("old_multioutput_tree","45-output reference",(8,-12))]:
        row=comparison.set_index("method").loc[method]
        axes[0].annotate(label,(row.macro_style_rmse,row.pils_rmse),xytext=offset,textcoords="offset points",fontsize=8,
                         arrowprops={"arrowstyle":"-","color":"#68717C","lw":.6})
    axes[0].set(xlabel="Non-Pils LOSO macro RMSE (selection data)",ylabel="Pils RMSE (repeatedly viewed diagnostic)",title="Prediction trade-off; lower is better",xlim=(.36,.422),ylim=(.509,.567))
    beer=pils_rows[pils_rows.beer.eq(81)].set_index("method")
    methods=["old_multioutput_tree",selected,"tail_ridge_aging100","aux4_tail_ridge10"]
    values=beer.loc[methods,"prediction"].to_numpy()
    axes[1].barh(np.arange(4),values,color=["#929AA3","#267DA7","#B98632","#C5AC79"])
    axes[1].set_yticks(np.arange(4),["45-output reference","Selected stable ridge","Tail ridge (not selected)","Auxiliary tail (not selected)"],fontsize=8)
    axes[1].axvline(audit["stale_observed"],color="#333D46",label="Observed beer81",linestyle="--")
    axes[1].axvline(audit["frozen_local_top25_stale_max"],color="#929AA3",label="Previous top25 maximum",linestyle=":")
    axes[1].set(xlabel="Standardized stale_hops",title="Beer81 underprediction",xlim=(-.3,1.85))
    axes[1].legend(frameon=False,fontsize=8,loc="lower right")
    for ax in axes:ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.subplots_adjust(wspace=.75,bottom=.2)
    fig.text(.1,.025,"Pils labels did not select the main model. The tail configuration improves this diagnostic point estimate\nbut fails training-only eligibility; no candidate is promoted using this diagnostic.",fontsize=9,color="#5C6470")
    save(fig,"beer81_and_prediction")
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.1),sharey=True)
    for ax,arm in zip(axes,["refined","fixed"]):
        part=frontier[frontier.arm.eq(arm)]
        for pool,color in COLORS.items():
            for ranking,marker in [("margin","o"),("geometry","s")]:
                rows=part[part.pool.eq(pool)&part.ranking.eq(ranking)&part.fresh_consistency_passed.gt(0)]
                ax.scatter(100*rows.median_improvement_passed,rows.fresh_consistency_passed,
                    marker=marker,facecolors=color if ranking=="margin" else "none",edgecolors=color,s=50,label=f"{pool} / {ranking}")
        point=part[part.pool.eq(manifest["default"]["pool"])&part.bound.eq("retain80")&part.ranking.eq("margin")].iloc[0]
        ax.scatter(100*point.median_improvement_passed,point.fresh_consistency_passed,marker="*",s=250,color="#27323C",zorder=5)
        ax.set(title=f"{arm}: fixed policy grid",xlabel="Median Tripel distance reduction among passing spectra (%)",ylim=(-.3,11.5),xlim=(0,100),yticks=range(0,12))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Sources passing fresh checks (out of 11)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(.5,.035))
    fig.subplots_adjust(bottom=.29,wspace=.15)
    fig.text(.12,.005,"Stars: predeclared contracted / retain80 / margin. No audit-driven default switch; finite observed grid, not a global Pareto proof.",fontsize=8,color="#5C6470")
    save(fig,"proximity_consistency_grid")
    lines=["# 第三问：beer 81误差诊断与迁移接近度保留","",
        "## 本轮结果","",
        f"beer 81身份连接正确，陈旧感目标为 **{audit['stale_observed']:.4f}**（全250款第{audit['stale_descending_rank_all250']}高）。"
        "论坛术语表也有较高的陈旧/纸板/纸质描述；没有证据据此改标签或删除该样本。",
        f"稳定性选型保留 `{selected}`，Pils诊断RMSE **{selected_row.pils_rmse:.4f}**，仍未优于本轮旧45输出对照 **{old.pils_rmse:.4f}**。"
        f"预声明尾部加权备选的Pils点误差为 **{tail.pils_rmse:.4f}**，但其非Pils留出误差变差，未获准替换主模型。",
        f"主预测模型迁移臂通过 **{transfer['arm_defaults']['refined']['fresh_consistency_passed']}/11**；"
        f"固定树方法对照的中央候选、retain80、margin通过 **{transfer['arm_defaults']['fixed']['fresh_consistency_passed']}/11**，"
        f"所选谱目标距离改善中位数 **{transfer['arm_defaults']['fixed']['median_improvement_selected']:.1%}**。"
        "两臂分别报告，没有依据审计把较高通过数的一臂改名为默认。","",
        "## beer 81为什么难预测","",
        f"1.csv与8.xlsx精确条码一致，4.csv/7.csv条码只是给定精度较低；始终以beer精确连接。"
        f"该行混合化学输入没有缺失或非法校准负值，只有{audit['sample_profile']['outside_nonpils_feature_ranges']}项略超非Pils训练范围。"
        "相对量中重复负值是附件给定表示，不能擅自当作浓度或检测下限。",
        f"旧局部模型的25近邻目标范围为 [{audit['frozen_local_top25_stale_min']:.4f}, {audit['frozen_local_top25_stale_max']:.4f}]，"
        f"实际值{audit['stale_observed']:.4f}超出该凸包，非负加权均值结构无法达到它。训练239款中仅"
        f"{audit['nonpils_stale_at_least_beer81']}款同等高尾，且均不在这25个近邻中。"
        "这说明高缺陷尾部稀疏、化学相似性与该感官目标不一致，不能证明所有化学模型都不可能预测。",
        f"新主模型对81预测 **{selected81:.4f}**，尾部加权备选为 **{tail81:.4f}**，仍明显低估。"
        "缺陷可能涉及未记录的储存、批次或品评差异，当前附件无法识别真实原因。7.csv是数值术语表而非原始评论；"
        "它仅用于语义佐证，未用作生成谱推理特征。所有Pils主指标保留11源。","",
        "## 预测的内部改善与外推代价","",
        "固定缺失指示维数、平滑岭回归、辅助感官训练监督及训练内尾部权重均为本轮预声明比较。"
        "辅助44感官只在训练阶段学习化学方向，推理只接受226维化学输入。"
        "主规则先应用完整风格留出精度与非退化门槛，再在一标准误差容忍带内按训练内响应稳定性选择。"
        "该容忍带是选择启发式，不是置信区间。",
        f"4外层分组、内层完整风格留出评价整个规则：pooled RMSE **{nested_values.loc['nested_stability_rule','rmse']:.4f}**，"
        f"同折旧45输出树 **{nested_values.loc['old_multioutput_tree','rmse']:.4f}**、固定单目标树 **{nested_values.loc['fixed_single_tree','rmse']:.4f}**。"
        "这是复用本批非Pils数据的嵌套内部点比较，不是新增外部测试。",
        "|配置|LOSO宏RMSE|LOSO pooled RMSE|Pils诊断RMSE|训练内准入|主方法|","|---|---:|---:|---:|---|---|"]
    for row in comparison.itertuples():
        lines.append(f"|{row.method}|{row.macro_style_rmse:.4f}|{row.rmse:.4f}|{row.pils_rmse:.4f}|{'是' if row.eligible_for_one_se else '否'}|{'是' if row.method==selected else ''}|")
    lines.extend(["",f"尾部备选的LOSO宏RMSE为{tail.macro_style_rmse:.4f}、pooled为{tail.rmse:.4f}，Pils R²仍为{tail_r2:.4f}，因此不能因Pils较小就后验换主方法。"
        "全部可选配置工件均保存于 `../outputs/q3_refined/stale/candidate_models/`，未选配置明确标作开发诊断。",
        "", "|60次bootstrap响应|噪声RMS|信号RMS|噪声/信号|","|---|---:|---:|---:|"])
    for row in response.itertuples():lines.append(f"|{row.method}|{row.pair_delta_noise_rms:.4f}|{row.pair_delta_signal_rms:.4f}|{row.response_noise_ratio:.4f}|")
    lines.extend(["","响应对由非Pils真实化学邻居固定产生；其中包含拟合内点，只是训练内稳定性诊断，不能证明迁移后的真实感官因果效应。",
        "","![81与预测误差](../outputs/q3_refined/figures/beer81_and_prediction.png)","",
        "## 同时保护一致性与Tripel接近幅度","",
        "新增中央候选把冻结500个目标谱经原高斯秩映射后，按0.2/0.4/0.6向Tripel中心收缩，再经经验逆CDF还原。"
        "它是搜索proposal，不当作更高质量生成模型的证据；每源投影后重新检查原完整226维几何、NN支持和十阈值保护。",
        f"1500个中央目标谱中有{transfer['contracted_targets_unique']}个唯一向量。逐源投影后去重数和重复率另存pool_uniqueness.csv；"
        "6004表示名义预算，不等于6004个独立新样本。",
        "retain80要求每源保留最早冻结默认谱至少80%的目标距离改善，并保持最少10%改善。"
        "另完整保存retain100及绝对10%/30%/50%/60%下限、两种排序、三个池的经验权衡网格。"
        "默认在审计前固定为refined/contracted_6004/retain80/margin，没有根据审计改选。",
        "筛选同时要求新旧委员会配对Q95不增加及两个中心预测不增加。旧中心确为45输出模型，但旧bootstrap仍是单目标120树；"
        "45输出中心只作单点一致检查，不声称已做该架构的bootstrap审计。新20与新种子旧协议20只审计已选谱，失败不重选。",
        "","|臂|候选池|retain80排序|选出/11|fresh通过/11|通过谱改善中位数|所选谱改善中位数|保留率中位数|",
        "|---|---|---|---:|---:|---:|---:|---:|"])
    for row in frontier[frontier.bound.eq("retain80")].itertuples():
        lines.append(f"|{row.arm}|{row.pool}|{row.ranking}|{row.selected}|{row.fresh_consistency_passed}|{number(row.median_improvement_passed,True)}|{number(row.median_improvement_selected,True)}|{number(row.median_retention_selected,True)}|")
    lines.extend(["", "同一臂、同一审计及下限才可直接比较。geometry是预声明几何优先对照，margin优先预测余量；"
        "两者的观察结果均保留，不按通过数宣布一个新默认。没有找到谱的源始终计入11分母。",
        "","|同本轮规则重检历史固定谱（retain80）|可用谱|fresh通过/11|原改善中位数|","|---|---:|---:|---:|"])
    for row in ref_counts[ref_counts.bound.eq("retain80")].itertuples():
        lines.append(f"|{row.arm}/{row.reference}|{row.available}|{row.passed}|{row.median_gain:.1%}|")
    lines.extend(["", "上一轮保守谱的8/11和19.9%使用旧规则，不能直接当成本轮统计提升率；这里按新下限和同一新审计重检。",
        "","![接近度和一致性网格](../outputs/q3_refined/figures/proximity_consistency_grid.png)","",
        "完整72个网格点见 `../outputs/q3_refined/transfer/frontier.csv`，792条策略记录、选谱、逐情景浓度及配对差均保留。"
        "这是有限候选的经验权衡，不证明连续空间的全局Pareto最优，也没有真实品评覆盖率。","",
        "## 复现与边界","",
        "Pils/81目标已经多轮查看，属于开发诊断。尾部候选的点改善、嵌套内部改善和模型检查通过均不能替代新批次外测。"
        "全部目标仍为附件标准化尺度，未恢复原0–6分。",
        "诊断详见 `q3_beer81_audit.md`，预测方法见 `q3_stale_refined.md`，独立审查见 `q3_refinement_review.md`。",
        "","```bash",".venv/bin/python solution_a/run_q3_refined.py","```",""])
    (SOLUTION/"docs/q3_refinement_results.md").write_text("\n".join(lines),encoding="utf-8", newline="\n")
    summary={"selected_method":selected,"selected_pils_rmse":float(selected_row.pils_rmse),
        "tail_candidate_pils_rmse":float(tail.pils_rmse),"tail_candidate_eligible":bool(tail.eligible_for_one_se),
        "matched_old_pils_rmse":float(old.pils_rmse),"beer81_observed":audit["stale_observed"],"beer81_selected_prediction":selected81,
        "nested_rule_rmse":float(nested_values.loc["nested_stability_rule","rmse"]),
        "default_results":transfer["default_results"],"arm_defaults":transfer["arm_defaults"],
        "candidate_rows":len(comparison),"frontier_rows":len(frontier),"source_denominator":11,"figure_pairs":2}
    (OUT/"report_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8", newline="\n")
    (OUT/"report_sources.json").write_text(json.dumps(SOURCES,ensure_ascii=False,indent=2)+"\n",encoding="utf-8", newline="\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
