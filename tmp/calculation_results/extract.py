from zipfile import ZipFile
import pandas as pd, json
from pathlib import Path
z=ZipFile('solution_a/submission/A2600730附件.zip')
spec=[
('Q1全样本拟合值','q1/fitted_standardized_NOT_validation.csv','250款啤酒的全样本拟合值，非独立验证结果。标准化尺度，不能解释为原始0—6分。'),
('Q1折外预测','q1/oof_extra_trees.csv','交叉验证折外预测，标准化尺度。'),
('Q1评价指标','q1/metrics_overall.csv','模型整体交叉验证评价指标。'),
('Q2关键化合物','q2/top3_compounds.csv','45项感官属性的Top-3关联化合物及稳定性。相关性不代表因果关系。'),
('Q3生成Tripel','q3_optimized/generation/generated_Tripel.csv','生成化学谱。校准量与相对量的单位和定义见论文。'),
('Q3生成Kriek','q3_optimized/generation/generated_Kriek.csv','生成化学谱。校准量与相对量的单位和定义见论文。'),
('Q3生成Lambic','q3_optimized/generation/generated_Lambic.csv','生成化学谱。校准量与相对量的单位和定义见论文。'),
('Q3生成感官画像','q3_optimized/generated_sensory_profiles.csv','生成样本的模型预测感官画像，标准化尺度。'),
('Q3主方案迁移结果','q3_refined/transfer/transfer_summary.csv','论文主方案：固定树、中央6004候选池、retain80、余量优先。含未通过的样本。'),
('Q3主方案化学谱','q3_refined/transfer/selected_spectra.csv','仅保留主方案对应的选中化学谱；是否通过全部检查须结合迁移结果表。'),
]
summary=pd.read_csv(z.open('paper_results/q3_refined/transfer/transfer_summary.csv'))
main=summary[(summary.arm=='fixed')&(summary.pool=='contracted_6004')&(summary.bound=='retain80')&(summary.ranking=='margin')]
assert len(main)==11
out=[]
for name,path,note in spec:
 d=pd.read_csv(z.open('paper_results/'+path),dtype={'beer_id':str})
 if name=='Q3主方案迁移结果':d=main
 if name=='Q3主方案化学谱':d=d[d.transfer_id.isin(main.transfer_id)]
 if name=='Q3生成感官画像':d=d[d.target_style.isin(['Tripel','Kriek','Lambic'])]
 vals=json.loads(d.to_json(orient='values',force_ascii=False,double_precision=15))
 out.append(dict(name=name,source='paper_results/'+path,note=note,rows=[d.columns.tolist()]+vals))
Path('tmp/calculation_results/data.json').write_text(json.dumps(out,ensure_ascii=False))
print([(s['name'],len(s['rows'])-1,len(s['rows'][0])) for s in out])
