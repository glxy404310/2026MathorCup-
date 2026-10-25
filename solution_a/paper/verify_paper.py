"""Verify saved manuscript numbers, references, pagination and source integrity."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

from docx import Document
from lxml import etree
import pandas as pd
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPO=ROOT.parent
parser=argparse.ArgumentParser()
parser.add_argument('--render-dir', default=str(HERE/'qa/render'))
parser.add_argument('--write-page-map',action='store_true')
args=parser.parse_args()
render=Path(args.render_dir)
pdf=HERE/'qa/main.pdf'
docx=HERE/'啤酒风味建模提交稿.docx'
source=json.loads((HERE/'manuscript.json').read_text(encoding="utf-8"))
equations=sum(b['type']=='equation' for b in source['blocks'])
refs=json.loads((HERE/'references.json').read_text(encoding="utf-8"))

def norm(text): return re.sub(r'\s+','',text)
read=PdfReader(pdf)
texts=[page.extract_text() for page in read.pages]
compact=[norm(t) for t in texts]
assert '关键词' in texts[0] and '摘要' in compact[0]
assert '目录' in compact[1]
assert '1问题重述' in compact[2]
appendix=next(i for i,t in enumerate(compact[2:],2) if '附录A感官字段及逐维精度' in t)
body_pages=appendix-2
assert body_pages<=30,body_pages
assert len(list(render.glob('page-*.png')))==len(read.pages)
page_map={}
for mark in json.loads((HERE/'qa/bookmarks.json').read_text(encoding="utf-8")):
    matches=[i for i,t in enumerate(compact[2:],2) if norm(mark['text']) in t]
    assert matches,mark['text']
    page_map[mark['text']]=matches[0]-1
if args.write_page_map:
    (HERE/'qa/page_map.json').write_text(json.dumps(page_map,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')

doc=Document(docx)
assert len(doc.sections)==3
for section in doc.sections:
    assert abs(section.page_width.cm-21)<.01 and abs(section.page_height.cm-29.7)<.01
assert not doc.core_properties.author and not doc.core_properties.last_modified_by
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with ZipFile(docx) as z:
    tree=etree.fromstring(z.read('word/document.xml'))
    assert len(tree.xpath('//m:oMath',namespaces=ns))==equations
    assert len(tree.xpath('//w:drawing',namespaces=ns))==6
    assert not tree.xpath('//m:nary/m:e[not(*)]',namespaces=ns)
    media={hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/')}
    for image in (HERE/'figures').glob('*.png'):
        assert hashlib.sha256(image.read_bytes()).hexdigest() in media,image.name

blocks=source['blocks']
assert sum(b['type']=='equation' for b in blocks)==equations
assert sum(b['type']=='figure' for b in blocks)==6
citations=[]
for b in blocks:
    if b['type']=='reference':continue
    content=json.dumps(b,ensure_ascii=False)
    for n in re.findall(r'\[(\d+)\]',content):
        if int(n) not in citations:citations.append(int(n))
assert citations==list(range(1,len(refs)+1)),citations
assert [r['number'] for r in refs]==list(range(1,len(refs)+1))
table_lookup={b['caption']:b for b in blocks if b['type']=='table'}
q1=next(b for cap,b in table_lookup.items() if cap.startswith('表A-1'))
assert len(q1['rows'])==45
metrics=pd.read_csv(ROOT/'outputs/q1/metrics_by_target.csv').query("model=='extra_trees'")
assert q1['rows']==[[r.target,f'{r.mae:.4f}',f'{r.rmse:.4f}',f'{r.macro_r2:.4f}',f'{r.spearman:.4f}'] for r in metrics.itertuples()]
top=next(b for cap,b in table_lookup.items() if cap.startswith('表B-1'))
assert len(top['rows'])==45
evidence=pd.read_csv(HERE/'chemical_evidence_final.csv')
original=pd.read_csv(ROOT/'outputs/q2/top3_compounds.csv')
keys=['sensory_target','rank','compound']
assert evidence[keys].equals(original[keys])
dictionary=json.loads((HERE/'chemical_references_final.json').read_text(encoding="utf-8"))
assert set(dictionary['evidence_counts'])=={'direct','general','indirect','unverified'}
for r in dictionary['references']:
    assert next(x for x in refs if x['number']==r['number'])['title']==r['title']
transfer=next(b for cap,b in table_lookup.items() if cap.startswith('表C-1'))
assert len(transfer['rows'])==11
summary=pd.read_csv(ROOT/'outputs/q3_refined/transfer/transfer_summary.csv')
chosen=summary.loc[summary.arm.eq('fixed') & summary.pool.eq('contracted_6004') & summary.bound.eq('retain80') & summary.ranking.eq('margin')]
assert int(chosen.fresh_consistency_passed.sum())==10
assert chosen.loc[~chosen.fresh_consistency_passed,'source_beer'].tolist()==[96]
assert abs(chosen.target_improvement.median()-.5774027685828544)<1e-9

assert len(next(b for cap,b in table_lookup.items() if cap.startswith('表B-2'))['rows'])==135
assert len(next(b for cap,b in table_lookup.items() if cap.startswith('表C-2'))['rows'])==72
assert len(next(b for cap,b in table_lookup.items() if cap.startswith('表B-3'))['rows'])==45
supplement=json.loads((ROOT/'supplement/verification.json').read_text(encoding='utf-8'))
assert supplement['status']=='PASS'
# Independently reconcile the supplementary tables with saved result rows.
supp=ROOT/'supplement/results'
model_names={'mean':'训练折均值','extra_trees':'ExtraTrees',
             'gbr_per_target':'逐目标GBR','hgb_per_target':'逐目标HGB'}
random_metrics=pd.read_csv(supp/'q1/metrics_overall.csv').query("protocol=='random_3fold'")
group_metrics=pd.read_csv(supp/'q1/balanced_group_metrics_overall.csv')
expected_models=[]
for label,frame in [('随机三折',random_metrics),('整风格三折',group_metrics)]:
    expected_models += [[label,model_names[r.model],f'{r.rmse:.4f}',f'{r.macro_r2:.4f}'] for r in frame.itertuples()]
assert next(b for cap,b in table_lookup.items() if cap.startswith('表5-3'))['rows']==expected_models
importance=pd.read_csv(supp/'q2_permutation/importance.csv')
expected_cases=[]
for label,target,feature in [('酯味指甲油(F)—乙酸乙酯','F_esters_ethac','ethyl_acetate'),
                             ('4VG感官—4VG','X4vg','X4VG'),
                             ('麦芽总体味(F)—乙醇','F_malt_all','ethanol..v.v.'),
                             ('陈旧酒花—乙酸乙酯','stale_hops','ethyl_acetate')]:
    r=importance.loc[importance.sensory_target.eq(target)&importance.feature.eq(feature)].iloc[0]
    expected_cases.append([label,str(int(r.candidate_rank)),f'{r.delta_mse:.5f}',f'{r.fold_sd:.5f}',f'{int(r.positive_folds)}/5'])
assert next(b for cap,b in table_lookup.items() if cap.startswith('表6-3'))['rows']==expected_cases
perm_top=pd.read_csv(supp/'q2_permutation/top3.csv')
perm_comparison=pd.read_csv(supp/'q2_permutation/top3_comparison.csv').set_index('sensory_target')
expected_perm=[]
for target in perm_comparison.index:
    selected=perm_top.loc[perm_top.sensory_target.eq(target)].sort_values('candidate_rank')
    expected_perm.append([target]+[f'{r.feature}\nΔ={r.delta_mse:.3g}；{r.positive_folds}/5' for r in selected.itertuples()]
                         +[f"{perm_comparison.loc[target,'jaccard']:.2f}"])
assert next(b for cap,b in table_lookup.items() if cap.startswith('表B-3'))['rows']==expected_perm
defects=next(b for cap,b in table_lookup.items() if cap.startswith('表7-2'))['rows']
assert len(defects)==5 and [r[0] for r in defects]==['DMS','Diacetyl','4VG','trans-2-nonenal','Stale hops']
assert defects[3][2]=='附件空缺' and '代理检查' in defects[4][-1]
paired=pd.read_csv(supp/'q1_indicator_ablation/paired_metrics.csv')
cohorts={'all':'全部样本','original_missing':'原始有缺失','original_complete':'原始完整'}
expected_ablation=[[cohorts[r.cohort],str(r.n),f'{r.rmse_with_indicator:.6f}',
                    f'{r.rmse_without_indicator:.6f}',f'{r.delta_rmse:+.6f}']
                   for r in paired.loc[paired.scenario.eq('original')].itertuples()]
assert next(b for cap,b in table_lookup.items() if cap.startswith('表5-4'))['rows']==expected_ablation
module_case=pd.read_csv(supp/'reference_tables/ethyl_acetate_F_esters_ethac_by_fold.csv')
expected_module=[[str(r.fold),str(r.size),f'{r.individual_delta_mse:+.5f}',f'{r.module_delta_mse:+.5f}']
                 for r in module_case.itertuples()]
assert next(b for cap,b in table_lookup.items() if cap.startswith('表6-4'))['rows']==expected_module
assert next(b for cap,b in table_lookup.items() if cap.startswith('表B-4'))['rows']==[
    [f'第{r.fold}折\n{r.size}项',r.memberlist] for r in module_case.itertuples()]
concentrations=pd.read_csv(supp/'reference_tables/transfer_defect_concentrations.csv')
from supplement_sections import concentration_rows
assert next(b for cap,b in table_lookup.items() if cap.startswith('表C-3'))['rows']==concentration_rows(concentrations)
assert next(b for cap,b in table_lookup.items() if cap.startswith('表7-3'))['rows']==concentration_rows(
    concentrations.loc[concentrations.source_beer.isin([81,96,124])])
assert len(next(b for cap,b in table_lookup.items() if cap.startswith('表7-4'))['rows'])==4
assert any('792条策略中16条候选编号' in b.get('text','') for b in blocks)
from submission_content import source_files
inventory=json.loads((HERE/'qa/source_inventory.json').read_text(encoding='utf-8'))
assert [p.relative_to(ROOT).as_posix() for p in source_files(ROOT)]==[r['path'] for r in inventory['files']]
for row in inventory['files']:
    assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
final=PdfReader(ROOT/'submission/A2600730.pdf')
assert len(final.pages)==inventory['total_pages']
assert inventory['main_pages']==len(read.pages)
assert not final.metadata.get('/Author')
for row in inventory['files']:
    recovered=[]
    def source_text(text, cm, tm, font, size):
        if abs(size-9)<.01 and tm[4]>=97.8:
            recovered.append(text)
    for page in final.pages[row['first_pdf_page']-1:row['last_pdf_page']]:
        page.extract_text(visitor_text=source_text)
    expected=(ROOT/row['path']).read_text(encoding='utf-8-sig')
    assert norm(''.join(recovered))==norm(expected), ('printed_source_text',row['path'])
assert len(final.pages[1]['/Annots'])==len(inventory['contents_links'])
for annot,link in zip(final.pages[1]['/Annots'],inventory['contents_links']):
    dest=annot.get_object()['/Dest']
    assert final.get_page_number(dest[0].get_object())==link['target_pdf_page']-1

# This is read-only: never rerun experiment writers for paper formatting.
sys.path.insert(0,str(ROOT))
from run_q3_refined import preserve_refined
preserve_refined()
raw=REPO/'2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题'/'2026年MathorCup大数据挑战赛-赛道A初赛'/'附件'
for r in pd.read_csv(ROOT/'outputs/audit/source_files.csv').itertuples():
    assert hashlib.sha256((raw/r.source).read_bytes()).hexdigest()==r.sha256,r.source

report={'status':'PASS','main_pdf_pages':len(read.pages),'final_pdf_pages':len(final.pages),'printed_source_files':len(inventory['files']),'printed_source_lines':inventory['source_lines'],'front_matter_pages':2,'body_including_references_pages':body_pages,
        'appendix_pages':len(read.pages)-appendix,'native_equations':equations,'figures':6,'references':len(refs),
        'q1_target_rows':45,'top3_positions':135,'transfer_sources':11,'raw_source_hashes_unchanged':9,
        'frozen_refined_protection_passed':True,'source_refs_reconciled':True,
        'supplementary_tables_reconciled':True,
        'reference_revision_tables_reconciled':True,'concentration_checks':44,'module_case_folds':5,
        'visual_review':'Recorded separately in qa/visual_review.json after inspection of page PNGs.',
        'artifact_sha256':{'docx':hashlib.sha256(docx.read_bytes()).hexdigest(),'main_pdf':hashlib.sha256(pdf.read_bytes()).hexdigest(),'final_pdf':hashlib.sha256((ROOT/'submission/A2600730.pdf').read_bytes()).hexdigest()}}
(HERE/'qa/paper_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))

