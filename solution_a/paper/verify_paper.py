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
parser.add_argument('--render-dir',required=True)
parser.add_argument('--write-page-map',action='store_true')
args=parser.parse_args()
render=Path(args.render_dir)
pdf=next(render.glob('*.pdf'))
docx=HERE/'啤酒风味建模论文初稿.docx'
source=json.loads((HERE/'manuscript.json').read_text())
refs=json.loads((HERE/'references.json').read_text())

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
for mark in json.loads((HERE/'qa/bookmarks.json').read_text()):
    matches=[i for i,t in enumerate(compact[2:],2) if norm(mark['text']) in t]
    assert matches,mark['text']
    page_map[mark['text']]=matches[0]-1
if args.write_page_map:
    (HERE/'qa/page_map.json').write_text(json.dumps(page_map,ensure_ascii=False,indent=2)+'\n')

doc=Document(docx)
assert len(doc.sections)==3
for section in doc.sections:
    assert abs(section.page_width.cm-21)<.01 and abs(section.page_height.cm-29.7)<.01
assert not doc.core_properties.author and not doc.core_properties.last_modified_by
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with ZipFile(docx) as z:
    tree=etree.fromstring(z.read('word/document.xml'))
    assert len(tree.xpath('//m:oMath',namespaces=ns))==30
    assert len(tree.xpath('//w:drawing',namespaces=ns))==6
    assert not tree.xpath('//m:nary/m:e[not(*)]',namespaces=ns)
    media={hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/')}
    for image in (HERE/'figures').glob('*.png'):
        assert hashlib.sha256(image.read_bytes()).hexdigest() in media,image.name

blocks=source['blocks']
assert sum(b['type']=='equation' for b in blocks)==30
assert sum(b['type']=='figure' for b in blocks)==6
citations=[]
for b in blocks:
    if b['type']=='reference':continue
    content=json.dumps(b,ensure_ascii=False)
    for n in re.findall(r'\[(\d+)\]',content):
        if int(n) not in citations:citations.append(int(n))
assert citations==list(range(1,9)),citations
assert [r['number'] for r in refs]==list(range(1,9))
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
dictionary=json.loads((HERE/'chemical_references_final.json').read_text())
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

# This is read-only: never rerun experiment writers for paper formatting.
sys.path.insert(0,str(ROOT))
from run_q3_refined import preserve_refined
preserve_refined()
raw=REPO/'2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题'/'2026年MathorCup大数据挑战赛-赛道A初赛'/'附件'
for r in pd.read_csv(ROOT/'outputs/audit/source_files.csv').itertuples():
    assert hashlib.sha256((raw/r.source).read_bytes()).hexdigest()==r.sha256,r.source

report={'status':'PASS','pages':len(read.pages),'front_matter_pages':2,'body_including_references_pages':body_pages,
        'appendix_pages':len(read.pages)-appendix,'native_equations':30,'figures':6,'references':8,
        'q1_target_rows':45,'top3_positions':135,'transfer_sources':11,'raw_source_hashes_unchanged':9,
        'frozen_historical_files_unchanged':196,'source_refs_reconciled':True,
        'visual_review':'Recorded separately in qa/visual_review.json after inspection of page PNGs.',
        'artifact_sha256':{'docx':hashlib.sha256(docx.read_bytes()).hexdigest(),'pdf':hashlib.sha256(pdf.read_bytes()).hexdigest()}}
(HERE/'qa/paper_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))

