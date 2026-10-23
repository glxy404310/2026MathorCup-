"""Package source programs and inspectable paper results without raw attachments."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
entries={}
for path in sorted((ROOT/'src').glob('*.py')):
    entries[f'solution_a/src/{path.name}']=path
for path in sorted(ROOT.glob('run*.py')):
    entries[f'solution_a/{path.name}']=path
for name in ['requirements-lock.txt','final_model_selection.json']:
    entries[f'solution_a/{name}']=ROOT/name
entries['solution_a/docs/final_model_selection.md']=ROOT/'docs/final_model_selection.md'
entries['solution_a/docs/use_statement.md']=ROOT/'docs/use_statement.md'
for name in ['make_core.py','make_figures.py','build_paper.py','native_math.py',
             'figure_manifest.json','references.json','chemical_evidence_final.csv',
             'chemical_references_final.json','chemical_references.json','chemical_evidence.csv','manuscript.json']:
    entries[f'solution_a/paper/{name}']=HERE/name
for path in sorted((HERE/'sections').glob('*.json')):
    entries[f'solution_a/paper/sections/{path.name}']=path
for path in sorted((HERE/'figures').iterdir()):
    if path.suffix in ['.png','.svg']:
        entries[f'solution_a/paper/figures/{path.name}']=path
for path in sorted((ROOT/'outputs').rglob('*')):
    if path.is_file() and path.suffix in ['.csv','.json']:
        entries['paper_results/'+str(path.relative_to(ROOT/'outputs'))]=path

guide='''啤酒风味建模论文初稿电子附录

本包与论文初稿配套，表D-1的所有路径均相对于本包根目录。

1. solution_a/src/ 保存全部实验源程序；solution_a/run_all.py 为统一复现入口。
2. solution_a/requirements-lock.txt 固定本次计算依赖，建议Python 3.13独立环境。
3. paper_results/ 保存本次论文引用及历轮比较的CSV和JSON；它是结果备份，不是运行目录。
4. solution_a/paper/ 保存正文结构化源稿、图表脚本、原生公式构建、文献及逐行化学证据。
   chemical_evidence_final.csv的source_ref及chemical_references_final.json采用论文最终参考编号。
5. 原始赛题附件不在本包内。复现时将原题文件夹与solution_a并列，保持原始目录及文件名。
   在新目录中安装依赖后运行 python solution_a/run_all.py，将重建solution_a/outputs/。
   不要把paper_results复制为outputs后直接运行，以免不完整的模型工件触发冻结校验。
6. 模型工件可由源程序重新训练。此包不包含较大的joblib权重及候选逐项压缩缓存；
   完整候选搜索和核验由源程序重建，逐源选谱、72策略点及十情景检查已保存。
7. 论文再生另需python-docx和lxml，并需保留赛题论文模板；图表需可用中文字体。

初稿的实质边界：未恢复原始0—6评分；部分化学关联没有匹配的直接机制依据；
Stale hops与迁移通过仅为模型检查，尚无迁移后实际品评或生产验证。
'''
manifest={'scope':'Source programs, paper evidence and saved numerical results; raw attachments excluded.',
          'program_files':sum(name.endswith('.py') and '/paper/' not in name for name in entries),
          'files':[{'path':name,'bytes':p.stat().st_size,
                    'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                   for name,p in entries.items()]}
out=HERE/'论文初稿支撑材料.zip'
with ZipFile(out,'w',compression=ZIP_DEFLATED,compresslevel=7) as archive:
    archive.writestr('README.txt',guide)
    archive.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for name,path in entries.items(): archive.write(path,name)
with ZipFile(out) as archive:
    assert archive.testzip() is None
    for folder in ['paper_results/q1/','paper_results/q1_robust/','paper_results/q2/',
                   'paper_results/q3_optimized/','paper_results/q3_refined/']:
        assert any(name.startswith(folder) for name in archive.namelist()),folder
    for row in manifest['files']:
        assert hashlib.sha256(archive.read(row['path'])).hexdigest()==row['sha256']
(HERE/'qa/support_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(f'{out.name}: {len(entries)+2} files, {manifest["program_files"]} experiment programs, {out.stat().st_size/2**20:.2f} MiB')

