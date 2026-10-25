"""Package complete source programs and saved numerical evidence."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from submission_content import source_files

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / 'submission'
OUT.mkdir(exist_ok=True)
entries = {}
for path in source_files(ROOT):
    entries['solution_a/' + path.relative_to(ROOT).as_posix()] = path
for name in ['requirements-lock.txt', 'final_model_selection.json', 'README.md']:
    entries['solution_a/' + name] = ROOT / name
for path in (ROOT / 'docs').glob('*.md'):
    entries['solution_a/docs/' + path.name] = path
for pattern in ['*.json', '*.csv', '*.txt', '*.conf', 'sections/*.json', 'figures/*.png', 'figures/*.svg']:
    for path in HERE.glob(pattern):
        entries['solution_a/paper/' + path.relative_to(HERE).as_posix()] = path
for path in sorted((ROOT / 'outputs').rglob('*')):
    if path.is_file() and path.suffix in ['.csv', '.json']:
        entries['paper_results/' + path.relative_to(ROOT / 'outputs').as_posix()] = path
for path in sorted((ROOT / 'supplement/results').rglob('*')):
    if path.is_file() and path.suffix in ['.csv', '.json', '.npz']:
        entries['supplement_results/' + path.relative_to(ROOT / 'supplement/results').as_posix()] = path
entries['solution_a/supplement/README.txt'] = ROOT / 'supplement/README.txt'
entries['verification/supplement.json'] = ROOT / 'supplement/verification.json'
attributes = ROOT.parent / '.gitattributes'
if attributes.exists():
    entries['.gitattributes'] = attributes
for path in (ROOT / 'review/verification').rglob('*.json'):
    entries['verification/' + path.relative_to(ROOT / 'review/verification').as_posix()] = path
for name in ['source_inventory.json', 'paper_verification.json', 'visual_review.json']:
    path = HERE / 'qa' / name
    if path.exists():
        entries['verification/' + name] = path

guide = '''A题论文配套程序与数值结果

1. PDF附录E已印出全部源程序；本包保留可直接运行的文件及缩进。
2. solution_a/src/为主分析程序，run_all.py为统一入口，supplement/为补充分析，paper/为论文结构化源稿、表图和构建检查程序。
3. paper_results/及supplement_results/保存论文引用结果，不是运行输出目录。本包不含原始赛题附件、模型权重及大型候选缓存。
4. 原始赛题目录与solution_a并列，保持官方目录名。计算使用Python 3.13及requirements-lock.txt。
   Windows PowerShell在解压根目录运行：
   py -3.13 -m venv .venv
   .\\.venv\\Scripts\\python.exe -m pip install -r solution_a/requirements-lock.txt
   .\\.venv\\Scripts\\python.exe -X utf8 solution_a/run_all.py --rebuild
   全部模型、候选缓存、outputs与supplement/results由源码重建。--rebuild拒绝覆盖非空输出目录。
5. 对重建完成的完整目录，去掉--rebuild即可只读验证。报告写入solution_a/review/verification/。
6. 重建论文需原赛题Word模板、中文和数学字体；以下为Windows的Word兼容COM接口路线：
   python -m pip install -r solution_a/paper/requirements-paper.txt
   python solution_a/paper/make_figures.py
   python solution_a/paper/build_paper.py
   powershell -ExecutionPolicy Bypass -File solution_a/paper/export_pdf.ps1
   python solution_a/paper/source_appendix.py
   python solution_a/paper/package_support.py
   本次修订用独立LibreOffice运行时导出正文，macOS字体替换配置为solution_a/paper/fonts_macos.conf；
   导出前设置FONTCONFIG_FILE为该配置的绝对路径，使用明确的LibreOffice可执行文件和独立用户配置目录。
   正文PDF保存为paper/qa/main.pdf。源码附录可用--mono-font、--cjk-font、--math-font、
   --cjk-index、--math-index指定本机字体；本轮采用Monaco、Songti SC Regular（TTC索引6）及DejaVu Sans。
   paper脚本读取outputs；欲保留论文原始数值，可在单独的论文构建副本中将paper_results复制为solution_a/outputs，
   将supplement_results复制为solution_a/supplement/results，
   仅运行上述paper构建脚本，不对该不完整副本运行run_all.py。
   verify_paper.py须在完整重建目录运行，并先用Poppler的pdftoppm命令把qa/main.pdf渲染为qa/render/page-*.png。
7. MANIFEST.json记录包内文件SHA-256。verification内为原始冻结结果的只读验证和独立重建检查记录。
8. 原评分量尺不能从现有附件唯一恢复，文中标准化预测不冒充0—6评分；化学相关不等于感官因果，迁移结果尚未经实际品评。
'''
manifest = {'scope': 'Complete printed source, paper evidence and saved numerical results; original attachments excluded.',
            'program_files': len(source_files(ROOT)),
            'files': [{'path': name, 'bytes': p.stat().st_size,
                       'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for name, p in sorted(entries.items())]}
out = OUT / 'A2600730附件.zip'
with ZipFile(out, 'w', compression=ZIP_DEFLATED, compresslevel=7) as archive:
    archive.writestr('README.txt', guide)
    archive.writestr('MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    for name, path in sorted(entries.items()):
        archive.write(path, name)
with ZipFile(out) as archive:
    assert archive.testzip() is None
    for folder in ['q1', 'q1_robust', 'q2', 'q3_optimized', 'q3_refined']:
        assert any(name.startswith('paper_results/' + folder + '/') for name in archive.namelist()), folder
    for row in manifest['files']:
        assert hashlib.sha256(archive.read(row['path'])).hexdigest() == row['sha256']
    inventory = json.loads((HERE / 'qa/source_inventory.json').read_text(encoding='utf-8'))
    for row in inventory['files']:
        assert hashlib.sha256(archive.read('solution_a/' + row['path'])).hexdigest() == row['sha256'], row['path']
(HERE / 'qa/support_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'{out.name}: {len(entries) + 2} files, {manifest["program_files"]} programs, {out.stat().st_size / 2**20:.2f} MiB')
