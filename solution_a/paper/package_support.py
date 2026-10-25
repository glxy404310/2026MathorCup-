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
for pattern in ['*.json', '*.csv', 'sections/*.json', 'figures/*.png', 'figures/*.svg']:
    for path in HERE.glob(pattern):
        if path.name != 'reference_note.json':
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

overrides = {}
provenance_name = 'paper_results/q3_refined/beer81/source_manifest.json'
original_provenance = entries[provenance_name].read_bytes()
provenance = json.loads(original_provenance)
for row in provenance:
    original = Path(row['path'])
    if not original.is_absolute():
        original = ROOT.parent / original
    assert hashlib.sha256(original.read_bytes()).hexdigest() == row['sha256']
    row['path'] = original.relative_to(ROOT.parent).as_posix()
    assert hashlib.sha256((ROOT.parent / row['path']).read_bytes()).hexdigest() == row['sha256']
overrides[provenance_name] = (json.dumps(provenance, ensure_ascii=False, indent=2) + '\n').encode('utf-8')

guide = '''A题论文配套程序与数值结果

1. PDF附录E已印出建模计算、结果绘图及数值检验源程序；本包保留可直接运行的文件及缩进。
2. solution_a/src/为主分析程序，run_all.py为统一入口，supplement/为补充分析，paper/保存结果绘图程序、结构化论文和图表。
3. paper_results/及supplement_results/保存论文引用结果，不是运行输出目录。本包不含原始赛题附件、模型权重及大型候选缓存。
4. 原始赛题目录与solution_a并列，保持官方目录名。计算使用Python 3.13及requirements-lock.txt。
   Windows PowerShell在解压根目录运行：
   py -3.13 -m venv .venv
   .\\.venv\\Scripts\\python.exe -m pip install -r solution_a/requirements-lock.txt
   .\\.venv\\Scripts\\python.exe -X utf8 solution_a/run_all.py --rebuild
   全部模型、候选缓存、outputs与supplement/results由源码重建。--rebuild拒绝覆盖非空输出目录。
5. 对重建完成的完整目录，去掉--rebuild即可只读验证。报告写入solution_a/review/verification/。
6. 本包中beer81/source_manifest.json的本机绝对路径已转换为仓库相对路径，所指文件的SHA-256保持不变；原始冻结记录和数值结果未改写。
7. MANIFEST.json记录包内文件SHA-256。verification内为原始冻结结果的只读验证和独立重建检查记录。
8. 原评分量尺不能从现有附件唯一恢复，文中标准化预测不冒充0—6评分；化学相关不等于感官因果，迁移结果尚未经实际品评。
'''
manifest = {'scope': 'Modelling, numerical-check and result-plot source with saved numerical evidence; original attachments excluded.',
            'program_files': len(source_files(ROOT)),
            'path_normalization': {'path': provenance_name, 'fields': ['path'], 'referenced_file_checksums_unchanged': True},
            'files': [{'path': name, 'bytes': len(overrides[name]) if name in overrides else p.stat().st_size,
                       'sha256': hashlib.sha256(overrides[name] if name in overrides else p.read_bytes()).hexdigest()}
                      for name, p in sorted(entries.items())]}
out = OUT / 'A2600730附件.zip'
with ZipFile(out, 'w', compression=ZIP_DEFLATED, compresslevel=7) as archive:
    archive.writestr('README.txt', guide)
    archive.writestr('MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    for name, path in sorted(entries.items()):
        if name in overrides:
            archive.writestr(name, overrides[name])
        else:
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
