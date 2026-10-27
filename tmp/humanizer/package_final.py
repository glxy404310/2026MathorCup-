from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from pypdf import PdfReader,PdfWriter
from lxml import etree
import json,hashlib,shutil,re
root=Path.cwd(); tmp=root/'tmp/humanizer'; sub=root/'solution_a/submission'
hashb=lambda b:hashlib.sha256(b).hexdigest()
# Preserve user Word and scientific objects.
assert (sub/'AI工具使用详情说明.docx').read_bytes()==(tmp/'before/AI工具使用详情说明.docx').read_bytes()
a=ZipFile(tmp/'before/啤酒风味建模提交稿.docx'); b=ZipFile(root/'solution_a/paper/啤酒风味建模提交稿.docx')
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
x,y=[etree.fromstring(z.read('word/document.xml')) for z in (a,b)]
for xp in ['//w:tbl','//m:oMath']:
 assert [etree.tostring(e) for e in x.xpath(xp,namespaces=ns)]==[etree.tostring(e) for e in y.xpath(xp,namespaces=ns)]
for n in a.namelist():
 if n.startswith('word/media/'):assert a.read(n)==b.read(n)
main=PdfReader(tmp/'final_render/啤酒风味建模提交稿.pdf');old=PdfReader(tmp/'before/A2600730.pdf')
assert len(main.pages)==57 and len(old.pages)==354
w=PdfWriter();w.append(main);w.append(old,pages=(57,354),import_outline=False)
w.write(sub/'A2600730.pdf')
shutil.copy2(tmp/'ai_render/AI工具使用详情说明.pdf',sub/'AI工具使用详情说明.pdf')
oldzip=ZipFile(tmp/'before/A2600730附件.zip');entries={n:oldzip.read(n) for n in oldzip.namelist()}
log=json.loads((tmp/'edit_log.json').read_text());changed=[]
for n in log['support_changed']+log['json_synchronized']+['solution_a/docs/use_statement.md']:
 if n in entries:
  entries[n]=(root/n).read_bytes();changed.append(n)
readme=entries['README.txt'].decode().replace('文中标准化预测不冒充0—6评分','文中标准化预测不等同于原始0—6评分')
readme+='9. AI工具使用详情说明.pdf为本队提交的工具使用说明。\n'
entries['README.txt']=readme.encode()
entries['AI工具使用详情说明.pdf']=(sub/'AI工具使用详情说明.pdf').read_bytes()
# Every original code file and saved numerical result remains byte-identical.
protected=[n for n in entries if n.endswith('.py') or n.startswith(('paper_results/','supplement_results/'))]
for n in protected: assert entries[n]==oldzip.read(n)
report={'date':'2026-10-27','main_prose_revisions':len(log['main_changes']),'support_prose_files_revised':len(log['support_changed']),'tables_equations_images_unchanged':True,'original_programs_and_saved_results_unchanged':True,'ai_word_preserved':True,'paper_pdf_pages':354,'main_before_appendices_numbered_pages':24,'updated_entries':changed}
entries['verification/editorial_review_20261027.json']=json.dumps(report,ensure_ascii=False,indent=2).encode()
manifest=json.loads(entries['MANIFEST.json']);manifest['files']=[{'path':n,'bytes':len(v),'sha256':hashb(v)} for n,v in sorted(entries.items()) if n!='MANIFEST.json']
entries['MANIFEST.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
with ZipFile(sub/'A2600730附件.zip','w',ZIP_DEFLATED) as z:
 for n,v in entries.items():z.writestr(n,v)
with ZipFile(sub/'A2600730附件.zip') as z:
 assert z.testzip() is None
 for row in manifest['files']:assert hashb(z.read(row['path']))==row['sha256']
final=root/'solution_a/最终提交材料_20261027';final.mkdir(exist_ok=True)
for n in ['A2600730.pdf','A2600730承诺书.pdf','A2600730附件.zip']:shutil.copy2(sub/n,final/n)
with ZipFile(final.with_suffix('.zip'),'w',ZIP_DEFLATED) as z:
 for p in sorted(final.iterdir()):z.write(p,p.name)
with ZipFile(final.with_suffix('.zip')) as z:assert z.testzip() is None
# Recheck prose markers on final sources, excluding historical records and code.
pattern=re.compile('强感官保证|拼接赢家|不补造|最漂亮|值得注意的是|综上所述|赋能|至关重要|不冒充')
hits=[]
for n,v in entries.items():
 if n.endswith(('.md','.txt')) and not n.startswith('verification/'):
  for i,line in enumerate(v.decode('utf-8',errors='replace').splitlines(),1):
   if pattern.search(line):hits.append({'file':n,'line':i,'text':line})
report['second_pass_marker_hits']=hits
report['final_package']=str(final.with_suffix('.zip'));report['package_bytes']=final.with_suffix('.zip').stat().st_size
(tmp/'final_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
