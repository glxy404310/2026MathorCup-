import fs from 'node:fs/promises';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const data=JSON.parse(await fs.readFile('tmp/calculation_results/data.json','utf8'));
const wb=Workbook.create();
for(const d of data){
 const s=wb.worksheets.add(d.name);s.showGridLines=false;
 const n=d.rows.length,m=d.rows[0].length;
 const r=s.getRangeByIndexes(0,0,n,m);r.values=d.rows;
 r.format.font={name:'Arial',size:10};r.format.rowHeight=20;r.format.columnWidth=20;
 r.format.verticalAlignment='center';
 const h=s.getRangeByIndexes(0,0,1,m);h.format.fill='#273F59';h.format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};h.format.wrapText=true;h.format.rowHeight=60;
 s.freezePanes.freezeRows(1);s.freezePanes.freezeColumns(1);
 for(let c=0;c<m;c++){
  const values=d.rows.slice(1).map(x=>x[c]).filter(x=>x!==null);
  const col=s.getRangeByIndexes(1,c,n-1,1);
  if(values.length&&values.every(x=>typeof x==='number'))col.setNumberFormat(values.every(Number.isInteger)?'0':'0.000000');
  else {const max=Math.max(d.rows[0][c].length,...values.slice(0,1000).map(x=>String(x).length));s.getRangeByIndexes(0,c,n,1).format.columnWidth=Math.min(72,Math.max(20,max+2));}
 }
}
const notes=wb.worksheets.add('结果说明');notes.showGridLines=false;
const rows=[['工作表','结果含义','附件内来源'],...data.map(x=>[x.name,x.note,x.source]),['AI辅助','数据处理、程序编写与结果整理使用OpenAI Codex，GPT-6.1 Sol；详见AI工具使用详情说明。',''],['数据范围','本文件汇总论文现有结果，不是新增官方验证集的预测结果。','']];
notes.getRangeByIndexes(0,0,rows.length,3).values=rows;
notes.getRangeByIndexes(0,0,rows.length,3).format.font={name:'Arial',size:11};
notes.getRange('A1:C1').format.fill='#273F59';notes.getRange('A1:C1').format.font={bold:true,color:'#FFFFFF'};
notes.getRange('A1:A13').format.columnWidth=26;notes.getRange('B1:B13').format.columnWidth=80;notes.getRange('C1:C13').format.columnWidth=76;
notes.getRange('A1:C13').format.wrapText=true;notes.getRange('A1:C13').format.rowHeight=44;
wb.recalculate();
console.log((await wb.inspect({kind:'sheet',include:'id,name',maxChars:2000})).ndjson);
for(const d of [...data,{name:'结果说明'}]){
 const p=await wb.render({sheetName:d.name,range:d.name==='结果说明'?'A1:B6':'A1:F6',scale:1,format:'png'});
 await fs.writeFile('tmp/calculation_results/'+d.name+'.png',new Uint8Array(await p.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save('outputs/mathorcup_results/A2600730计算结果.xlsx');
console.log('exported');
