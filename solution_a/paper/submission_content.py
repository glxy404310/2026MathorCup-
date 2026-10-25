"""Tables and reproduction instructions included in the submitted paper."""
from pathlib import Path
import re

import pandas as pd


def source_files(root):
    """Use one inventory for the printed appendix and the source archive."""
    paths = list(root.glob("*.py"))
    paths += list((root / "src").glob("*.py"))
    paths += list((root / "paper").glob("*.py"))
    paths += list((root / "paper").glob("*.ps1"))
    paths += list((root / "tests").glob("*.py"))
    paths += list((root / "supplement").glob("*.py"))
    return sorted(set(paths), key=lambda p: p.relative_to(root).as_posix())


def add_submission_appendices(blocks, root, paper, mapping):
    def paragraph(text):
        return {"type": "paragraph", "text": text}

    def heading(text, level=1):
        return {"type": "heading", "level": level, "text": text}

    def table(caption, headers, rows, widths):
        return {"type": "table", "caption": caption, "headers": headers,
                "rows": rows, "widths": widths, "compact": True}

    evidence = pd.read_csv(paper / "chemical_evidence_final.csv").fillna("")
    labels = {"direct": "直接气味依据", "general": "一般背景",
              "indirect": "间接关联", "unverified": "尚未证实"}
    rows = []
    for r in evidence.to_dict("records"):
        chemical = str(r.get("chemical_name_display") or r["chemical_name"])
        source = str(r.get("origin_display") or r["origin_paper"])
        unit = str(r.get("unit_display") or r["unit"] or "相对量")
        threshold = str(r.get("threshold_display") or "附件未给匹配阈值")
        claim = str(r.get("evidence_claim") or r.get("evidence_support_note", ""))
        boundary = str(r.get("evidence_boundary") or "相关系数不构成因果证据。")
        refs = str(r.get("source_ref", ""))
        rows.append([f"{r['sensory_target']}\n第{int(r['rank'])}项",
                     f"{chemical}\n{r['compound']}",
                     f"{source}；{unit}\n{threshold}",
                     f"{labels[r['evidence_category']]} {refs}\n{claim}\n{boundary}"])
    extra = [paragraph("表B-2给出每个候选的化学名称、附件来源、量纲及文献支持范围。"
                       "阈值只用于对应的校准浓度；空缺阈值不补造。一般背景与间接关联均不视为目标感官机制已经证实。"),
             table("表B-2 完整候选的化学信息及证据", ["感官与名次", "化合物", "来源与量纲", "依据及适用范围"],
                   rows, [2.5, 3.8, 3.5, 6.2])]
    idx = next(i for i, b in enumerate(blocks) if b.get("text", "").startswith("附录C"))
    blocks[idx:idx] = extra
    from supplement_sections import permutation_appendix
    idx = next(i for i, b in enumerate(blocks) if b.get("text", "").startswith("附录C"))
    blocks[idx:idx] = permutation_appendix(root)

    frontier = pd.read_csv(root / "outputs/q3_refined/transfer/frontier.csv")
    pool = {"legacy_2004": "原2004", "legacy_6004": "原6004", "contracted_6004": "收缩6004"}
    bound = {"retain80": "保留80%", "retain100": "保留100%", "absolute10": "≥10%",
             "absolute30": "≥30%", "absolute50": "≥50%", "absolute60": "≥60%"}
    gridrows = []
    for r in frontier.itertuples():
        value = "—" if pd.isna(r.median_improvement_selected) else f"{r.median_improvement_selected:.2%}"
        gridrows.append(["固定树" if r.arm == "fixed" else "岭回归", pool.get(r.pool, r.pool),
                         bound.get(r.bound, r.bound), "余量" if r.ranking == "margin" else "几何",
                         f"{r.selected}/11", f"{r.fresh_consistency_passed}/11", value])
    blocks += [table("表C-2 全部72组迁移策略结果", ["模型", "候选集", "接近下限", "排序", "选出", "通过", "改善中位数"],
                     gridrows, [1.6, 2.2, 2.1, 1.4, 1.5, 1.5, 3.7]),
               paragraph("改善中位数按全部选中谱计算，包括复核未通过者；无候选时记为横线。"
                         "每组分母均为11，不以选出的条数代替源样本总数。逐化合物十情景检查及全部数值文件随支撑材料提交。")]
    from supplement_sections import concentration_rows, CONCENTRATION_HEADERS, CONCENTRATION_WIDTHS
    concentration = pd.read_csv(root / "supplement/results/reference_tables/transfer_defect_concentrations.csv")
    blocks += [paragraph("表C-3列出余量优先主方案全部11款源的44项浓度检查。每行上界为十情景中最严值，"
                         "剩余裕量为上界减迁移浓度，单位与该行校准浓度相同；源浓度超过阈值时，上界可以高于感知阈值。"
                         "化学检查不代替表C-1的Stale hops模型复核。"),
               table("表C-3 全部源的迁移前后校准浓度与保护裕量", CONCENTRATION_HEADERS,
                     concentration_rows(concentration), CONCENTRATION_WIDTHS)]
    blocks += [heading("附录D 软件环境与运行方法"),
               paragraph("模型原计算环境为Python 3.13，精确依赖列于表D-1。程序采用UTF-8文本、"
                         "正斜杠相对路径和固定换行规则，支持Windows。原始赛题附件与solution_a文件夹并列，"
                         "保持官方目录名；数据不重复打包。所有计算源码及论文构建程序全文列于附录E。"),
               table("表D-1 计算依赖版本", ["软件包", "版本"],
                     [line.split("==", 1) for line in (root / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()
                      if "==" in line], [8, 8]),
               paragraph("在Windows PowerShell中进入解压后的根目录，建立独立环境并安装依赖："),
               {"type": "code", "text": "py -3.13 -m venv .venv\n.\\.venv\\Scripts\\python.exe -m pip install -r solution_a/requirements-lock.txt"},
               paragraph("对包含完整outputs的仓库，默认只核查现有结果，不重新训练或改写冻结记录："),
               {"type": "code", "text": ".\\.venv\\Scripts\\python.exe -X utf8 solution_a/run_all.py"},
               paragraph("支撑材料中的paper_results和supplement_results为引用结果副本，不作为训练输出目录。首次完整重建在新解压目录执行："),
               {"type": "code", "text": ".\\.venv\\Scripts\\python.exe -X utf8 solution_a/run_all.py --rebuild"},
               paragraph("重建须保留全部原始附件，依次执行数据核验、预测、关联分析、风格生成、迁移及补充验证。"
                         "程序对非空历史或补充输出拒绝覆盖；重新试验应另建副本。补充验证源码与结果位于supplement目录，run_supplement.py可单独核查。计算耗时随硬件变化。"
                         "历史独立重建中，主方案关键比较一致，全探索网格仍有候选与计数差异，具体范围见第8.2节及核验记录。"),
               paragraph("论文构建另需python-docx、lxml、reportlab和matplotlib。正文可经Windows的Word兼容COM接口，"
                         "或明确配置中文及数学字体的LibreOffice导出，再拼接文字型源码附录。"
                         "本次修订使用macOS下的独立LibreOffice运行时，保持A4、12pt正文及原生公式；"
                         "源程序附录支持显式指定等宽、中文与数学字体。完整依赖、命令及文件校验方法见支撑材料README.txt。"),
               heading("附录E 完整计算机源程序"),
               paragraph("本附录按文件给出计算、校验及论文构建所用源程序。源代码以可选取的文本排印，"
                         "左侧为原文件行号；长行折行处不另编号。运行时使用支撑包中的同名文件，"
                         "避免PDF阅读器复制时丢失缩进。文件清单、源码全文和支撑包采用同一份程序目录。")]
    files = source_files(root)
    blocks.append(table("表E-1 源程序目录", ["文件（相对solution_a）", "行数"],
                        [[p.relative_to(root).as_posix(), str(len(p.read_text(encoding="utf-8-sig").splitlines()))]
                         for p in files], [13.5, 2.5]))
    return blocks
