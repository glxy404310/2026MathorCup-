"""Print every source line and combine it with the main paper PDF."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Link
from pypdf.generic import NameObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

from submission_content import source_files

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "submission"
QA = HERE / "qa"
OUT.mkdir(exist_ok=True)
QA.mkdir(exist_ok=True)


def build(mono_font="C:/Windows/Fonts/consola.ttf",
          cjk_font="C:/Windows/Fonts/simsun.ttc",
          math_font="C:/Windows/Fonts/cambria.ttc", cjk_index=0, math_index=1):
    main = PdfReader(QA / "main.pdf")
    pdfmetrics.registerFont(TTFont("CodeMono", mono_font))
    pdfmetrics.registerFont(TTFont("CodeCJK", cjk_font, subfontIndex=cjk_index))
    pdfmetrics.registerFont(TTFont("CodeMath", math_font, subfontIndex=math_index))
    width, height = A4
    left, right, top, bottom = 25 * mm, width - 25 * mm, height - 20 * mm, 22 * mm
    code_x = left + 27
    font_size, leading = 9, 11.5
    path = QA / "source.pdf"
    c = canvas.Canvas(str(path), pagesize=A4, pageCompression=1)
    c.setTitle("完整计算机源程序")
    c.setAuthor("")
    page = 0
    y = top
    records = []
    geometry = []

    def font(char):
        for face in ("CodeMono", "CodeCJK", "CodeMath"):
            if ord(char) in pdfmetrics.getFont(face).face.charToGlyph:
                return face
        raise ValueError(f"No embedded font covers U+{ord(char):04X}")

    def draw(text, x, baseline, size=font_size):
        runs = []
        for char in text:
            if runs and runs[-1][0] == font(char):
                runs[-1][1] += char
            else:
                runs.append([font(char), char])
        for face, run in runs:
            c.setFont(face, size)
            c.drawString(x, baseline, run)
            x += pdfmetrics.stringWidth(run, face, size)
        assert x <= right + 0.1, (page, text, x, right)
        geometry.append({"page": page, "x1": round(x, 2), "y": round(baseline, 2)})

    def new_page(name, continuation=False):
        nonlocal page, y
        if page:
            c.showPage()
        page += 1
        c.setFont("CodeMono", 10)
        c.drawCentredString(width / 2, 17.5 * mm, str(len(main.pages) - 2 + page))
        y = top - 10
        draw(name + ("（续）" if continuation else ""), left, y, 10)
        y -= 22

    for p in source_files(ROOT):
        relative = p.relative_to(ROOT).as_posix()
        content = p.read_text(encoding="utf-8-sig")
        lines = content.splitlines()
        new_page(relative)
        start = page
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        draw("SHA-256: " + sha, left, y, 8)
        y -= 18
        for number, line in enumerate(lines, 1):
            line = line.expandtabs(4)
            chunks, current, occupied = [], "", 0.0
            for char in line:
                advance = pdfmetrics.stringWidth(char, font(char), font_size)
                if occupied + advance > right - code_x:
                    chunks.append(current)
                    current, occupied = "", 0.0
                current += char
                occupied += advance
            chunks.append(current)
            for j, chunk in enumerate(chunks):
                if y < bottom:
                    new_page(relative, True)
                if j == 0:
                    c.setFont("CodeMono", 7)
                    c.setFillGray(.45)
                    c.drawRightString(code_x - 7, y, str(number))
                    c.setFillGray(0)
                draw(chunk, code_x, y)
                y -= leading
        records.append({"path": relative, "lines": len(lines), "sha256": sha,
                        "first_pdf_page": len(main.pages) + start,
                        "last_pdf_page": len(main.pages) + page})
    c.save()
    writer = PdfWriter(clone_from=QA / "main.pdf")
    # Rebuild internal contents links: COM exporters may misread PAGEREF switches.
    contents = writer.pages[1]
    exported_rows = sorted({(round(float(a.get_object()["/Rect"][1]), 1),
                             round(float(a.get_object()["/Rect"][3]), 1))
                            for a in contents.get("/Annots", [])}, reverse=True)
    if "/Annots" in contents:
        del contents[NameObject("/Annots")]
    blocks = json.loads((HERE / "manuscript.json").read_text(encoding="utf-8"))["blocks"]
    headings = [b for b in blocks if b["type"] == "heading" and
                (b["level"] == 1 or (b["level"] == 2 and re.match(r"[567]\.\d+ ", b["text"])))]
    normalized = [re.sub(r"\s+", "", p.extract_text()) for p in main.pages]
    toc_links = []
    for i, b in enumerate(headings):
        text = b["text"]
        target = next(j for j in range(2, len(normalized)) if re.sub(r"\s+", "", text) in normalized[j])
        y0, y1 = exported_rows[i] if len(exported_rows) == len(headings) else (734.55 - i * 13, 747.55 - i * 13)
        annotation = writer.add_annotation(1, Link(rect=(70.9, y0, 524.5, y1), target_page_index=target))
        annotation["/Dest"][0] = writer.pages[target].indirect_reference
        writer.add_outline_item(text, target)
        toc_links.append({"text": text, "target_pdf_page": target + 1, "rect": [70.9, y0, 524.5, y1]})
    writer.append(path)
    for r in records:
        writer.add_outline_item(r["path"], r["first_pdf_page"] - 1)
    writer.add_metadata({"/Title": "啤酒风味预测、关键化合物关联与风格迁移", "/Author": ""})
    final = OUT / "A2600730.pdf"
    with final.open("wb") as f:
        writer.write(f)
    manifest = {"main_pages": len(main.pages), "source_pages": page,
                "total_pages": len(main.pages) + page, "files": records,
                "source_lines": sum(r["lines"] for r in records), "contents_links": toc_links,
                "layout": {"font_pt": font_size, "leading_pt": leading,
                           "all_lines_within_margins": True,
                           "contents_rectangles": "exported" if len(exported_rows) == len(headings) else "legacy",
                           "fonts": {"mono": Path(mono_font).name, "cjk": Path(cjk_font).name,
                                     "cjk_index": cjk_index, "math": Path(math_font).name, "math_index": math_index}}}
    (QA / "source_inventory.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("files", "contents_links")}, ensure_ascii=False))
    print(final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono-font", default="C:/Windows/Fonts/consola.ttf")
    parser.add_argument("--cjk-font", default="C:/Windows/Fonts/simsun.ttc")
    parser.add_argument("--math-font", default="C:/Windows/Fonts/cambria.ttc")
    parser.add_argument("--cjk-index", type=int, default=0)
    parser.add_argument("--math-index", type=int, default=1)
    args = parser.parse_args()
    build(args.mono_font, args.cjk_font, args.math_font, args.cjk_index, args.math_index)
