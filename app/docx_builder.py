import os
from datetime import UTC, datetime
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from app.schemas import SectionContent, TableContent, TaskResult


def _add_page_number_field(run) -> None:
    """Insert a dynamic PAGE field into a Word document run."""
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._r.append(fld)


def _style_header_cell(cell, text: str) -> None:
    """Apply background shading and white bold font to a table header cell."""
    cell.text = text
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "1F4E79")
    tc_pr.append(shd)

    for p in cell.paragraphs:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(10)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)


def build_docx(
    run_id: str,
    title: str,
    doc_type: str,
    assumptions: list[str],
    tasks: list[TaskResult],
    output_dir: str = "outputs",
) -> str:
    """Construct a Word document matching requirements and save it to output_dir."""
    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        frun_label = fp.add_run("Page ")
        frun_label.font.name = "Calibri"
        frun_label.font.size = Pt(9)
        frun_label.font.color.rgb = RGBColor(128, 128, 128)
        frun_page = fp.add_run()
        frun_page.font.name = "Calibri"
        frun_page.font.size = Pt(9)
        frun_page.font.color.rgb = RGBColor(128, 128, 128)
        _add_page_number_field(frun_page)

    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(30, 30, 30)

    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    run_title = title_p.add_run(title)
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(22)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(31, 78, 121)

    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_before = Pt(0)
    meta_p.paragraph_format.space_after = Pt(18)
    formatted_type = doc_type.replace("_", " ").upper()
    formatted_date = datetime.now(UTC).strftime("%B %d, %Y")
    run_meta = meta_p.add_run(f"DOCUMENT TYPE: {formatted_type}  |  DATE: {formatted_date}")
    run_meta.font.name = "Calibri"
    run_meta.font.size = Pt(10)
    run_meta.font.bold = True
    run_meta.font.color.rgb = RGBColor(112, 128, 144)

    assump_heading = doc.add_heading("Assumptions and Decisions", level=1)
    assump_heading.paragraph_format.space_before = Pt(12)
    assump_heading.paragraph_format.space_after = Pt(6)

    if assumptions:
        doc.add_paragraph(
            "The following assumptions and conflict resolutions guided the generation of this document:"
        )
        for assumption in assumptions:
            doc.add_paragraph(assumption, style="List Bullet")
    else:
        doc.add_paragraph("No explicit assumptions were required for this request.")

    for task in tasks:
        if task.status == "failed" or task.content is None:
            failed_h = doc.add_heading(task.title, level=1)
            failed_h.paragraph_format.space_before = Pt(14)
            failed_h.paragraph_format.space_after = Pt(6)
            fail_p = doc.add_paragraph(
                f"[Generation failed for this section. Reason: {task.error_message or 'Unknown error'}. Manual drafting required.]"
            )
            for r in fail_p.runs:
                r.font.italic = True
                r.font.color.rgb = RGBColor(180, 40, 40)
            continue

        if isinstance(task.content, SectionContent):
            h = doc.add_heading(task.content.title or task.title, level=1)
            h.paragraph_format.space_before = Pt(14)
            h.paragraph_format.space_after = Pt(6)

            for paragraph in task.content.paragraphs:
                p = doc.add_paragraph(paragraph)
                p.paragraph_format.space_after = Pt(6)

            for bullet in task.content.bullet_points:
                bp = doc.add_paragraph(bullet, style="List Bullet")
                bp.paragraph_format.space_after = Pt(3)

        elif isinstance(task.content, TableContent):
            h = doc.add_heading(task.content.title or task.title, level=1)
            h.paragraph_format.space_before = Pt(14)
            h.paragraph_format.space_after = Pt(6)

            if task.content.description:
                desc_p = doc.add_paragraph(task.content.description)
                desc_p.paragraph_format.space_after = Pt(6)

            headers = task.content.headers
            rows = task.content.rows
            if headers:
                table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
                table.style = "Table Grid"
                table.alignment = WD_TABLE_ALIGNMENT.CENTER

                for col_idx, header_text in enumerate(headers):
                    cell = table.cell(0, col_idx)
                    _style_header_cell(cell, header_text)

                for row_idx, row_data in enumerate(rows):
                    for col_idx in range(len(headers)):
                        val = row_data[col_idx] if col_idx < len(row_data) else ""
                        cell = table.cell(row_idx + 1, col_idx)
                        cell.text = str(val)
                        for p in cell.paragraphs:
                            for r in p.runs:
                                r.font.name = "Calibri"
                                r.font.size = Pt(10)

                spacer = doc.add_paragraph()
                spacer.paragraph_format.space_before = Pt(6)
                spacer.paragraph_format.space_after = Pt(0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(output_dir, f"{run_id}.docx")
    doc.save(file_path)
    return file_path
