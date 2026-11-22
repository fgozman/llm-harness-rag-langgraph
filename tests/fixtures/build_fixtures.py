"""Regenerate the binary fixture documents in tests/fixtures/docs/.

All content is fictional. Run: `uv run python tests/fixtures/build_fixtures.py`.
Text formats (md, html, txt) are hand-written and committed directly.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
from docx import Document
from openpyxl import Workbook
from pptx import Presentation
from pptx.util import Inches, Pt

OUT = Path(__file__).parent / "docs"


def build_docx(path: Path) -> None:
    doc = Document()
    doc.add_heading("Northwind Robotics Warranty Policy", level=0)
    doc.add_heading("1. Coverage", level=1)
    doc.add_paragraph(
        "All Northwind arms are covered for 36 months from the date of shipment. "
        "Coverage includes servo motors, harmonic drives, and the joint controller boards."
    )
    doc.add_heading("1.1 Exclusions", level=2)
    for item in (
        "Damage from operation above rated payload",
        "Water ingress beyond IP54",
        "Unauthorized firmware",
    ):
        doc.add_paragraph(item, style="List Bullet")
    doc.add_heading("2. Claim process", level=1)
    doc.add_paragraph("Claims must be filed through the partner portal within 14 days of the fault.")
    table = doc.add_table(rows=1, cols=3)
    table.rows[0].cells[0].text = "Tier"
    table.rows[0].cells[1].text = "Response time"
    table.rows[0].cells[2].text = "Loaner unit"
    for row in (("Standard", "5 business days", "No"), ("Priority", "48 hours", "Yes")):
        cells = table.add_row().cells
        for cell, value in zip(cells, row, strict=True):
            cell.text = value
    doc.save(str(path))


def build_pptx(path: Path) -> None:
    prs = Presentation()
    slides = [
        ("Q3 Product Roadmap", "Northwind Robotics internal briefing", "Presenter: keep this slide short."),
        (
            "Launch: NRX-4471-B gripper",
            "Payload 7.5 kg, cycle time 0.42 s, ships in October",
            "The NRX-4471-B replaces the NRX-3300 line; pricing is 4,950 EUR per unit.",
        ),
        ("Risks", "Supplier delay on harmonic drives", "Mitigation: second source qualified in Vilnius."),
    ]
    for title, body, notes in slides:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
        slide.notes_slide.notes_text_frame.text = notes
    box = prs.slides[0].shapes.add_textbox(Inches(1), Inches(6), Inches(6), Inches(1))
    box.text_frame.text = "Confidential"
    box.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
    prs.save(str(path))


def build_xlsx(path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Shipments"
    ws.append(["Region", "Model", "Units Q2", "Units Q3"])
    for row in (
        ("EMEA", "NRX-3300", 410, 380),
        ("EMEA", "NRX-4471-B", 0, 125),
        ("APAC", "NRX-3300", 290, 305),
        ("Americas", "NRX-3300", 515, 498),
    ):
        ws.append(list(row))
    prices = wb.create_sheet("Prices")
    prices.append(["Model", "List price EUR"])
    prices.append(["NRX-3300", 3900])
    prices.append(["NRX-4471-B", 4950])
    wb.save(str(path))


def build_pdf(path: Path) -> None:
    pages = [
        (
            "Northwind Robotics Safety Manual",
            [
                "Chapter 1: Installation",
                "Mount the arm on a base rated for at least 3x the arm's static load.",
                "Anchor bolts must be torqued to 85 Nm and re-checked after 100 operating hours.",
            ],
        ),
        (
            "Chapter 2: Emergency stop",
            [
                "The emergency stop circuit is a dual-channel Category 3 design.",
                "Stopping distance at full speed is 210 mm for the NRX-3300 and 165 mm for the NRX-4471-B.",
                "Test the emergency stop at the start of every shift.",
            ],
        ),
    ]
    doc = pymupdf.open()
    for title, lines in pages:
        page = doc.new_page()
        page.insert_text((72, 80), title, fontsize=18)
        y = 120.0
        for line in lines:
            size = 14 if line.startswith("Chapter") else 11
            page.insert_text((72, y), line, fontsize=size)
            y += 24
    doc.save(str(path))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_docx(OUT / "warranty_policy.docx")
    build_pptx(OUT / "roadmap_q3.pptx")
    build_xlsx(OUT / "shipments.xlsx")
    build_pdf(OUT / "safety_manual.pdf")


if __name__ == "__main__":
    main()
