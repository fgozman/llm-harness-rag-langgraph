"""Canary documents: locally generated FICTION with invented facts, for format coverage and abstention tests.

Every fact lives in exactly one place (file + location) recorded in eval/canary_facts.jsonl. Unanswerable
"near-miss" questions ask about plausible details that the documents deliberately never mention.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from docx import Document
from openpyxl import Workbook
from pptx import Presentation
from pptx.shapes.autoshape import Shape

from corpus_builder.registry import Registry, SourceRecord
from corpus_builder.stats import SourceStats

FICTION_MARK = "FICTION: canary document generated for RAG evaluation; all names and facts are invented."
LICENSE = "CC0 1.0 (generated fiction)"
COMPANY = "Vellmark Aerostructures"


@dataclass(frozen=True, slots=True)
class Fact:
    id: str
    file: str
    location: str
    question: str
    answer: str
    answerable: bool = True


PPTX = "canary/vellmark_board_briefing_2026-q3.pptx"
POLICY = "canary/vellmark_supplier_quality_policy.docx"
INCIDENT = "canary/vellmark_incident_review_2026-09.docx"
XLSX = "canary/vellmark_fleet_metrics_2026.xlsx"

FACTS: list[Fact] = [
    Fact("pptx-1", PPTX, "slide 2", "What revenue did Vellmark report for Q3 2026?", "EUR 41.7 million"),
    Fact(
        "pptx-2",
        PPTX,
        "slide 2 speaker notes",
        "Which customer drove Vellmark's Q3 backlog growth?",
        "Norrsken Air",
    ),
    Fact(
        "pptx-3",
        PPTX,
        "slide 3",
        "What is the name of Vellmark's composite wing-rib programme?",
        "Project Tern",
    ),
    Fact(
        "pptx-4",
        PPTX,
        "slide 3 speaker notes",
        "When is Project Tern's first article inspection?",
        "14 January 2027",
    ),
    Fact(
        "pptx-5",
        PPTX,
        "slide 4 speaker notes",
        "Which plant has the highest scrap rate at Vellmark?",
        "Kalmar",
    ),
    Fact(
        "pol-1",
        POLICY,
        "section 'Supplier tiers'",
        "How often are Tier A suppliers audited?",
        "every 18 months",
    ),
    Fact(
        "pol-2", POLICY, "table 1", "What is the maximum allowed defect rate for Tier B suppliers?", "350 ppm"
    ),
    Fact("pol-3", POLICY, "table 1", "Which supplier tier requires a resident quality engineer?", "Tier C"),
    Fact(
        "pol-4",
        POLICY,
        "section 'Escalation'",
        "Who chairs Vellmark's supplier escalation board?",
        "Ingrid Solberg",
    ),
    Fact(
        "inc-1", INCIDENT, "section 'Summary'", "Which part failed in incident VA-2026-031?", "titanium shim"
    ),
    Fact(
        "inc-2",
        INCIDENT,
        "section 'Root cause'",
        "What torque value was wrongly specified in VA-2026-031?",
        "62 Nm",
    ),
    Fact(
        "inc-3",
        INCIDENT,
        "section 'Corrective actions'",
        "By when must the VA-2026-031 fixture be replaced?",
        "31 October 2026",
    ),
    Fact("xls-1", XLSX, "sheet 'Fleet'", "How many flight hours did airframe VX-214 log in 2026?", "1,932"),
    Fact("xls-2", XLSX, "sheet 'Fleet'", "Which operator flies airframe VX-209?", "Bris Regional"),
    Fact(
        "xls-3",
        XLSX,
        "sheet 'Warranty'",
        "How much was paid in warranty claims for the VR-9 actuator?",
        "EUR 118,400",
    ),
    # Near misses: plausible but never stated anywhere in the canary documents.
    Fact("neg-1", PPTX, "-", "What revenue did Vellmark report for Q2 2026?", "", answerable=False),
    Fact("neg-2", POLICY, "-", "How often are Tier D suppliers audited?", "", answerable=False),
    Fact("neg-3", INCIDENT, "-", "Which part failed in incident VA-2026-032?", "", answerable=False),
    Fact("neg-4", XLSX, "-", "How many flight hours did airframe VX-221 log in 2026?", "", answerable=False),
]


def _build_pptx(path: Path) -> None:
    prs = Presentation()
    slides = [
        (
            "Vellmark Aerostructures — Q3 2026 Board Briefing",
            "Prepared for the Board, October 2026",
            "Fictional.",
        ),
        (
            "Q3 2026 results",
            "Revenue EUR 41.7 million\nOperating margin 9.4%\nBacklog EUR 212 million",
            "Backlog growth was driven almost entirely by Norrsken Air's follow-on order for 40 shipsets.",
        ),
        (
            "Programmes",
            "Project Tern: composite wing ribs for regional jets\nProject Skua: cabin floor beams",
            "Project Tern's first article inspection is scheduled for 14 January 2027.",
        ),
        (
            "Operations",
            "Three plants: Kalmar, Tampere, Gdansk",
            "Kalmar currently has the highest scrap rate of the three plants at 4.1 percent.",
        ),
    ]
    for title, body, notes in slides:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        title_shape, body_shape = slide.shapes.title, slide.placeholders[1]
        notes_frame = slide.notes_slide.notes_text_frame
        assert isinstance(title_shape, Shape) and isinstance(body_shape, Shape) and notes_frame is not None
        title_shape.text_frame.text = title
        body_shape.text_frame.text = body
        notes_frame.text = notes
    prs.core_properties.title = "Vellmark Q3 2026 Board Briefing"
    prs.core_properties.comments = FICTION_MARK
    prs.save(str(path))


def _build_policy(path: Path) -> None:
    doc = Document()
    doc.add_heading("Vellmark Aerostructures Supplier Quality Policy", level=0)
    doc.add_heading("Scope", level=1)
    doc.add_paragraph(f"This policy applies to all suppliers of flight hardware to {COMPANY}.")
    doc.add_heading("Supplier tiers", level=1)
    doc.add_paragraph(
        "Suppliers are classified into three tiers. Tier A suppliers are audited every 18 months; "
        "Tier B and Tier C suppliers are audited annually."
    )
    table = doc.add_table(rows=1, cols=3)
    for cell, text in zip(
        table.rows[0].cells, ("Tier", "Max defect rate", "On-site requirement"), strict=True
    ):
        cell.text = text
    for row in (
        ("Tier A", "150 ppm", "None"),
        ("Tier B", "350 ppm", "Quarterly visit"),
        ("Tier C", "800 ppm", "Resident quality engineer"),
    ):
        cells = table.add_row().cells
        for cell, text in zip(cells, row, strict=True):
            cell.text = text
    doc.add_heading("Escalation", level=1)
    doc.add_paragraph(
        "Repeated nonconformances go to the supplier escalation board, chaired by Ingrid Solberg."
    )
    doc.core_properties.title = "Vellmark Supplier Quality Policy"
    doc.core_properties.comments = FICTION_MARK
    doc.save(str(path))


def _build_incident(path: Path) -> None:
    doc = Document()
    doc.add_heading("Incident Review VA-2026-031", level=0)
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(
        "On 9 September 2026 a titanium shim cracked during final assembly of a wing-rib subassembly "
        "at Tampere."
    )
    doc.add_heading("Root cause", level=1)
    doc.add_paragraph(
        "The work instruction specified 62 Nm instead of the design value of 26 Nm for the shim fasteners."
    )
    doc.add_heading("Corrective actions", level=1)
    doc.add_paragraph(
        "The assembly fixture must be replaced by 31 October 2026 and the work instruction reissued."
    )
    doc.core_properties.title = "Vellmark Incident Review VA-2026-031"
    doc.core_properties.comments = FICTION_MARK
    doc.save(str(path))


def _build_xlsx(path: Path) -> None:
    wb = Workbook()
    fleet = wb.active
    assert fleet is not None
    fleet.title = "Fleet"
    fleet.append(["Airframe", "Operator", "Flight hours 2026", "Status"])
    for row in (
        ("VX-209", "Bris Regional", "2,105", "Active"),
        ("VX-214", "Norrsken Air", "1,932", "Active"),
        ("VX-217", "Norrsken Air", "640", "Maintenance"),
    ):
        fleet.append(list(row))
    warranty = wb.create_sheet("Warranty")
    warranty.append(["Component", "Claims", "Paid"])
    warranty.append(["VR-9 actuator", 12, "EUR 118,400"])
    warranty.append(["Door seal kit", 31, "EUR 22,750"])
    wb.properties.title = "Vellmark Fleet Metrics 2026"
    wb.properties.description = FICTION_MARK
    wb.save(str(path))


_BUILDERS = {PPTX: _build_pptx, POLICY: _build_policy, INCIDENT: _build_incident, XLSX: _build_xlsx}
_TITLES = {
    PPTX: "Vellmark Q3 2026 Board Briefing (fiction)",
    POLICY: "Vellmark Supplier Quality Policy (fiction)",
    INCIDENT: "Vellmark Incident Review VA-2026-031 (fiction)",
    XLSX: "Vellmark Fleet Metrics 2026 (fiction)",
}


def build_canaries(out_dir: Path, facts_path: Path, registry: Registry) -> SourceStats:
    stats = SourceStats("canary")
    for rel, builder in _BUILDERS.items():
        target = out_dir / rel
        if target.exists() and registry.has(rel):
            stats.skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        builder(target)
        registry.upsert(
            SourceRecord(
                rel,
                "canary",
                _TITLES[rel],
                "generated locally",
                date.today().isoformat(),
                LICENSE,
                fiction=True,
                notes=FICTION_MARK,
            )
        )
        stats.added += 1
    facts_path.parent.mkdir(parents=True, exist_ok=True)
    facts_path.write_text("".join(json.dumps(asdict(f)) + "\n" for f in FACTS))
    return stats
