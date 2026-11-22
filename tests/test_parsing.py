from pathlib import Path

import pytest

from rag.ingestion.models import BlockKind, ParsedDocument
from rag.ingestion.parsing import DocumentParser, parse_text

FIXTURES = Path(__file__).parent / "fixtures" / "docs"


def test_text_paragraphs_become_blocks(tmp_path: Path) -> None:
    p = tmp_path / "notes.txt"
    p.write_text("First para\nstill first.\n\n\nSecond para.\n")
    doc = parse_text(p)
    assert doc.title == "notes"
    assert [b.text for b in doc.blocks] == ["First para\nstill first.", "Second para."]
    assert all(b.kind is BlockKind.TEXT for b in doc.blocks)
    assert doc.parser == "text"


def test_txt_never_touches_docling(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = DocumentParser(pdf_ocr=False)

    def boom(path: Path) -> ParsedDocument:
        raise AssertionError("docling should not be used for .txt")

    monkeypatch.setattr(parser, "_parse_docling", boom)
    doc = parser.parse(FIXTURES / "release_notes.txt")
    assert any("4.3.1" in b.text for b in doc.blocks)


def test_pdf_falls_back_to_pymupdf_when_docling_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = DocumentParser(pdf_ocr=False)

    def boom(path: Path) -> ParsedDocument:
        raise RuntimeError("docling exploded")

    monkeypatch.setattr(parser, "_parse_docling", boom)
    doc = parser.parse(FIXTURES / "safety_manual.pdf")
    assert doc.parser == "pymupdf"
    assert {b.page for b in doc.blocks} == {1, 2}
    assert any("85 Nm" in b.text for b in doc.blocks)


def test_markdown_falls_back_to_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = DocumentParser(pdf_ocr=False)

    def boom(path: Path) -> ParsedDocument:
        raise RuntimeError("docling exploded")

    monkeypatch.setattr(parser, "_parse_docling", boom)
    doc = parser.parse(FIXTURES / "engineering" / "servo_tuning.md")
    assert doc.parser == "text"
    assert any("E-217" in b.text for b in doc.blocks)


def test_formats_without_fallback_reraise(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = DocumentParser(pdf_ocr=False)

    def boom(path: Path) -> ParsedDocument:
        raise RuntimeError("docling exploded")

    monkeypatch.setattr(parser, "_parse_docling", boom)
    with pytest.raises(RuntimeError, match="docling exploded"):
        parser.parse(FIXTURES / "warranty_policy.docx")


# --- Docling behavior observed in the Phase 1 spike, pinned here -----------------------------------


@pytest.fixture(scope="module")
def docling_parser() -> DocumentParser:
    return DocumentParser(pdf_ocr=False)


def headings(doc: ParsedDocument) -> list[tuple[int, str]]:
    return [(b.level, b.text) for b in doc.blocks if b.kind is BlockKind.HEADING]


@pytest.mark.integration
def test_docling_markdown_keeps_heading_levels_and_tables(docling_parser: DocumentParser) -> None:
    doc = docling_parser.parse(FIXTURES / "engineering" / "servo_tuning.md")
    assert doc.parser == "docling" and doc.title == "Servo Tuning Guide"
    assert (1, "Default gains") in headings(doc) and (2, "Step 1: Proportional gain") in headings(doc)
    (table,) = [b for b in doc.blocks if b.kind is BlockKind.TABLE]
    assert table.text.splitlines()[0].startswith("| Joint")
    assert any(b.kind is BlockKind.LIST_ITEM for b in doc.blocks)


@pytest.mark.integration
def test_docling_pptx_includes_speaker_notes_with_slide_pages(docling_parser: DocumentParser) -> None:
    doc = docling_parser.parse(FIXTURES / "roadmap_q3.pptx")
    notes = [b for b in doc.blocks if b.text.startswith("Speaker notes:")]
    assert len(notes) == 3
    assert any("4,950 EUR" in b.text and b.page == 2 for b in notes)
    assert (0, "Launch: NRX-4471-B gripper") in headings(doc)


@pytest.mark.integration
def test_docling_xlsx_sheets_become_headings(docling_parser: DocumentParser) -> None:
    doc = docling_parser.parse(FIXTURES / "shipments.xlsx")
    assert headings(doc) == [(1, "Shipments"), (1, "Prices")]
    assert sum(b.kind is BlockKind.TABLE for b in doc.blocks) == 2


@pytest.mark.integration
def test_docling_pdf_has_pages(docling_parser: DocumentParser) -> None:
    doc = docling_parser.parse(FIXTURES / "safety_manual.pdf")
    # Docling labels this PDF's title as a section header; a leading heading is promoted to title.
    assert doc.title == "Northwind Robotics Safety Manual"
    assert {b.page for b in doc.blocks} == {1, 2}
    assert any("85 Nm" in b.text for b in doc.blocks)


@pytest.mark.integration
def test_docling_html_skips_furniture_but_keeps_untrusted_text_as_data(
    docling_parser: DocumentParser,
) -> None:
    doc = docling_parser.parse(FIXTURES / "support_faq.html")
    assert [h for h in headings(doc) if h[1] == "Northwind Support FAQ"] == [(0, "Northwind Support FAQ")]
    assert any("PWNED" in b.text for b in doc.blocks)


@pytest.mark.integration
def test_docling_docx_lists_and_table(docling_parser: DocumentParser) -> None:
    doc = docling_parser.parse(FIXTURES / "warranty_policy.docx")
    assert (2, "1.1 Exclusions") in headings(doc)
    assert sum(b.kind is BlockKind.LIST_ITEM for b in doc.blocks) == 3
    assert any(b.kind is BlockKind.TABLE and "Priority" in b.text for b in doc.blocks)
