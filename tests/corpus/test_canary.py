import json
from datetime import date
from pathlib import Path

from docx import Document
from openpyxl import load_workbook
from pptx import Presentation
from pptx.shapes.autoshape import Shape

from corpus_builder.canary import FICTION_MARK, build_canaries
from corpus_builder.registry import Registry


def pptx_location_text(path: Path, location: str) -> str:
    prs = Presentation(str(path))
    slide_no = int(location.split()[1])
    slide = prs.slides[slide_no - 1]
    if "notes" in location:
        frame = slide.notes_slide.notes_text_frame
        assert frame is not None
        return frame.text
    return "\n".join(shape.text_frame.text for shape in slide.shapes if isinstance(shape, Shape))


def docx_location_text(path: Path, location: str) -> str:
    doc = Document(str(path))
    if location.startswith("table"):
        table = doc.tables[int(location.split()[1]) - 1]
        return "\n".join(" | ".join(c.text for c in row.cells) for row in table.rows)
    section = location.removeprefix("section ").strip("'\"")
    out: list[str] = []
    inside = False
    for p in doc.paragraphs:
        if p.style is not None and (p.style.name or "").startswith("Heading"):
            inside = p.text == section
            continue
        if inside:
            out.append(p.text)
    return "\n".join(out)


def xlsx_location_text(path: Path, location: str) -> str:
    sheet = location.split("'")[1]
    ws = load_workbook(path, read_only=True)[sheet]
    return "\n".join(
        " | ".join("" if v is None else str(v) for v in row) for row in ws.iter_rows(values_only=True)
    )


def location_text(path: Path, location: str) -> str:
    if path.suffix == ".pptx":
        return pptx_location_text(path, location)
    if path.suffix == ".docx":
        return docx_location_text(path, location)
    return xlsx_location_text(path, location)


def test_canaries_cover_formats_and_every_fact_is_where_it_claims(tmp_path: Path) -> None:
    out = tmp_path / "corpus"
    facts_path = tmp_path / "eval" / "canary_facts.jsonl"
    registry = Registry(out / "sources.json")
    stats = build_canaries(out, facts_path, registry)
    files = sorted(p.relative_to(out).as_posix() for p in (out / "canary").iterdir())
    assert [Path(f).suffix for f in files].count(".docx") == 2
    assert [Path(f).suffix for f in files].count(".pptx") == 1
    assert [Path(f).suffix for f in files].count(".xlsx") == 1
    assert stats.added == 4

    facts = [json.loads(line) for line in facts_path.read_text().splitlines()]
    answerable = [f for f in facts if f["answerable"]]
    assert len(answerable) >= 12 and len(facts) > len(answerable)
    assert len({f["id"] for f in facts}) == len(facts)
    for fact in answerable:
        text = location_text(out / fact["file"], fact["location"])
        assert fact["answer"] in text, fact["id"]
    for fact in facts:
        assert fact["file"] in files

    assert any("speaker notes" in f["location"] for f in answerable), "PPTX notes must carry facts"
    assert all(r.fiction for r in registry.records)


def test_canary_files_carry_fiction_metadata(tmp_path: Path) -> None:
    out = tmp_path / "corpus"
    build_canaries(out, tmp_path / "facts.jsonl", Registry(out / "sources.json"))
    for path in (out / "canary").iterdir():
        if path.suffix == ".docx":
            assert FICTION_MARK in (Document(str(path)).core_properties.comments or "")
        elif path.suffix == ".pptx":
            assert FICTION_MARK in (Presentation(str(path)).core_properties.comments or "")
        else:
            assert FICTION_MARK in (load_workbook(path).properties.description or "")


def test_canary_docx_creation_dates_are_set(tmp_path: Path) -> None:
    # Not python-docx's template default (2013-12-23): the metadata date dates undated documents.
    from rag.ingestion.dates import metadata_date

    out = tmp_path / "corpus"
    build_canaries(out, tmp_path / "facts.jsonl", Registry(out / "sources.json"))
    policy = out / "canary" / "vellmark_supplier_quality_policy.docx"
    assert metadata_date(policy) == date(2026, 10, 12)


def test_canary_build_is_idempotent(tmp_path: Path) -> None:
    out = tmp_path / "corpus"
    registry = Registry(out / "sources.json")
    build_canaries(out, tmp_path / "facts.jsonl", registry)
    again = build_canaries(out, tmp_path / "facts.jsonl", registry)
    assert (again.added, again.skipped) == (0, 4)


def test_builder_ragignore_excludes_provenance_files() -> None:
    from corpus_builder.registry import RAGIGNORE

    assert "SOURCES.md" in RAGIGNORE.splitlines()
