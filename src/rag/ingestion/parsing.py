# Docling is partially untyped and re-exports its public API without __all__ (so pyright calls the documented
# `docling_core.types.doc` imports "private"). Relax only those checks, only in this adapter.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportPrivateImportUsage=false
"""Convert files to `ParsedDocument`: Docling first, lightweight fallbacks when it fails."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from docling_core.types.doc import (
    CodeItem,
    DoclingDocument,
    GroupItem,
    GroupLabel,
    ListItem,
    NodeItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)
from docling_core.types.doc.document import ContentLayer

from rag.ingestion.models import Block, BlockKind, ParsedDocument

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

log = logging.getLogger(__name__)

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_IX_HEADER = re.compile(rb"<ix:header\b.*?</ix:header>", re.S | re.I)
_IX_NAMESPACE = b"http://www.xbrl.org/2013/inlineXBRL"
_HTML_SUFFIXES = frozenset({".htm", ".html"})


_PREFIX_SYMBOLS = frozenset({"$", "€", "£", "("})
_SUFFIX_SYMBOLS = frozenset({"%", ")", "%)"})
# Dash variants: hyphen, em dash (U+2014), en dash (U+2013).
_SEC_PART = re.compile(r"^PART\s+[IVX]+\s*(?:[-\u2014\u2013:.]|$)", re.I)
_SEC_ITEM = re.compile(r"^ITEM\s+\d+[A-Z]?\.\s+\S", re.I)
_MAX_SEC_HEADING_CHARS = 150


def _fill_run(row: list[str], start: int, step: int, old: str, new: str) -> None:
    # Spanned values are repeated across columns; update the whole run so the copies stay identical.
    i = start
    while 0 <= i < len(row) and row[i] == old:
        row[i] = new
        i += step


def _attach_symbols(row: list[str]) -> list[str]:
    out = list(row)
    for c, cell in enumerate(row):
        if cell in _PREFIX_SYMBOLS:
            j = next((k for k in range(c + 1, len(out)) if out[k] and out[k] not in _PREFIX_SYMBOLS), None)
            if j is not None:
                value = out[j]
                _fill_run(out, j, 1, value, cell + value)
                for k in range(c, j):
                    out[k] = cell + value
        elif cell in _SUFFIX_SYMBOLS:
            j = next((k for k in range(c - 1, -1, -1) if out[k] and out[k] not in _SUFFIX_SYMBOLS), None)
            if j is not None:
                value = out[j]
                _fill_run(out, j, -1, value, value + cell)
                for k in range(j + 1, c + 1):
                    out[k] = value + cell
    return out


def clean_table(rows: list[list[str]]) -> list[list[str]]:
    """Compact HTML-derived tables (e.g. SEC filings): attach $/%/) to numbers, drop empty rows, and drop
    columns whose every cell is empty or repeats its left neighbour (span copies, spacers). Loses no text."""
    rows = [_attach_symbols([cell.strip() for cell in row]) for row in rows]
    rows = [row for row in rows if any(row)]
    if not rows:
        return []
    keep = [0]
    for c in range(1, len(rows[0])):
        last = keep[-1]
        if not all(row[c] in ("", row[last]) for row in rows):
            keep.append(c)
    if not any(row[0] for row in rows):
        keep = keep[1:] or keep
    return [[row[c] for c in keep] for row in rows]


def rows_to_markdown(rows: list[list[str]]) -> str:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |"

    if not rows:
        return ""
    return "\n".join([line(rows[0]), line(["---"] * len(rows[0])), *(line(r) for r in rows[1:])])


def promote_sec_headings(blocks: list[Block]) -> list[Block]:
    """SEC filings style "PART I" / "Item 2. ..." as bold text rather than <h*>; restore them as headings."""
    out: list[Block] = []
    for b in blocks:
        text = b.text.strip()
        is_short = len(text) <= _MAX_SEC_HEADING_CHARS and not text.endswith(".")
        if b.kind is BlockKind.TEXT and is_short and _SEC_PART.match(text):
            out.append(Block(BlockKind.HEADING, text, level=1, page=b.page))
        elif b.kind is BlockKind.TEXT and is_short and _SEC_ITEM.match(text):
            out.append(Block(BlockKind.HEADING, text, level=2, page=b.page))
        else:
            out.append(b)
    return out


def strip_ixbrl_header(content: bytes) -> bytes:
    # The ix:header block holds hidden XBRL contexts/metadata (CIKs, flags) that would pollute the text.
    return _IX_HEADER.sub(b"", content)


def _paragraph_blocks(text: str, page: int | None = None) -> list[Block]:
    return [Block(BlockKind.TEXT, p.strip(), page=page) for p in _PARAGRAPH_BREAK.split(text) if p.strip()]


def parse_text(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    return ParsedDocument(title=path.stem, blocks=_paragraph_blocks(text), parser="text")


def parse_pdf_pymupdf(path: Path) -> ParsedDocument:
    import pymupdf

    blocks: list[Block] = []
    with pymupdf.open(path) as pdf:
        title = (pdf.metadata or {}).get("title") or path.stem
        for idx in range(pdf.page_count):
            blocks.extend(_paragraph_blocks(str(pdf.load_page(idx).get_text()), page=idx + 1))
    return ParsedDocument(title=title, blocks=blocks, parser="pymupdf")


_FALLBACKS: dict[str, Callable[[Path], ParsedDocument]] = {
    ".pdf": parse_pdf_pymupdf,
    ".md": parse_text,
    ".markdown": parse_text,
}


class DocumentParser:
    def __init__(self, *, pdf_ocr: bool) -> None:
        self._pdf_ocr = pdf_ocr

    def parse(self, path: Path) -> ParsedDocument:
        # Docling has no plain-text input format; .txt is read directly.
        if path.suffix.lower() == ".txt":
            return parse_text(path)
        try:
            return self._parse_docling(path)
        except Exception as exc:
            fallback = _FALLBACKS.get(path.suffix.lower())
            if fallback is None:
                raise
            log.warning("docling failed on %s (%s); using fallback parser", path, exc)
            return fallback(path)

    @cached_property
    def _converter(self) -> DocumentConverter:
        # Imported lazily: docling pulls in torch and layout models, too heavy for commands that don't parse.
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions(do_ocr=self._pdf_ocr, do_table_structure=True)
        return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})

    def _parse_docling(self, path: Path) -> ParsedDocument:
        if self._is_inline_xbrl(path):
            blocks = promote_sec_headings(_docling_blocks(self._convert_inline_xbrl(path)))
        else:
            blocks = _docling_blocks(self._converter.convert(path, raises_on_error=True).document)
        return ParsedDocument(title=_title(blocks, path), blocks=blocks, parser="docling")

    @staticmethod
    def _is_inline_xbrl(path: Path) -> bool:
        if path.suffix.lower() not in _HTML_SUFFIXES:
            return False
        with path.open("rb") as fh:
            return _IX_NAMESPACE in fh.read(65536)

    @staticmethod
    def _convert_inline_xbrl(path: Path) -> DoclingDocument:
        # Docling routes inline XBRL (SEC 10-Q/10-K) to its XBRL backend, which needs arelle and keeps only
        # XBRL text blocks + facts, dropping the narrative. Parse it as HTML instead, the same way Docling
        # itself converts XBRL text blocks.
        from docling.backend.html_backend import HTMLDocumentBackend
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.document import InputDocument

        stream = BytesIO(strip_ixbrl_header(path.read_bytes()))
        in_doc = InputDocument(
            path_or_stream=stream, format=InputFormat.HTML, backend=HTMLDocumentBackend, filename=path.name
        )
        return HTMLDocumentBackend(in_doc=in_doc, path_or_stream=stream).convert()


def _title(blocks: list[Block], path: Path) -> str:
    title = next((b.text for b in blocks if b.kind is BlockKind.HEADING and b.level == 0), None)
    if title:
        return title
    # PDF layout models often label the title as a section header; a heading that opens the document is it.
    if blocks and blocks[0].kind is BlockKind.HEADING:
        return blocks[0].text
    return path.stem


def _docling_blocks(doc: DoclingDocument) -> list[Block]:
    # BODY + NOTES: speaker notes carry real content; FURNITURE (page headers/footers, <title>) is noise.
    layers = {ContentLayer.BODY, ContentLayer.NOTES}
    blocks: list[Block] = []
    for item, _depth in doc.iterate_items(with_groups=True, included_content_layers=layers):
        block = _to_block(item, doc)
        if block is not None and block.text.strip():
            blocks.append(block)
    return blocks


def _table_markdown(item: TableItem, doc: DoclingDocument) -> str:
    grid = item.data.grid
    if not grid:
        return item.export_to_markdown(doc=doc)
    return rows_to_markdown(clean_table([[cell.text for cell in row] for row in grid]))


def _to_block(item: NodeItem, doc: DoclingDocument) -> Block | None:  # noqa: PLR0911 - flat type dispatch
    if isinstance(item, GroupItem):
        # Spreadsheet sheets are groups; their names are the only section structure an XLSX has.
        return Block(BlockKind.HEADING, item.name, level=1) if item.label == GroupLabel.SHEET else None
    prov = getattr(item, "prov", None) or []
    page = prov[0].page_no if prov else None
    if isinstance(item, TableItem):
        return Block(BlockKind.TABLE, _table_markdown(item, doc), page=page)
    if not isinstance(item, TextItem):
        return None  # pictures etc.; their captions arrive as separate text items
    if isinstance(item, TitleItem):
        return Block(BlockKind.HEADING, item.text, level=0, page=page)
    if isinstance(item, SectionHeaderItem):
        return Block(BlockKind.HEADING, item.text, level=item.level, page=page)
    if isinstance(item, ListItem):
        return Block(BlockKind.LIST_ITEM, f"{item.marker or '-'} {item.text}", page=page)
    if isinstance(item, CodeItem):
        return Block(BlockKind.CODE, f"```\n{item.text}\n```", page=page)
    if item.content_layer == ContentLayer.NOTES:
        return Block(BlockKind.TEXT, f"Speaker notes: {item.text}", page=page)
    return Block(BlockKind.TEXT, item.text, page=page)
