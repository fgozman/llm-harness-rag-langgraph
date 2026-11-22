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
        result = self._converter.convert(path, raises_on_error=True)
        doc = result.document
        blocks = _docling_blocks(doc)
        return ParsedDocument(title=_title(blocks, path), blocks=blocks, parser="docling")


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


def _to_block(item: NodeItem, doc: DoclingDocument) -> Block | None:  # noqa: PLR0911 - flat type dispatch
    if isinstance(item, GroupItem):
        # Spreadsheet sheets are groups; their names are the only section structure an XLSX has.
        return Block(BlockKind.HEADING, item.name, level=1) if item.label == GroupLabel.SHEET else None
    prov = getattr(item, "prov", None) or []
    page = prov[0].page_no if prov else None
    if isinstance(item, TableItem):
        return Block(BlockKind.TABLE, item.export_to_markdown(doc=doc), page=page)
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
