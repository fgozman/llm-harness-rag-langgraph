"""Data model shared by ingestion stages: parser output, chunks, and parents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class BlockKind(StrEnum):
    HEADING = "heading"
    TEXT = "text"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"


@dataclass(frozen=True, slots=True)
class Block:
    """One structural element of a parsed document, in reading order.

    `level` is meaningful only for headings: 0 is a document/slide title, 1+ are section levels.
    Tables carry Markdown text so the chunker can split them on row boundaries.
    """

    kind: BlockKind
    text: str
    level: int = 0
    page: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    title: str
    blocks: list[Block]
    parser: str


@dataclass(frozen=True, slots=True)
class ParentSection:
    """The larger unit handed to the LLM after a child chunk is retrieved ("small-to-big")."""

    parent_id: str
    text: str
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True, slots=True)
class Chunk:
    """A small retrieval unit. `text` is the original content shown to users and the LLM."""

    chunk_id: str
    parent_id: str
    chunk_index: int
    text: str
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    token_count: int
    context: str | None = field(default=None)

    def embedding_text(self, title: str) -> str:
        # Indexed text = contextual blurb + location header + original text; display text stays original.
        location = " > ".join((title, *self.section_path))
        parts = [p for p in (self.context, location, self.text) if p]
        return "\n\n".join(parts)
