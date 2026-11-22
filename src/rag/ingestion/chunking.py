"""Structure-aware parent/child chunking.

Pipeline: blocks -> sections (by heading) -> atomic units (sentences / table-row groups) -> parents (packed
sections, <= parent budget) -> children (packed units, <= child budget, with sentence-level overlap).
Children never cross section or parent boundaries, and tables are only ever split between rows.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from rag.config import ChunkingSettings
from rag.ingestion.models import Block, BlockKind, Chunk, ParentSection, ParsedDocument

TokenCounter = Callable[[str], int]

# Stable namespace so chunk/parent IDs are reproducible across runs and machines.
_ID_NAMESPACE = uuid.UUID("6f1c2d4e-8a51-4b8e-9f3a-2c7d1e5b9a40")
_SENTENCE_START = re.compile(r"(?<=[.!?])\s+(?=\S)")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}")


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    tokens: int
    page: int | None
    block_idx: int
    is_table: bool


@dataclass(slots=True)
class _Section:
    path: tuple[str, ...]
    units: list[_Unit] = field(default_factory=list[_Unit])


def make_id(doc_id: str, key: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{doc_id}:{key}"))


def chunk_document(
    doc: ParsedDocument, doc_id: str, count_tokens: TokenCounter, cfg: ChunkingSettings
) -> tuple[list[ParentSection], list[Chunk]]:
    sections = _build_sections(doc, count_tokens, cfg)
    parents: list[ParentSection] = []
    chunks: list[Chunk] = []
    for group in _pack_parents(sections, count_tokens, cfg):
        parent_id = make_id(doc_id, f"parent:{len(parents)}")
        parents.append(_make_parent(parent_id, group))
        for section in group:
            for units in _pack_children(section.units, count_tokens, cfg):
                text = _join(units)
                start, end = _page_range(units)
                chunks.append(
                    Chunk(
                        chunk_id=make_id(doc_id, f"chunk:{len(chunks)}"),
                        parent_id=parent_id,
                        chunk_index=len(chunks),
                        text=text,
                        section_path=section.path,
                        page_start=start,
                        page_end=end,
                        token_count=count_tokens(text),
                    )
                )
    return parents, chunks


# --- sections and units ---------------------------------------------------------------------------


def _build_sections(doc: ParsedDocument, count_tokens: TokenCounter, cfg: ChunkingSettings) -> list[_Section]:
    stack: list[tuple[int, str]] = []
    sections: list[_Section] = [_Section(path=())]
    for idx, block in enumerate(doc.blocks):
        if block.kind is BlockKind.HEADING:
            stack = [(lvl, txt) for lvl, txt in stack if lvl < block.level]
            # The document title is supplied separately as metadata; repeating it in every path adds noise.
            if block.text != doc.title:
                stack.append((block.level, block.text))
            sections.append(_Section(path=tuple(txt for _, txt in stack)))
            continue
        sections[-1].units.extend(_block_units(block, idx, count_tokens, cfg))
    return [s for s in sections if s.units]


def _block_units(block: Block, idx: int, count_tokens: TokenCounter, cfg: ChunkingSettings) -> list[_Unit]:
    tokens = count_tokens(block.text)
    is_table = block.kind is BlockKind.TABLE
    if tokens <= cfg.child_max_tokens:
        return [_Unit(block.text, tokens, block.page, idx, is_table)]
    pieces = (
        _split_table(block.text, count_tokens, cfg.child_target_tokens)
        if is_table
        else _split_text(block.text, count_tokens, cfg.child_target_tokens)
    )
    return [_Unit(p, count_tokens(p), block.page, idx, is_table) for p in pieces]


def _split_table(text: str, count_tokens: TokenCounter, budget: int) -> list[str]:
    lines = text.splitlines()
    header_len = 2 if len(lines) > 1 and _TABLE_SEPARATOR.match(lines[1]) else 1
    header, rows = lines[:header_len], lines[header_len:]
    header_tokens = count_tokens("\n".join(header))
    pieces: list[list[str]] = []
    current: list[str] = []
    current_tokens = header_tokens
    for row in rows:
        row_tokens = count_tokens(row)
        if current and current_tokens + row_tokens > budget:
            pieces.append(current)
            current, current_tokens = [], header_tokens
        # A row larger than the budget still goes in whole: cutting a row would corrupt the table.
        current.append(row)
        current_tokens += row_tokens
    if current:
        pieces.append(current)
    return ["\n".join(header + piece) for piece in pieces]


def _split_text(text: str, count_tokens: TokenCounter, budget: int) -> list[str]:
    pieces: list[str] = []
    for sentence in _SENTENCE_START.split(text):
        if count_tokens(sentence) <= budget:
            pieces.append(sentence)
        else:
            pieces.extend(_split_words(sentence, count_tokens, budget))
    return pieces


def _split_words(text: str, count_tokens: TokenCounter, budget: int) -> Iterator[str]:
    current: list[str] = []
    for word in text.split():
        if current and count_tokens(" ".join([*current, word])) > budget:
            yield " ".join(current)
            current = []
        current.append(word)
    if current:
        yield " ".join(current)


# --- parents ---------------------------------------------------------------------------------------


def _heading_line(path: tuple[str, ...]) -> str:
    return f"## {' > '.join(path)}" if path else ""


def _split_section(section: _Section, count_tokens: TokenCounter, budget: int) -> list[_Section]:
    heading_tokens = count_tokens(_heading_line(section.path))
    pieces: list[_Section] = []
    current = _Section(section.path)
    tokens = heading_tokens
    for unit in section.units:
        if current.units and tokens + unit.tokens > budget:
            pieces.append(current)
            current, tokens = _Section(section.path), heading_tokens
        current.units.append(unit)
        tokens += unit.tokens
    pieces.append(current)
    return pieces


def _section_tokens(section: _Section, count_tokens: TokenCounter) -> int:
    return count_tokens(_heading_line(section.path)) + sum(u.tokens for u in section.units)


def _pack_parents(
    sections: list[_Section], count_tokens: TokenCounter, cfg: ChunkingSettings
) -> list[list[_Section]]:
    budget = cfg.parent_max_tokens
    pieces = [p for s in sections for p in _split_section(s, count_tokens, budget)]
    groups: list[list[_Section]] = []
    group_tokens = 0
    for piece in pieces:
        tokens = _section_tokens(piece, count_tokens)
        # Pack only siblings under the same top-level heading so a parent stays topically coherent.
        same_top = bool(groups) and groups[-1][0].path[:1] == piece.path[:1]
        if same_top and group_tokens + tokens <= budget:
            groups[-1].append(piece)
            group_tokens += tokens
        else:
            groups.append([piece])
            group_tokens = tokens
    return groups


def _make_parent(parent_id: str, group: list[_Section]) -> ParentSection:
    parts = [p for s in group for p in (_heading_line(s.path), _join(s.units)) if p]
    start, end = _page_range([u for s in group for u in s.units])
    return ParentSection(
        parent_id=parent_id,
        text="\n\n".join(parts),
        section_path=group[0].path,
        page_start=start,
        page_end=end,
    )


# --- children --------------------------------------------------------------------------------------


def _pack_children(
    units: list[_Unit], count_tokens: TokenCounter, cfg: ChunkingSettings
) -> list[list[_Unit]]:
    target, limit = cfg.child_target_tokens, cfg.child_max_tokens
    overlap_budget = int(cfg.overlap_ratio * target)
    chunks: list[list[_Unit]] = []
    current: list[_Unit] = []
    tokens = 0
    has_content = False
    for unit in units:
        if has_content and tokens + unit.tokens > target:
            chunks.append(current)
            # Tables get no carried-over prefix: each table piece already repeats its header row.
            overlap = None if unit.is_table else _overlap(current[-1], count_tokens, overlap_budget)
            current, tokens, has_content = ([overlap], overlap.tokens, False) if overlap else ([], 0, False)
        if not has_content and current and tokens + unit.tokens > limit:
            current, tokens = [], 0
        current.append(unit)
        tokens += unit.tokens
        has_content = True
    if has_content:
        chunks.append(current)
    return chunks


def _overlap(unit: _Unit, count_tokens: TokenCounter, budget: int) -> _Unit | None:
    if unit.is_table or budget <= 0:
        return None
    # Take the longest sentence-aligned suffix within budget, as a substring so it stays verbatim.
    for match in _SENTENCE_START.finditer(unit.text):
        suffix = unit.text[match.end() :]
        tokens = count_tokens(suffix)
        if tokens <= budget:
            return _Unit(suffix, tokens, unit.page, unit.block_idx, is_table=False)
    if count_tokens(unit.text) <= budget and _SENTENCE_START.search(unit.text) is None:
        return _Unit(unit.text, count_tokens(unit.text), unit.page, unit.block_idx, is_table=False)
    return None


def _join(units: list[_Unit]) -> str:
    # Pieces of the same source block (sentences) rejoin with a space; separate blocks with a blank line.
    out = ""
    for i, unit in enumerate(units):
        if i:
            out += " " if unit.block_idx == units[i - 1].block_idx else "\n\n"
        out += unit.text
    return out


def _page_range(units: list[_Unit]) -> tuple[int | None, int | None]:
    pages = [u.page for u in units if u.page is not None]
    return (min(pages), max(pages)) if pages else (None, None)
