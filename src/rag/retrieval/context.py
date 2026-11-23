"""Small-to-big expansion: retrieved child chunks -> deduplicated parent passages within a token budget."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from rag.ingestion.models import ParentSection
from rag.retrieval.types import ContextPassage, RankedChunk, payload_section

TokenCounter = Callable[[str], int]


@dataclass(slots=True)
class _Group:
    parent_id: str
    best: RankedChunk
    children: list[RankedChunk] = field(default_factory=list[RankedChunk])


def _group_by_parent(ranked: Sequence[RankedChunk]) -> list[_Group]:
    groups: dict[str, _Group] = {}
    for chunk in ranked:
        pid = str(chunk.payload["parent_id"])
        group = groups.setdefault(pid, _Group(pid, chunk))
        group.children.append(chunk)
    return list(groups.values())  # insertion order = rank of each group's best child


def _chunk_index(chunk: RankedChunk) -> int:
    value = chunk.payload.get("chunk_index", 0)
    return value if isinstance(value, int) else 0


def _page(chunk: RankedChunk, key: str) -> int | None:
    value = chunk.payload.get(key)
    return value if isinstance(value, int) else None


def _choose(
    group: _Group, parent: ParentSection | None, count_tokens: TokenCounter, remaining: int
) -> tuple[str, list[RankedChunk], bool] | None:
    """Prefer the whole parent; else the group's children in document order; else the best child alone."""
    if parent is not None and count_tokens(parent.text) <= remaining:
        return parent.text, group.children, True
    in_order = sorted(group.children, key=_chunk_index)
    joined = "\n\n".join(str(c.payload["text"]) for c in in_order)
    if len(in_order) > 1 and count_tokens(joined) <= remaining:
        return joined, in_order, False
    best_text = str(group.best.payload["text"])
    if count_tokens(best_text) <= remaining:
        return best_text, [group.best], False
    return None


def build_context(
    ranked: Sequence[RankedChunk],
    parents: Mapping[str, ParentSection],
    count_tokens: TokenCounter,
    budget: int,
) -> list[ContextPassage]:
    passages: list[ContextPassage] = []
    remaining = budget
    for group in _group_by_parent(ranked):
        parent = parents.get(group.parent_id)
        choice = _choose(group, parent, count_tokens, remaining)
        if choice is None:
            continue  # too big for what's left; a smaller, lower-ranked passage may still fit
        text, used, expanded = choice
        remaining -= count_tokens(text)
        best = group.best
        if expanded and parent is not None:
            pages = (parent.page_start, parent.page_end)
            section = parent.section_path
        else:
            starts = [p for c in used if (p := _page(c, "page_start")) is not None]
            ends = [p for c in used if (p := _page(c, "page_end")) is not None]
            pages = (min(starts) if starts else None, max(ends) if ends else None)
            section = payload_section(best.payload)
        passages.append(
            ContextPassage(
                index=len(passages) + 1,
                parent_id=group.parent_id,
                chunk_ids=tuple(c.chunk_id for c in used),
                text=text,
                expanded=expanded,
                score=best.score,
                file_path=str(best.payload.get("file_path", "")),
                file_name=str(best.payload.get("file_name", "")),
                title=str(best.payload.get("title", "")),
                section_path=section,
                page_start=pages[0],
                page_end=pages[1],
                snippet=str(best.payload.get("text", "")),
            )
        )
    return passages
