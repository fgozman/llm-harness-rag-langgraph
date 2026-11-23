"""Map inline [n] markers to passage sources, and compose the cautious (partially verified) answer."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag.graph.state import Citation, Passage, SupportedClaim

NOT_FOUND = "Not found in the documents."
_SNIPPET_CHARS = 240
# [1], [1, 2], [1,2], [1][3] — not array indexing like a[0] (preceded by a word character).
_MARKER = re.compile(r"(?<!\w)\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


_MARKER_WITH_SPACE = re.compile(r"(\s*)" + _MARKER.pattern)


def _numbers(match: re.Match[str]) -> list[int]:
    return [int(part) for part in match.group(match.lastindex or 1).split(",")]


def citation_numbers(text: str) -> list[int]:
    """Cited passage numbers in order of first use."""
    seen: dict[int, None] = {}
    for match in _MARKER.finditer(text):
        for n in _numbers(match):
            seen.setdefault(n, None)
    return list(seen)


def strip_invalid_citations(text: str, passages: Sequence[Passage]) -> str:
    valid = {p.index for p in passages}

    def keep(match: re.Match[str]) -> str:
        numbers = [str(n) for n in _numbers(match) if n in valid]
        # A marker that disappears takes its leading space with it ("Made up [7]." -> "Made up.").
        return f"{match.group(1)}[{', '.join(numbers)}]" if numbers else ""

    return _MARKER_WITH_SPACE.sub(keep, text)


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _SNIPPET_CHARS else flat[:_SNIPPET_CHARS].rstrip() + "…"


def to_citation(p: Passage) -> Citation:
    return Citation(
        index=p.index,
        file_path=p.file_path,
        file_name=p.file_name,
        title=p.title,
        page_start=p.page_start,
        page_end=p.page_end,
        section=" > ".join(p.section_path),
        snippet=_snippet(p.snippet or p.text),
    )


def build_citations(text: str, passages: Sequence[Passage]) -> list[Citation]:
    by_index = {p.index: p for p in passages}
    return [to_citation(by_index[n]) for n in citation_numbers(text) if n in by_index]


def cautious_answer(
    *, supported: Sequence[SupportedClaim], unsupported: Sequence[str], passages: Sequence[Passage]
) -> tuple[str, list[Citation]]:
    """Answer that states what could and couldn't be verified after the regeneration cap was hit."""
    valid = {p.index for p in passages}
    verified = [c for c in supported if any(n in valid for n in c.citations)]
    if not verified:
        return f"{NOT_FOUND} I could not verify an answer from the retrieved passages.", []
    lines = ["I could not fully verify an answer against the documents.", "", "Verified:"]
    for claim in verified:
        marks = "".join(f"[{n}]" for n in claim.citations if n in valid)
        lines.append(f"- {claim.claim} {marks}")
    if unsupported:
        lines += ["", "Could not verify:", *(f"- {c}" for c in unsupported)]
    text = "\n".join(lines)
    return text, build_citations(text, passages)
