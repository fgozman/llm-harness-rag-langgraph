"""Golden evaluation set: LLM-generated, mechanically and LLM-verified, then human-reviewed.

Candidates are sampled per category from the indexed corpus (seeded). Every generated item must quote its
evidence verbatim (checked deterministically) and pass an LLM judge (self-contained, supported, unambiguous).
Unanswerable questions are kept only if retrieval over the real index doesn't answer them.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from rag.llm.base import LLM, Message
from rag.prompts import golden as prompt
from rag.retrieval.types import payload_section

Category = Literal[
    "exact_figure",
    "temporal",
    "comparison",
    "table_lookup",
    "release_notes",
    "factual",
    "unanswerable",
    "canary",
    "canary_unanswerable",
]
DEFAULT_QUOTAS: dict[str, int] = {
    "exact_figure": 14,
    "temporal": 10,
    "comparison": 8,
    "table_lookup": 12,
    "release_notes": 8,
    "factual": 10,
}
_SEC_NAME = re.compile(r"^([A-Z]+)_(10-Q|8-K)_.*filed-(\d{4}-\d{2})-\d{2}")
_ARXIV_NAME = re.compile(r"^\d{4}\.\d{4,5}\.pdf$")
_WORD = re.compile(r"[a-z0-9$%.,]+")
_CUTOFF_MONTH = "2026-07"

# --- records -----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    file_name: str
    file_path: str
    title: str
    section: str
    page_start: int | None
    page_end: int | None
    text: str
    token_count: int

    def describe(self) -> str:
        pages = f", p. {self.page_start}" if self.page_start else ""
        section = f", section: {self.section}" if self.section else ""
        return f"{self.title} (file {self.file_name}{pages}{section})".replace('"', "'")

    @property
    def location(self) -> str:
        pages = f"p. {self.page_start}" if self.page_start else ""
        return ", ".join(p for p in (self.section, pages) if p)


class SourceRef(BaseModel):
    file_name: str
    file_path: str
    chunk_id: str | None
    location: str


class GoldenItem(BaseModel):
    id: str
    category: Category
    question: str
    reference_answer: str
    answerable: bool
    sources: list[SourceRef]
    supporting_quote: str
    reviewed: bool = False
    generator: str
    verification: dict[str, Any]


# --- LLM schemas -------------------------------------------------------------------------------------


class GeneratedQA(BaseModel):
    question: str
    answer: str
    supporting_quote: str = Field(description="Verbatim excerpt from a passage that supports the answer.")
    source_passages: list[int] = Field(description="Indexes of the passages needed to answer.")
    usable: bool = Field(description="False if no good question of the requested kind is possible.")


class QAVerdict(BaseModel):
    self_contained: bool
    supported: bool
    unambiguous: bool
    note: str


class UnanswerableQuestion(BaseModel):
    question: str
    why: str = Field(description="Why the collection cannot answer it.")


class UnanswerableBatch(BaseModel):
    questions: list[UnanswerableQuestion]


class AnswerabilityVerdict(BaseModel):
    answerable: bool
    note: str


# --- deterministic helpers ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    # Table markup and whitespace differ between how a model copies a row and how it was rendered.
    return " ".join(re.sub(r"[|*`#]", " ", text).split()).lower()


def quote_in_text(quote: str, text: str) -> bool:
    needle = _normalize(quote)
    return bool(needle) and needle in _normalize(text)


def sec_info(file_name: str) -> tuple[str, str, str] | None:
    """(company, form, filing month) for SEC file names written by the corpus builder."""
    match = _SEC_NAME.match(file_name)
    return (match[1], match[2], match[3]) if match else None


def is_table(text: str) -> bool:
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("|")) >= 3


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def best_match(
    target: ChunkRecord, candidates: Sequence[ChunkRecord], *, min_similarity: float
) -> ChunkRecord | None:
    scored = [(similarity(target.text, c.text), c) for c in candidates]
    best = max(scored, key=lambda s: s[0], default=None)
    return best[1] if best is not None and best[0] >= min_similarity else None


def _spread[T](items: Sequence[T], key: Callable[[T], str]) -> list[T]:
    """Interleave groups so consecutive candidates come from different documents/companies."""
    groups: dict[str, list[T]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    out: list[T] = []
    while any(groups.values()):
        for name in list(groups):
            if groups[name]:
                out.append(groups[name].pop(0))
    return out


def _shuffled[T](items: Iterable[T], rng: random.Random) -> list[T]:
    out = list(items)
    rng.shuffle(out)
    return out


def plan_candidates(
    pool: Sequence[ChunkRecord], rng: random.Random
) -> dict[str, list[tuple[ChunkRecord, ...]]]:
    """Ordered candidate chunk tuples per category (deterministic for a given seed)."""
    by_filing: dict[tuple[str, str], list[ChunkRecord]] = defaultdict(list)
    for c in pool:
        info = sec_info(c.file_name)
        if info and info[1] == "10-Q" and is_table(c.text):
            month = "pre" if info[2] < _CUTOFF_MONTH else info[2]
            by_filing[(info[0], month)].append(c)

    def company(c: ChunkRecord) -> str:
        info = sec_info(c.file_name)
        return info[0] if info else c.file_name

    figures = [
        c
        for c in pool
        if (info := sec_info(c.file_name)) and info[2] >= _CUTOFF_MONTH and is_table(c.text) and "$" in c.text
    ]
    temporal: list[tuple[ChunkRecord, ...]] = []
    comparison: list[tuple[ChunkRecord, ...]] = []
    companies = sorted({key[0] for key in by_filing})
    for name in companies:
        nov, aug, pre = (by_filing.get((name, m), []) for m in ("2026-11", "2026-08", "pre"))
        for target in _shuffled(nov, rng):
            if (match := best_match(target, aug, min_similarity=0.3)) is not None:
                comparison.append((target, match))
            if (match := best_match(target, pre, min_similarity=0.3)) is not None:
                temporal.append((target, match))
        for target in _shuffled(aug, rng):
            if (match := best_match(target, nov, min_similarity=0.3)) is not None:
                temporal.append((target, match))

    def doc_kind(c: ChunkRecord) -> str:
        if "release_notes" in c.file_path:
            return "release_notes"
        if "federal_register" in c.file_path:
            return "federal_register"
        if c.file_name.startswith("gao-"):
            return "gao"
        if _ARXIV_NAME.match(c.file_name):
            return "arxiv"
        return "other"

    tables = [
        c for c in pool if doc_kind(c) in ("arxiv", "gao") and is_table(c.text) and re.search(r"\d", c.text)
    ]
    notes = [c for c in pool if doc_kind(c) == "release_notes" and c.token_count >= 40]
    facts = [
        c
        for c in pool
        if doc_kind(c) in ("federal_register", "gao", "arxiv")
        and not is_table(c.text)
        and c.token_count >= 150
    ]
    return {
        "exact_figure": [(c,) for c in _spread(_shuffled(figures, rng), company)],
        "temporal": _spread(_shuffled(temporal, rng), lambda p: company(p[0])),
        "comparison": _spread(_shuffled(comparison, rng), lambda p: company(p[0])),
        "table_lookup": [(c,) for c in _spread(_shuffled(tables, rng), lambda c: c.file_name)],
        "release_notes": [(c,) for c in _spread(_shuffled(notes, rng), lambda c: c.file_name)],
        "factual": [(c,) for c in _spread(_shuffled(facts, rng), doc_kind)],
    }


def load_pool(client: QdrantClient, collection: str) -> list[ChunkRecord]:
    fields = [
        "file_name",
        "file_path",
        "title",
        "section_path",
        "page_start",
        "page_end",
        "text",
        "token_count",
    ]
    records: list[ChunkRecord] = []
    offset: Any = None
    while True:
        points, offset = client.scroll(collection, limit=1000, offset=offset, with_payload=fields)
        for p in points:
            d: dict[str, Any] = dict(p.payload or {})
            records.append(
                ChunkRecord(
                    chunk_id=str(p.id),
                    file_name=str(d.get("file_name", "")),
                    file_path=str(d.get("file_path", "")),
                    title=str(d.get("title", "")),
                    section=" > ".join(payload_section(d)),
                    page_start=d.get("page_start") if isinstance(d.get("page_start"), int) else None,
                    page_end=d.get("page_end") if isinstance(d.get("page_end"), int) else None,
                    text=str(d.get("text", "")),
                    token_count=int(d.get("token_count", 0) or 0),
                )
            )
        if offset is None:
            # Stable order regardless of Qdrant's internal ordering, so sampling is reproducible.
            return sorted(records, key=lambda r: r.chunk_id)


def coverage_summary(pool: Sequence[ChunkRecord]) -> str:
    """What the collection contains, for writing plausible-but-unanswerable questions."""
    filings = sorted({f"{i[0]} {i[1]} filed {i[2]}" for c in pool if (i := sec_info(c.file_name))})
    titles: dict[str, set[str]] = defaultdict(set)
    for c in pool:
        if sec_info(c.file_name) is None:
            folder = Path(c.file_path).parent.name
            titles[folder].add(c.title[:120])
    lines = ["SEC filings: " + "; ".join(filings)]
    for folder, names in sorted(titles.items()):
        lines.append(f"{folder}: " + "; ".join(sorted(names)[:30]))
    return "\n".join(lines)


# --- generation --------------------------------------------------------------------------------------


def _user(text: str) -> list[Message]:
    return [Message("user", text)]


def _source(c: ChunkRecord) -> SourceRef:
    return SourceRef(file_name=c.file_name, file_path=c.file_path, chunk_id=c.chunk_id, location=c.location)


class GoldenGenerator:
    def __init__(self, llm: LLM, *, generator_name: str = "") -> None:
        self._llm = llm
        self._name = generator_name

    def generate_category(
        self, category: str, candidates: Sequence[tuple[ChunkRecord, ...]], quota: int
    ) -> list[GoldenItem]:
        items: list[GoldenItem] = []
        for chunks in candidates:
            if len(items) >= quota:
                break
            item = self._try(category, chunks, len(items) + 1)
            if item is not None:
                items.append(item)
        return items

    def _try(self, category: str, chunks: tuple[ChunkRecord, ...], number: int) -> GoldenItem | None:
        passages = [(c.describe(), c.text) for c in chunks]
        qa = self._llm.structured(
            "golden",
            system=prompt.SYSTEM,
            messages=_user(prompt.generation_message(category, passages)),
            schema=GeneratedQA,
        )
        if not qa.usable or not qa.question.strip():
            return None
        used = [chunks[i - 1] for i in qa.source_passages if 1 <= i <= len(chunks)] or [chunks[0]]
        if not any(quote_in_text(qa.supporting_quote, c.text) for c in used):
            return None  # the model's evidence isn't in the source: don't trust the reference answer
        verdict = self._llm.structured(
            "judge",
            system=prompt.VERIFY_SYSTEM,
            messages=_user(
                prompt.verification_message(qa.question, qa.answer, [(c.describe(), c.text) for c in used])
            ),
            schema=QAVerdict,
        )
        if not (verdict.self_contained and verdict.supported and verdict.unambiguous):
            return None
        return GoldenItem(
            id=f"{category}-{number:03d}",
            category=category,  # type: ignore[arg-type]  # validated by pydantic
            question=qa.question.strip(),
            reference_answer=qa.answer.strip(),
            answerable=True,
            sources=[_source(c) for c in used],
            supporting_quote=qa.supporting_quote,
            generator=self._name,
            verification={"quote_found": True, "judge": verdict.model_dump()},
        )

    def unanswerable(
        self, coverage: str, count: int, *, retrieve: Callable[[str], list[tuple[str, str]]]
    ) -> list[GoldenItem]:
        batch = self._llm.structured(
            "golden",
            system=prompt.UNANSWERABLE_SYSTEM,
            # Ask for extras: some will turn out to be answerable and get dropped.
            messages=_user(prompt.unanswerable_message(coverage, count + max(2, count // 2))),
            schema=UnanswerableBatch,
        )
        items: list[GoldenItem] = []
        for candidate in batch.questions:
            if len(items) >= count:
                break
            verdict = self._llm.structured(
                "judge",
                system=prompt.ANSWERABILITY_SYSTEM,
                messages=_user(
                    prompt.answerability_message(candidate.question, retrieve(candidate.question))
                ),
                schema=AnswerabilityVerdict,
            )
            if verdict.answerable:
                continue
            items.append(
                GoldenItem(
                    id=f"unanswerable-{len(items) + 1:03d}",
                    category="unanswerable",
                    question=candidate.question.strip(),
                    reference_answer="Not found in the documents.",
                    answerable=False,
                    sources=[],
                    supporting_quote="",
                    generator=self._name,
                    verification={"why": candidate.why, "retrieval_judge": verdict.model_dump()},
                )
            )
        return items


def canary_items(facts_path: Path, *, corpus_dir: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for line in facts_path.read_text().splitlines():
        fact = json.loads(line)
        answerable = bool(fact["answerable"])
        file_name = Path(fact["file"]).name
        sources = (
            [
                SourceRef(
                    file_name=file_name,
                    file_path=str(corpus_dir / fact["file"]),
                    chunk_id=None,
                    location=fact["location"],
                )
            ]
            if answerable
            else []
        )
        items.append(
            GoldenItem(
                id=f"canary-{fact['id']}",
                category="canary" if answerable else "canary_unanswerable",
                question=fact["question"],
                reference_answer=fact["answer"] if answerable else "Not found in the documents.",
                answerable=answerable,
                sources=sources,
                supporting_quote=fact["answer"],
                generator="canary (hand-written fiction)",
                verification={"canary": True},
            )
        )
    return items


def save_golden(items: Sequence[GoldenItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(item.model_dump_json() + "\n" for item in items))


def load_golden(path: Path) -> list[GoldenItem]:
    return [GoldenItem.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
