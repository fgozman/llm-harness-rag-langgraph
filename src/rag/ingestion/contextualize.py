"""Contextual retrieval: a short LLM-written blurb situating each chunk within its document.

Each request = cached prefix (system + document or document window) + the varying chunk. Chunks of the same
section share an identical prefix so prompt caching hits. Provider specifics live in `context_backends`;
large runs use a batch API when the backend has one (Anthropic Message Batches).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Protocol

from rag.config import IngestionSettings, ModelPricing
from rag.ingestion.models import Block, BlockKind, Chunk, ParsedDocument
from rag.prompts import contextualize as prompt

log = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]

# Above this many input tokens Haiku 5.5 switches to a 5x pricier tier.
_PRICE_TIER_LIMIT = 100_000


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.input + other.input,
            self.output + other.output,
            self.cache_read + other.cache_read,
            self.cache_write_5m + other.cache_write_5m,
            self.cache_write_1h + other.cache_write_1h,
        )

    @property
    def prompt_total(self) -> int:
        return self.input + self.cache_read + self.cache_write_5m + self.cache_write_1h


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    usage: TokenUsage
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class ContextPrompt:
    """Provider-neutral prompt: a stable, cacheable prefix (system + document) and the varying chunk."""

    system: str
    document: str
    chunk: str


@dataclass(frozen=True, slots=True)
class ContextRequest:
    custom_id: str
    prompt: ContextPrompt


@dataclass(frozen=True, slots=True)
class ContextualizeResult:
    chunks: list[list[Chunk]]
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    failed_chunks: int = 0
    cost_usd: float = 0.0


class Contextualizer(Protocol):
    def contextualize(self, docs: Sequence[tuple[ParsedDocument, list[Chunk]]]) -> ContextualizeResult: ...


class ContextBackend(Protocol):
    @property
    def supports_batch(self) -> bool: ...

    def complete(self, prompt: ContextPrompt) -> Completion: ...

    def complete_batch(self, requests: Sequence[ContextRequest]) -> dict[str, Completion]:
        """Return completions by custom_id; failed/expired requests are simply absent."""
        ...


# --- document context ------------------------------------------------------------------------------


def _render_sections(doc: ParsedDocument) -> list[tuple[tuple[str, ...], str]]:
    stack: list[tuple[int, str]] = []
    sections: list[tuple[tuple[str, ...], list[str]]] = [((), [])]
    for block in doc.blocks:
        if block.kind is BlockKind.HEADING:
            stack = [(lvl, txt) for lvl, txt in stack if lvl < block.level]
            if block.text != doc.title:
                stack.append((block.level, block.text))
            sections.append((tuple(t for _, t in stack), [_heading_md(block)]))
        else:
            sections[-1][1].append(block.text)
    return [(path, "\n\n".join(parts)) for path, parts in sections if parts]


def _heading_md(block: Block) -> str:
    return f"{'#' * min(block.level + 1, 6)} {block.text}"


def _truncate_words(text: str, count_tokens: TokenCounter, budget: int) -> str:
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:  # binary search for the longest prefix within budget
        mid = (lo + hi + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def _take_lines(lines: list[str], count_tokens: TokenCounter, budget: int) -> str:
    kept: list[str] = []
    used = 0
    for line in lines:
        cost = count_tokens(line)
        if used + cost > budget:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept)


def document_context(
    doc: ParsedDocument, section_path: tuple[str, ...], count_tokens: TokenCounter, budget: int
) -> tuple[str, bool]:
    """Full document if it fits; otherwise a heading outline plus the sections around the chunk's section."""
    sections = _render_sections(doc)
    full = "\n\n".join(text for _, text in sections)
    if count_tokens(full) <= budget:
        return full, False

    outline_lines = ["Outline:"] + ["  " * (len(p) - 1) + f"- {p[-1]}" for p, _ in sections if p]
    outline = _take_lines(outline_lines, count_tokens, budget // 3)
    remaining = budget - count_tokens(outline) - 2

    target = next((i for i, (p, _) in enumerate(sections) if p == section_path), None)
    if target is None:
        target = next((i for i, (p, _) in enumerate(sections) if p[:1] == section_path[:1]), 0)
    chosen = {target: _truncate_words(sections[target][1], count_tokens, remaining)}
    used = count_tokens(chosen[target])
    # Grow the window outward (after, then before) so the chunk sits near its middle.
    for offset in range(1, len(sections)):
        grew = False
        for idx in (target + offset, target - offset):
            if 0 <= idx < len(sections):
                cost = count_tokens(sections[idx][1])
                if used + cost <= remaining:
                    chosen[idx] = sections[idx][1]
                    used += cost
                    grew = True
        if not grew:
            break
    window = "\n\n".join(chosen[i] for i in sorted(chosen))
    return f"{outline}\n\n{window}", True


# --- contextualizer --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Job:
    doc_idx: int
    chunk_idx: int
    request: ContextRequest
    prefix_key: tuple[int, tuple[str, ...]]


class LLMContextualizer:
    def __init__(
        self,
        backend: ContextBackend,
        *,
        settings: IngestionSettings,
        count_tokens: TokenCounter,
        pricing: ModelPricing | None,
    ) -> None:
        self._backend = backend
        self._settings = settings
        self._count_tokens = count_tokens
        self._pricing = pricing

    @property
    def backend(self) -> ContextBackend:
        return self._backend

    def contextualize(self, docs: Sequence[tuple[ParsedDocument, list[Chunk]]]) -> ContextualizeResult:
        n_chunks = sum(len(chunks) for _, chunks in docs)
        if n_chunks == 0:
            return ContextualizeResult(chunks=[list(chunks) for _, chunks in docs])
        mode = self._settings.contextualize_mode
        wants_batch = mode == "batch" or (
            mode == "auto" and n_chunks >= self._settings.contextualize_batch_threshold
        )
        use_batch = wants_batch and self._backend.supports_batch
        jobs = self._jobs(docs)
        completions = self._run_batch(jobs) if use_batch else self._run_sync(jobs)
        return self._assemble(docs, jobs, completions, batch=use_batch)

    def _jobs(self, docs: Sequence[tuple[ParsedDocument, list[Chunk]]]) -> list[_Job]:
        budget = self._settings.contextualize_max_prompt_tokens
        jobs: list[_Job] = []
        prefixes: dict[tuple[int, tuple[str, ...]], str] = {}
        for di, (doc, chunks) in enumerate(docs):
            for ci, chunk in enumerate(chunks):
                key = (di, chunk.section_path)
                if key not in prefixes:
                    # Leave room for the chunk itself and the instructions inside the same prompt budget.
                    doc_budget = max(budget - chunk.token_count - 200, 1)
                    text, excerpt = document_context(doc, chunk.section_path, self._count_tokens, doc_budget)
                    prefixes[key] = prompt.document_block(doc.title, text, excerpt)
                request_prompt = ContextPrompt(prompt.SYSTEM, prefixes[key], prompt.chunk_block(chunk.text))
                jobs.append(_Job(di, ci, ContextRequest(f"d{di}-c{ci}", request_prompt), key))
        return jobs

    def _run_sync(self, jobs: list[_Job]) -> dict[str, Completion]:
        # Send one request per shared prefix first so it writes the cache; the rest then read from it.
        primers: dict[tuple[int, tuple[str, ...]], _Job] = {}
        for job in jobs:
            primers.setdefault(job.prefix_key, job)
        first_wave = list(primers.values())
        primer_ids = {j.request.custom_id for j in first_wave}
        second_wave = [j for j in jobs if j.request.custom_id not in primer_ids]
        results: dict[str, Completion] = {}
        with ThreadPoolExecutor(max_workers=self._settings.contextualize_concurrency) as pool:
            for wave in (first_wave, second_wave):
                for custom_id, completion in pool.map(self._safe_complete, wave):
                    if completion is not None:
                        results[custom_id] = completion
        return results

    def _safe_complete(self, job: _Job) -> tuple[str, Completion | None]:
        try:
            return job.request.custom_id, self._backend.complete(job.request.prompt)
        except Exception as exc:
            # One failed blurb shouldn't fail the file: the chunk is still indexed, just without context.
            log.warning("contextualization failed for %s: %s", job.request.custom_id, exc)
            return job.request.custom_id, None

    def _run_batch(self, jobs: list[_Job]) -> dict[str, Completion]:
        try:
            return self._backend.complete_batch([j.request for j in jobs])
        except Exception as exc:
            log.warning("contextualization batch failed: %s", exc)
            return {}

    def _assemble(
        self,
        docs: Sequence[tuple[ParsedDocument, list[Chunk]]],
        jobs: list[_Job],
        completions: dict[str, Completion],
        *,
        batch: bool,
    ) -> ContextualizeResult:
        out = [list(chunks) for _, chunks in docs]
        usage = TokenUsage()
        failed = 0
        for job in jobs:
            completion = completions.get(job.request.custom_id)
            text = _clean_context(completion) if completion else ""
            if completion is None or not text:
                failed += 1
                continue
            usage += completion.usage
            if completion.usage.prompt_total > _PRICE_TIER_LIMIT:
                log.warning(
                    "contextualization prompt exceeded 100K tokens (%d)", completion.usage.prompt_total
                )
            out[job.doc_idx][job.chunk_idx] = replace(out[job.doc_idx][job.chunk_idx], context=text)
        return ContextualizeResult(
            chunks=out,
            input_tokens=usage.input,
            output_tokens=usage.output,
            cache_read_tokens=usage.cache_read,
            cache_write_tokens=usage.cache_write_5m + usage.cache_write_1h,
            failed_chunks=failed,
            cost_usd=self._cost(usage, batch=batch),
        )

    def _cost(self, usage: TokenUsage, *, batch: bool) -> float:
        if self._pricing is None:
            return 0.0
        p = self._pricing
        dollars = (
            usage.input * p.input
            + usage.output * p.output
            + usage.cache_read * p.cache_read
            + usage.cache_write_5m * p.cache_write_5m
            + usage.cache_write_1h * p.cache_write_1h
        ) / 1_000_000
        return dollars * (0.5 if batch else 1.0)


def _clean_context(completion: Completion) -> str:
    text = completion.text.strip()
    if completion.truncated:
        # Hit max_tokens: drop the dangling partial sentence rather than index half a thought.
        end = text.rfind(".")
        text = text[: end + 1] if end > 0 else text
    return text
