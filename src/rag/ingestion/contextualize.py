"""Contextual retrieval: a short LLM-written blurb situating each chunk within its document.

Each request = cached prefix (system + document or document window) + the varying chunk. Chunks of the same
section share an identical prefix so prompt caching hits. The chat model behind it is any LangChain
provider (see `context_backends`).
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Protocol

from rag.config import IngestionSettings, ModelPricing
from rag.ingestion.models import Block, BlockKind, Chunk, ParsedDocument
from rag.llm.base import LLMError
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
    cache_write: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.input + other.input,
            self.output + other.output,
            self.cache_read + other.cache_read,
            self.cache_write + other.cache_write,
        )

    @property
    def prompt_total(self) -> int:
        return self.input + self.cache_read + self.cache_write


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
    def complete(self, prompt: ContextPrompt) -> Completion: ...


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
        jobs = self._jobs(docs)
        return self._assemble(docs, jobs, self._run(jobs))

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

    def _run(self, jobs: list[_Job]) -> dict[str, Completion]:
        # Send one request per shared prefix first so it writes the cache; the rest then read from it.
        primers: dict[tuple[int, tuple[str, ...]], _Job] = {}
        for job in jobs:
            primers.setdefault(job.prefix_key, job)
        first_wave = list(primers.values())
        primer_ids = {j.request.custom_id for j in first_wave}
        second_wave = [j for j in jobs if j.request.custom_id not in primer_ids]
        results: dict[str, Completion] = {}
        pool = ThreadPoolExecutor(max_workers=self._settings.contextualize_concurrency)
        try:
            for wave in (first_wave, second_wave):
                # Thread pools don't inherit contextvars; a per-job copy keeps the tracing context (LLM spans
                # stay inside the ingest trace). One copy per job: a Context can't be entered concurrently.
                futures = [
                    pool.submit(contextvars.copy_context().run, self._safe_complete, job) for job in wave
                ]
                for future in futures:
                    custom_id, completion = future.result()
                    if completion is not None:
                        results[custom_id] = completion
        finally:
            # After a fatal error, don't send the requests still queued (they would all fail).
            pool.shutdown(cancel_futures=True)
        return results

    def _safe_complete(self, job: _Job) -> tuple[str, Completion | None]:
        try:
            return job.request.custom_id, self._backend.complete(job.request.prompt)
        except LLMError as exc:
            if not exc.transient:
                # Bad key/model/request, provider unreachable: fail the run instead of indexing every chunk
                # without context (the manifest would mark those files as done, never to be contextualized).
                raise
            # A rate limit/server error/timeout that outlasted the integration's retries shouldn't fail the
            # file: the chunk is still indexed, just without context.
            log.warning("contextualization failed for %s: %s", job.request.custom_id, exc)
            return job.request.custom_id, None

    def _assemble(
        self,
        docs: Sequence[tuple[ParsedDocument, list[Chunk]]],
        jobs: list[_Job],
        completions: dict[str, Completion],
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
            cache_write_tokens=usage.cache_write,
            failed_chunks=failed,
            cost_usd=self._cost(usage),
        )

    def _cost(self, usage: TokenUsage) -> float:
        if self._pricing is None:
            return 0.0
        p = self._pricing
        # Cache writes at the 5-minute rate: the cache marker requests the default TTL.
        return (
            usage.input * p.input
            + usage.output * p.output
            + usage.cache_read * p.cache_read
            + usage.cache_write * p.cache_write_5m
        ) / 1_000_000


def _clean_context(completion: Completion) -> str:
    text = completion.text.strip()
    if completion.truncated:
        # Hit max_tokens: drop the dangling partial sentence rather than index half a thought.
        end = text.rfind(".")
        text = text[: end + 1] if end > 0 else text
    return text
