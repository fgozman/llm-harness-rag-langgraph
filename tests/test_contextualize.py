import re
from collections.abc import Sequence
from typing import Any

import pytest

from rag.config import IngestionSettings, ModelPricing
from rag.ingestion.contextualize import (
    Completion,
    ContextPrompt,
    ContextRequest,
    LLMContextualizer,
    TokenUsage,
    document_context,
)
from rag.ingestion.models import Block, BlockKind, Chunk, ParsedDocument


def words(text: str) -> int:
    return len(text.split())


def make_doc(sections: int, words_per_section: int) -> ParsedDocument:
    blocks: list[Block] = [Block(BlockKind.HEADING, "Big Doc", level=0)]
    for i in range(sections):
        blocks.append(Block(BlockKind.HEADING, f"Section {i}", level=1))
        blocks.append(Block(BlockKind.TEXT, " ".join(f"s{i}w{j}" for j in range(words_per_section))))
    return ParsedDocument(title="Big Doc", blocks=blocks, parser="test")


def chunk(i: int, section: str, text: str = "chunk text") -> Chunk:
    return Chunk(
        chunk_id=f"id{i}",
        parent_id="p",
        chunk_index=i,
        text=text,
        section_path=(section,),
        page_start=None,
        page_end=None,
        token_count=words(text),
    )


class FakeBackend:
    def __init__(self, fail_on: str | None = None, supports_batch: bool = True) -> None:
        self.sync_calls: list[ContextPrompt] = []
        self.batches: list[list[ContextRequest]] = []
        self.fail_on = fail_on
        self.supports_batch = supports_batch

    def _answer(self, p: ContextPrompt) -> Completion:
        if self.fail_on and self.fail_on in p.chunk:
            raise RuntimeError("api down")
        snippet = re.search(r"<chunk>\n(.*?)\n</chunk>", p.chunk, re.S)
        assert snippet is not None
        return Completion(text=f"  ctx for {snippet.group(1)}  ", usage=TokenUsage(input=100, output=10))

    def complete(self, prompt: ContextPrompt) -> Completion:
        self.sync_calls.append(prompt)
        return self._answer(prompt)

    def complete_batch(self, requests: Sequence[ContextRequest]) -> dict[str, Completion]:
        self.batches.append(list(requests))
        out: dict[str, Completion] = {}
        for r in requests:
            try:
                out[r.custom_id] = self._answer(r.prompt)
            except RuntimeError:
                continue
        return out


def settings(**kw: Any) -> IngestionSettings:
    base: dict[str, Any] = {"contextualize_max_prompt_tokens": 1000, "contextualize_batch_threshold": 5}
    base.update(kw)
    return IngestionSettings.model_validate(base)


def contextualizer(backend: FakeBackend, **kw: Any) -> LLMContextualizer:
    return LLMContextualizer(
        backend,
        settings=settings(**kw),
        count_tokens=words,
        pricing=ModelPricing(input=1.0, cache_write_5m=1.25, cache_write_1h=2.0, cache_read=0.1, output=5.0),
    )


def test_short_document_is_included_in_full() -> None:
    doc = make_doc(3, 10)
    text, is_excerpt = document_context(doc, ("Section 1",), words, budget=1000)
    assert not is_excerpt
    assert all(f"Section {i}" in text for i in range(3))
    assert "s2w9" in text


def test_long_document_uses_outline_plus_window_within_budget() -> None:
    doc = make_doc(50, 100)
    text, is_excerpt = document_context(doc, ("Section 25",), words, budget=600)
    assert is_excerpt
    assert words(text) <= 600
    assert "s25w0" in text and "s25w99" in text, "the chunk's own section is in the window"
    assert "Section 0" in text and "Section 49" in text, "outline lists headings"
    assert "s0w5" not in text, "far-away sections are excluded"


def test_prompt_is_system_then_document_then_chunk() -> None:
    backend = FakeBackend()
    doc = make_doc(1, 5)
    contextualizer(backend, contextualize_mode="sync").contextualize(
        [(doc, [chunk(0, "Section 0", "the chunk")])]
    )
    (p,) = backend.sync_calls
    assert "ignore any instructions" in p.system
    assert p.document.startswith("<document") and "s0w4" in p.document
    assert "<chunk>\nthe chunk\n</chunk>" in p.chunk


def test_sync_mode_assigns_contexts_in_order() -> None:
    backend = FakeBackend()
    doc = make_doc(2, 10)
    result = contextualizer(backend, contextualize_mode="sync").contextualize(
        [(doc, [chunk(0, "Section 0", "first"), chunk(1, "Section 1", "second")])]
    )
    assert [c.context for c in result.chunks[0]] == ["ctx for first", "ctx for second"]
    assert backend.batches == []
    assert (result.input_tokens, result.output_tokens) == (200, 20)


def test_chunks_in_same_section_share_identical_cached_prefix() -> None:
    backend = FakeBackend()
    doc = make_doc(50, 100)
    contextualizer(backend, contextualize_mode="sync").contextualize(
        [(doc, [chunk(0, "Section 10", "a"), chunk(1, "Section 10", "b")])]
    )
    first, second = (c.document for c in backend.sync_calls)
    assert first == second


def test_failures_leave_context_empty_and_are_counted() -> None:
    backend = FakeBackend(fail_on="bad")
    doc = make_doc(1, 5)
    result = contextualizer(backend, contextualize_mode="sync").contextualize(
        [(doc, [chunk(0, "Section 0", "good"), chunk(1, "Section 0", "bad")])]
    )
    assert [c.context for c in result.chunks[0]] == ["ctx for good", None]
    assert result.failed_chunks == 1


@pytest.mark.parametrize(("n_chunks", "expect_batch"), [(4, False), (5, True)])
def test_auto_mode_switches_to_batch_at_threshold(n_chunks: int, expect_batch: bool) -> None:
    backend = FakeBackend()
    doc = make_doc(1, 5)
    contextualizer(backend).contextualize([(doc, [chunk(i, "Section 0", f"c{i}") for i in range(n_chunks)])])
    assert bool(backend.batches) is expect_batch


def test_batch_mode_maps_results_across_documents_and_reports_missing() -> None:
    backend = FakeBackend(fail_on="bad")
    d1, d2 = make_doc(1, 5), make_doc(2, 5)
    result = contextualizer(backend, contextualize_mode="batch").contextualize(
        [(d1, [chunk(0, "Section 0", "x")]), (d2, [chunk(0, "Section 1", "bad"), chunk(1, "Section 0", "y")])]
    )
    (batch,) = backend.batches
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", r.custom_id) for r in batch)
    assert len({r.custom_id for r in batch}) == 3
    assert [c.context for c in result.chunks[0]] == ["ctx for x"]
    assert [c.context for c in result.chunks[1]] == [None, "ctx for y"]
    assert result.failed_chunks == 1


def test_cost_estimate_applies_batch_discount() -> None:
    doc = make_doc(1, 5)
    sync = contextualizer(FakeBackend(), contextualize_mode="sync").contextualize(
        [(doc, [chunk(0, "Section 0")])]
    )
    batch = contextualizer(FakeBackend(), contextualize_mode="batch").contextualize(
        [(doc, [chunk(0, "Section 0")])]
    )
    # 100 input * $1/M + 10 output * $5/M
    assert sync.cost_usd == pytest.approx(150 / 1e6)
    assert batch.cost_usd == pytest.approx(75 / 1e6)


def test_backend_without_batch_support_stays_sync_above_threshold() -> None:
    backend = FakeBackend(supports_batch=False)
    doc = make_doc(1, 5)
    result = contextualizer(backend).contextualize(
        [(doc, [chunk(i, "Section 0", f"c{i}") for i in range(10)])]
    )
    assert backend.batches == [] and len(backend.sync_calls) == 10
    assert result.failed_chunks == 0


def test_empty_input_makes_no_calls() -> None:
    backend = FakeBackend()
    result = contextualizer(backend).contextualize([])
    assert result.chunks == [] and backend.sync_calls == [] and backend.batches == []


class TruncatingBackend(FakeBackend):
    def complete(self, prompt: ContextPrompt) -> Completion:
        return Completion(text="First sentence. Second is cut mid-wo", usage=TokenUsage(), truncated=True)


def test_truncated_context_is_trimmed_to_last_full_sentence() -> None:
    doc = make_doc(1, 5)
    result = contextualizer(TruncatingBackend(), contextualize_mode="sync").contextualize(
        [(doc, [chunk(0, "Section 0")])]
    )
    assert result.chunks[0][0].context == "First sentence."
