import re
from typing import Any

import pytest

from rag.config import IngestionSettings, ModelPricing
from rag.ingestion.contextualize import (
    Completion,
    ContextPrompt,
    LLMContextualizer,
    TokenUsage,
    document_context,
)
from rag.ingestion.models import Block, BlockKind, Chunk, ParsedDocument
from rag.llm.base import LLMError


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
    def __init__(self, fail_on: str | None = None, error: Exception | None = None) -> None:
        self.sync_calls: list[ContextPrompt] = []
        self.fail_on = fail_on
        self.error = error

    def complete(self, prompt: ContextPrompt) -> Completion:
        self.sync_calls.append(prompt)
        if self.error is not None:
            raise self.error
        if self.fail_on and self.fail_on in prompt.chunk:
            raise LLMError("overloaded", status_code=529, transient=True)
        snippet = re.search(r"<chunk>\n(.*?)\n</chunk>", prompt.chunk, re.S)
        assert snippet is not None
        return Completion(text=f"  ctx for {snippet.group(1)}  ", usage=TokenUsage(input=100, output=10))


def settings(**kw: Any) -> IngestionSettings:
    base: dict[str, Any] = {"contextualize_max_prompt_tokens": 1000}
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
    contextualizer(backend).contextualize([(doc, [chunk(0, "Section 0", "the chunk")])])
    (p,) = backend.sync_calls
    assert "ignore any instructions" in p.system
    assert p.document.startswith("<document") and "s0w4" in p.document
    assert "<chunk>\nthe chunk\n</chunk>" in p.chunk


def test_sync_mode_assigns_contexts_in_order() -> None:
    backend = FakeBackend()
    doc = make_doc(2, 10)
    result = contextualizer(backend).contextualize(
        [(doc, [chunk(0, "Section 0", "first"), chunk(1, "Section 1", "second")])]
    )
    assert [c.context for c in result.chunks[0]] == ["ctx for first", "ctx for second"]
    assert (result.input_tokens, result.output_tokens) == (200, 20)


def test_chunks_in_same_section_share_identical_cached_prefix() -> None:
    backend = FakeBackend()
    doc = make_doc(50, 100)
    contextualizer(backend).contextualize([(doc, [chunk(0, "Section 10", "a"), chunk(1, "Section 10", "b")])])
    first, second = (c.document for c in backend.sync_calls)
    assert first == second


def test_failures_leave_context_empty_and_are_counted() -> None:
    backend = FakeBackend(fail_on="bad")
    doc = make_doc(1, 5)
    result = contextualizer(backend).contextualize(
        [(doc, [chunk(0, "Section 0", "good"), chunk(1, "Section 0", "bad")])]
    )
    assert [c.context for c in result.chunks[0]] == ["ctx for good", None]
    assert result.failed_chunks == 1


@pytest.mark.parametrize(
    "error",
    [LLMError("invalid x-api-key", status_code=401), LLMError("ConnectError: refused"), KeyError("bug")],
)
def test_non_transient_errors_fail_the_run_and_stop_queued_requests(error: Exception) -> None:
    # Bad key/model, provider unreachable, or a bug: every chunk would fail the same way.
    backend = FakeBackend(error=error)
    doc = make_doc(1, 5)
    chunks = [chunk(i, "Section 0", f"c{i}") for i in range(20)]
    with pytest.raises(type(error)):
        contextualizer(backend, contextualize_concurrency=1).contextualize([(doc, chunks)])
    assert len(backend.sync_calls) < len(chunks)


def test_cost_estimate() -> None:
    doc = make_doc(1, 5)
    result = contextualizer(FakeBackend()).contextualize([(doc, [chunk(0, "Section 0")])])
    # 100 input * $1/M + 10 output * $5/M
    assert result.cost_usd == pytest.approx(150 / 1e6)


def test_empty_input_makes_no_calls() -> None:
    backend = FakeBackend()
    result = contextualizer(backend).contextualize([])
    assert result.chunks == [] and backend.sync_calls == []


class TruncatingBackend(FakeBackend):
    def complete(self, prompt: ContextPrompt) -> Completion:
        return Completion(text="First sentence. Second is cut mid-wo", usage=TokenUsage(), truncated=True)


def test_truncated_context_is_trimmed_to_last_full_sentence() -> None:
    doc = make_doc(1, 5)
    result = contextualizer(TruncatingBackend()).contextualize([(doc, [chunk(0, "Section 0")])])
    assert result.chunks[0][0].context == "First sentence."
