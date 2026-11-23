import json
import logging
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from openinference.instrumentation import TracerProvider
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag import telemetry
from rag.config import (
    ChunkingSettings,
    GraphSettings,
    IngestionSettings,
    PhoenixSettings,
    RerankerSettings,
    RetrievalSettings,
)
from rag.graph.build import GraphDeps, build_graph, run_turn, state_serializer
from rag.ingestion.manifest import Manifest
from rag.ingestion.models import ParentSection
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.retriever import Retriever, SearchMode
from test_graph import THREE, FakeLLM, FakeRetriever, analysis, grades, ok, plan  # pyright: ignore[reportImplicitRelativeImport]
from test_pipeline import FakeDense, FakeIndex, FakeParser, FakeSparse  # pyright: ignore[reportImplicitRelativeImport]
from test_retriever import RANKINGS, FakeReranker, FakeSearcher  # pyright: ignore[reportImplicitRelativeImport]


def attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def by_name(spans: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    return {s.name: s for s in spans.get_finished_spans()}


# --- init ------------------------------------------------------------------------------------------


class RegisterSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


@pytest.fixture(autouse=True)
def _reset_tracing() -> Iterator[None]:
    telemetry.reset_for_tests()
    yield
    telemetry.reset_for_tests()


def test_disabled_tracing_never_registers(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = RegisterSpy()
    monkeypatch.setattr(telemetry, "_register", spy)
    calls = spy.calls
    assert telemetry.init_tracing(PhoenixSettings(enabled=False), probe=lambda url: True) is False
    assert calls == []


def test_unreachable_phoenix_disables_tracing_quickly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    spy = RegisterSpy()
    monkeypatch.setattr(telemetry, "_register", spy)
    calls = spy.calls
    start = time.monotonic()
    with caplog.at_level(logging.WARNING):
        enabled = telemetry.init_tracing(PhoenixSettings(collector_endpoint="http://127.0.0.1:9"))
    assert enabled is False and calls == []
    assert time.monotonic() - start < 2.0
    assert "tracing disabled" in caplog.text


def test_reachable_phoenix_registers_once_with_auto_instrumentation(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = RegisterSpy()
    monkeypatch.setattr(telemetry, "_register", spy)
    calls = spy.calls
    cfg = PhoenixSettings(collector_endpoint="http://localhost:6007/", project_name="rag-test")
    assert telemetry.init_tracing(cfg, probe=lambda url: True) is True
    assert telemetry.init_tracing(cfg, probe=lambda url: True) is True
    assert calls == [
        {
            "endpoint": "http://localhost:6007/v1/traces",
            "project_name": "rag-test",
            "batch": True,
            "auto_instrument": True,
            "verbose": False,
        }
    ]


# --- retrieval spans ---------------------------------------------------------------------------------


def retriever() -> Retriever:
    class Parents:
        def get_parents(self, parent_ids: Sequence[str]) -> dict[str, ParentSection]:
            return {p: ParentSection(p, f"parent {p}", ("S",), None, None) for p in parent_ids}

    return Retriever(
        searcher=FakeSearcher(RANKINGS),
        reranker=FakeReranker(favour="text of e"),
        parents=Parents(),
        count_tokens=lambda s: len(s.split()),
        retrieval=RetrievalSettings(candidates_per_ranking=20),
        reranker_settings=RerankerSettings(candidates=10, top_k=2),
    )


def test_retrieval_spans_follow_openinference_conventions(spans: InMemorySpanExporter) -> None:
    retriever().retrieve(["q"], mode="hybrid")
    named = by_name(spans)
    retrieve = attrs(named["hybrid_search"])
    assert retrieve["openinference.span.kind"] == "RETRIEVER"
    assert json.loads(retrieve["input.value"]) == {"queries": ["q"], "mode": "hybrid", "filters": None}
    assert retrieve["retrieval.documents.0.document.id"] == "a"
    assert retrieve["retrieval.documents.0.document.content"] == "text of a"
    assert isinstance(retrieve["retrieval.documents.0.document.score"], float)
    assert json.loads(retrieve["retrieval.documents.0.document.metadata"])["file_name"] == "a.md"

    fusion = attrs(named["rrf_fusion"])
    assert fusion["openinference.span.kind"] == "CHAIN" and fusion["rag.rrf.k"] == 60
    assert fusion["rag.rrf.rankings"] == 2 and fusion["rag.rrf.fused"] == 5

    rerank = attrs(named["rerank_passages"])
    assert rerank["openinference.span.kind"] == "RERANKER"
    assert rerank["reranker.query"] == "q" and rerank["reranker.top_k"] == 2
    assert rerank["reranker.model_name"] == "fake-reranker"
    assert rerank["reranker.input_documents.4.document.id"] == "d"
    assert rerank["reranker.output_documents.0.document.id"] == "e"
    assert rerank["reranker.output_documents.0.document.score"] == 1.0

    context = attrs(named["build_context"])
    assert context["openinference.span.kind"] == "CHAIN" and context["rag.context.passages"] == 2


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_retrieve_span_records_mode(spans: InMemorySpanExporter, mode: SearchMode) -> None:
    retriever().retrieve(["q"], mode=mode)
    assert json.loads(attrs(by_name(spans)["hybrid_search"])["input.value"])["mode"] == mode


# --- graph spans -------------------------------------------------------------------------------------


@pytest.fixture
def langchain_instrumented(tracer_provider: TracerProvider) -> Iterator[None]:
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    yield
    instrumentor.uninstrument()


def test_graph_nodes_traced_with_session_and_nested_retrieval(
    spans: InMemorySpanExporter, langchain_instrumented: None
) -> None:
    class TracedRetriever(FakeRetriever):
        def search(self, queries: Sequence[str], *, filters: Any = None) -> list[Any]:
            with telemetry.span("hybrid_search", "RETRIEVER"):
                return super().search(queries, filters=filters)

    llm = FakeLLM([analysis(), plan(), grades(True, False, True), ok()], ["A [1]."])
    graph = build_graph(
        GraphDeps(llm=llm, retriever=TracedRetriever([THREE]), settings=GraphSettings()),
        checkpointer=InMemorySaver(serde=state_serializer()),
    )
    run_turn(graph, "q", thread_id="thread-7", user_id="tester", metadata={"interface": "test"})
    finished = spans.get_finished_spans()
    names = {s.name for s in finished}
    for node in (
        "analyze_query",
        "rewrite_query",
        "retrieve",
        "rerank",
        "grade_documents",
        "generate",
        "check_answer",
        "respond",
    ):
        assert node in names, node
    assert {attrs(s).get("session.id") for s in finished} == {"thread-7"}
    assert {attrs(s).get("user.id") for s in finished} == {"tester"}
    contexts = [s.context for s in finished if s.context is not None]
    assert len({c.trace_id for c in contexts}) == 1, "one trace per turn"
    by_id = {s.context.span_id: s for s in finished if s.context is not None}
    search = next(s for s in finished if s.name == "hybrid_search")
    assert search.parent is not None and by_id[search.parent.span_id].name == "retrieve", (
        "nested under the node"
    )
    root = next(s for s in finished if s.parent is None)
    meta = json.loads(attrs(root)["metadata"])
    assert meta["interface"] == "test" and meta["prompt_versions"]["generate"].startswith("generate-v")


# --- ingestion spans ---------------------------------------------------------------------------------


def test_ingestion_stages_are_traced(spans: InMemorySpanExporter, tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("alpha content")
    pipeline = IngestionPipeline(
        manifest=Manifest(tmp_path / "rag.db"),
        parser=FakeParser(),
        dense=FakeDense(),
        sparse=FakeSparse(),
        index=FakeIndex(),
        count_tokens=lambda s: len(s.split()),
        chunking=ChunkingSettings(),
        ingestion=IngestionSettings(contextualize=False),
    )
    pipeline.run(root)
    named = by_name(spans)
    ingest = attrs(named["ingest"])
    assert ingest["openinference.span.kind"] == "CHAIN"
    assert ingest["rag.ingest.indexed"] == 1 and ingest["rag.ingest.chunks_written"] == 1
    for stage in ("parse", "chunk", "embed", "index"):
        assert stage in named, stage
    assert attrs(named["parse"])["rag.file.path"].endswith("a.md")


def test_api_key_is_passed_when_phoenix_auth_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import SecretStr

    spy = RegisterSpy()
    monkeypatch.setattr(telemetry, "_register", spy)
    telemetry.init_tracing(PhoenixSettings(api_key=SecretStr("k-123")), probe=lambda url: True)
    assert spy.calls[0]["api_key"] == "k-123"


def test_graph_without_instrumentation_prints_no_tracer_warnings(capsys: pytest.CaptureFixture[str]) -> None:
    llm = FakeLLM([analysis(), plan(), grades(True, False, True), ok()], ["A [1]."])
    graph = build_graph(
        GraphDeps(llm=llm, retriever=FakeRetriever([THREE]), settings=GraphSettings()),
        checkpointer=InMemorySaver(serde=state_serializer()),
    )
    run_turn(graph, "q", thread_id="t")
    # openinference prints (not logs) this when asked for a span without instrumentation.
    assert "Missing `tracer`" not in capsys.readouterr().out


def test_contextualization_worker_threads_keep_the_trace(spans: InMemorySpanExporter) -> None:
    # Sync contextualization calls the LLM from a thread pool; those LLM spans must stay in the ingest trace.
    from rag.ingestion.contextualize import Completion, ContextPrompt
    from test_contextualize import FakeBackend, chunk, contextualizer, make_doc  # pyright: ignore[reportImplicitRelativeImport]

    class TracingBackend(FakeBackend):
        def complete(self, prompt: ContextPrompt) -> Completion:
            with telemetry.span("llm_call", "LLM"):
                return super().complete(prompt)

    doc = make_doc(2, 5)
    with telemetry.span("contextualize", "CHAIN"):
        contextualizer(TracingBackend(), contextualize_mode="sync").contextualize(
            [(doc, [chunk(0, "Section 0", "a"), chunk(1, "Section 1", "b"), chunk(2, "Section 0", "c")])]
        )
    finished = spans.get_finished_spans()
    parent = next(s for s in finished if s.name == "contextualize")
    calls = [s for s in finished if s.name == "llm_call"]
    assert len(calls) == 3
    assert all(
        s.parent is not None and parent.context is not None and s.parent.span_id == parent.context.span_id
        for s in calls
    )
