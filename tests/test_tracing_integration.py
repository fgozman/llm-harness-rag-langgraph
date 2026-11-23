"""Phase 4: real `rag ask` subprocesses export to the running Phoenix; spans are read back with its client."""

import os
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from rag.config import Settings
from rag.factory import build_index, build_ingestion_pipeline

REPO = Path(__file__).resolve().parents[1]
PROJECT = "rag-tests"
QUESTION = "What is the payload of the NRX-4471-B gripper?"
NODES = (
    "analyze_query",
    "rewrite_query",
    "retrieve",
    "rerank",
    "grade_documents",
    "generate",
    "check_answer",
    "respond",
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, str]]:
    mp = pytest.MonkeyPatch()
    mp.chdir(REPO)
    s = Settings()
    s.qdrant.collection = f"test_tracing_{uuid.uuid4().hex[:8]}"
    s.state_dir = tmp_path_factory.mktemp("state")
    s.ingestion.contextualize = False
    assert build_ingestion_pipeline(s).run(REPO / "tests" / "fixtures" / "docs").failed == 0
    child_env = {
        **os.environ,
        "QDRANT__COLLECTION": s.qdrant.collection,
        "STATE_DIR": str(s.state_dir),
        "PHOENIX__ENABLED": "true",  # conftest disables Phoenix for unit tests
        "PHOENIX__PROJECT_NAME": PROJECT,
        "COLUMNS": "200",
    }
    yield child_env
    build_index(s).client.delete_collection(s.qdrant.collection)
    mp.undo()


def rag_ask(env: dict[str, str], thread: str) -> subprocess.CompletedProcess[str]:
    cmd = ["uv", "run", "rag", "ask", QUESTION, "--thread-id", thread]
    return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=600, check=False)


Span = dict[str, Any]


def session_spans(settings: Settings, session: str, *, wait_s: float = 20.0) -> list[Span]:
    """Spans of one session as plain records (pandas is only the client's transport format)."""
    from phoenix.client import Client

    client = Client(base_url=settings.phoenix.collector_endpoint)
    deadline = time.monotonic() + wait_s
    while True:
        frame: Any = client.spans.get_spans_dataframe(project_name=PROJECT, limit=5000)
        records = cast(list[Span], frame.to_dict("records"))
        spans = [r for r in records if r.get("attributes.session.id") == session]
        if set(NODES) <= {r["name"] for r in spans} or time.monotonic() > deadline:
            return spans


def test_rag_ask_produces_a_complete_trace_in_phoenix(env: dict[str, str]) -> None:
    thread = f"tracing-{uuid.uuid4().hex[:8]}"
    result = rag_ask(env, thread)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "7.5" in result.stdout
    spans = session_spans(Settings(), thread)

    assert len({s["context.trace_id"] for s in spans}) == 1
    assert set(NODES) <= {s["name"] for s in spans}
    names = {s["context.span_id"]: s["name"] for s in spans}

    def of_kind(kind: str) -> list[Span]:
        return [s for s in spans if s["span_kind"] == kind]

    (retriever,) = of_kind("RETRIEVER")
    assert names[retriever["parent_id"]] == "retrieve"
    docs = retriever["attributes.retrieval.documents"]
    assert len(docs) > 0 and {"document.id", "document.score", "document.content"} <= set(docs[0])

    (reranker,) = of_kind("RERANKER")
    assert names[reranker["parent_id"]] == "rerank"
    assert len(reranker["attributes.reranker.output_documents"]) > 0

    llm = of_kind("LLM")
    assert len(llm) >= 5  # analyze, rewrite, grade, generate, check
    assert all(s["attributes.llm.token_count.prompt"] > 0 for s in llm)
    assert all(s["attributes.llm.token_count.completion"] > 0 for s in llm)
    assert {names[s["parent_id"]] for s in llm} >= {"analyze_query", "generate", "check_answer"}


def test_rag_ask_works_when_phoenix_is_down(env: dict[str, str]) -> None:
    down = {**env, "PHOENIX__COLLECTOR_ENDPOINT": "http://127.0.0.1:9"}
    started = time.monotonic()
    result = rag_ask(down, f"down-{uuid.uuid4().hex[:8]}")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "7.5" in result.stdout
    assert "tracing disabled" in result.stderr
    assert "Missing `tracer`" not in result.stdout + result.stderr
    assert time.monotonic() - started < 120
