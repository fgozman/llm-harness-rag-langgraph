import json
import logging
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from graph_fakes import THREE, FakeLLM, FakeRetriever, analysis, grades, make, ok, plan
from rag import logs
from rag.api.app import TurnSlots, create_app
from rag.config import Settings
from rag.ingestion.pipeline import IngestStats, ProgressCallback
from rag.llm.base import LLMError, Message, TokenCallback


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Settings must not pick up the repo's .env / config.yaml.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_CONFIG_FILE", raising=False)


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "a.md").write_text("# A\n")
    return folder


def settings_for(docs: Path, **graph: Any) -> Settings:
    return Settings.model_validate({"docs_dir": docs, "state_dir": docs.parent / "state", "graph": graph})


def answering_llm(*answers: str) -> FakeLLM:
    structured: list[BaseModel] = []
    for _ in answers:
        structured += [analysis(), plan(), grades(True, False, True), ok()]
    return FakeLLM(structured, list(answers))


def client_for(
    settings: Settings,
    llm: Any,
    retriever: FakeRetriever | None = None,
    *,
    pipeline: Any = None,
    probe_status: dict[str, str] | None = None,
) -> TestClient:
    graph = make(llm, retriever or FakeRetriever([THREE]))
    statuses = probe_status or {}

    def check(url: str) -> str:
        return next((s for key, s in statuses.items() if key in url), "ok")

    app = create_app(
        settings,
        graph_factory=lambda _: graph,
        pipeline_factory=lambda _: pipeline,
        check=check,
    )
    return TestClient(app)


def sse(body: str) -> list[tuple[str, Any]]:
    """Parse an SSE body into (event, data) pairs, skipping comments (keep-alive pings)."""
    events: list[tuple[str, Any]] = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if fields:
            events.append((fields.get("event", "message"), json.loads(fields["data"])))
    return events


# --- /ask -------------------------------------------------------------------------------------------


def test_ask_streams_tokens_and_ends_with_a_citations_payload(docs: Path) -> None:
    llm = answering_llm("Revenue 10 [1] and costs 4 [2].")
    with client_for(settings_for(docs), llm) as client:
        response = client.post("/ask", json={"question": "What were revenue and costs?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse(response.text)
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "status"
    assert "answer_start" in kinds and "token" in kinds
    assert kinds[-1] == "final" and kinds.count("final") == 1
    tokens = "".join(data["text"] for kind, data in events if kind == "token")
    final = events[-1][1]
    assert tokens.strip() == final["text"] == "Revenue 10 [1] and costs 4 [2]."
    assert final["status"] == "answered"
    assert [(c["index"], c["file_name"], c["page_start"]) for c in final["citations"]] == [
        (1, "a.pdf", 3),
        (2, "c.pdf", 3),
    ]
    assert final["citations"][0]["snippet"] == "Revenue was 10."
    assert final["thread_id"] == response.headers["x-thread-id"]
    assert response.headers["x-request-id"]


def test_follow_up_on_the_same_thread_sees_the_history(docs: Path) -> None:
    llm = answering_llm("First [1].", "Second [1].")
    with client_for(settings_for(docs), llm) as client:
        first = client.post("/ask", json={"question": "Q1?", "thread_id": "conv-1"})
        second = client.post("/ask", json={"question": "and Q2?", "thread_id": "conv-1"})
    assert sse(first.text)[-1][1]["thread_id"] == "conv-1"
    assert sse(second.text)[-1][1]["text"] == "Second [1]."
    # The second turn's rewrite prompt contains the first exchange.
    assert "First [1]." in llm.prompts("QueryPlan")[1]


class SlowLLM(FakeLLM):
    def generate(
        self, task: Any, *, system: str, messages: Sequence[Message], on_token: TokenCallback | None = None
    ) -> str:
        time.sleep(1.0)
        return super().generate(task, system=system, messages=messages, on_token=on_token)


def test_turn_deadline_ends_the_stream_with_a_timeout_error(docs: Path) -> None:
    llm = SlowLLM([analysis(), plan(), grades(True, False, True), ok()], ["late"])
    started = time.monotonic()
    thread = "slow-thread"
    with client_for(settings_for(docs, turn_timeout_s=0.3), llm) as client:
        events = sse(client.post("/ask", json={"question": "q", "thread_id": thread}).text)
        assert time.monotonic() - started < 1.0
        kind, data = events[-1]
        assert kind == "error" and data["type"] == "timeout"
        assert "final" not in [k for k, _ in events]
        # The worker is still inside its (slow) LLM call: the conversation stays busy until it really stops.
        assert client.post("/ask", json={"question": "again", "thread_id": thread}).status_code == 409
        for _ in range(300):
            if client.get("/health").json()["active_turns"] == 0:
                break
            time.sleep(0.01)
        assert client.get("/health").json()["active_turns"] == 0


class FailingLLM(FakeLLM):
    def generate(
        self, task: Any, *, system: str, messages: Sequence[Message], on_token: TokenCallback | None = None
    ) -> str:
        raise LLMError("refused")


def test_provider_errors_end_the_stream_with_an_llm_error(docs: Path) -> None:
    llm = FailingLLM([analysis(), plan(), grades(True, False, True)])
    with client_for(settings_for(docs), llm) as client:
        events = sse(client.post("/ask", json={"question": "q"}).text)
    assert events[-1] == (
        "error",
        {"type": "llm_error", "message": "LLMError: refused", "thread_id": events[-1][1]["thread_id"]},
    )


class BrokenRetriever(FakeRetriever):
    def search(self, queries: Sequence[str], *, filters: Any = None) -> list[Any]:
        raise RuntimeError("secret internal detail")


def test_unexpected_errors_are_reported_without_internals(
    docs: Path, caplog: pytest.LogCaptureFixture
) -> None:
    llm = FakeLLM([analysis(), plan()])
    with client_for(settings_for(docs), llm, BrokenRetriever([])) as client:
        events = sse(client.post("/ask", json={"question": "q"}).text)
    kind, data = events[-1]
    assert kind == "error" and data["type"] == "internal_error"
    assert "secret" not in data["message"]
    assert any("secret internal detail" in (r.exc_text or "") for r in caplog.records if r.exc_info)


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "q", "thread_id": "../../etc/passwd"},
        {"question": "x" * 5000},
        {},
    ],
)
def test_invalid_questions_are_rejected(docs: Path, body: dict[str, Any]) -> None:
    with client_for(settings_for(docs), FakeLLM()) as client:
        assert client.post("/ask", json=body).status_code == 422


def test_turn_slots_reject_a_busy_thread_and_cap_concurrency() -> None:
    slots = TurnSlots(limit=2)
    slots.acquire("a")
    with pytest.raises(HTTPException) as busy:
        slots.acquire("a")
    assert busy.value.status_code == 409
    slots.acquire("b")
    with pytest.raises(HTTPException) as full:
        slots.acquire("c")
    assert full.value.status_code == 429
    slots.release("a")
    slots.acquire("c")
    assert slots.active == 2


def test_slot_is_released_after_the_stream(docs: Path) -> None:
    llm = answering_llm("One [1].", "Two [1].")
    settings = settings_for(docs)
    settings.api.max_concurrent_turns = 1
    with client_for(settings, llm) as client:
        for _ in range(2):  # would be 429/409 if the first turn kept its slot
            response = client.post("/ask", json={"question": "q", "thread_id": "same"})
            assert response.status_code == 200
        assert client.get("/health").json()["active_turns"] == 0


def test_request_logs_carry_the_request_id(docs: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.handler.addFilter(logs.ContextFilter())
    caplog.set_level(logging.INFO, logger="rag.api")
    with client_for(settings_for(docs), answering_llm("A [1].")) as client:
        client.post("/ask", json={"question": "q"}, headers={"X-Request-ID": "req-7"})
    mine = [r for r in caplog.records if getattr(r, "request_id", None) == "req-7"]
    messages = {r.getMessage() for r in mine}
    assert {"turn started", "turn finished", "request"} <= messages
    finished = next(r for r in mine if r.getMessage() == "turn finished")
    assert vars(finished)["outcome"] == "answered"
    assert vars(finished)["thread_id"]
    access = next(r for r in mine if r.getMessage() == "request")
    assert (vars(access)["path"], vars(access)["status"]) == ("/ask", 200)


# --- /health ----------------------------------------------------------------------------------------


def test_health_ok_when_qdrant_is_up_even_if_phoenix_is_down(docs: Path) -> None:
    settings = settings_for(docs)
    settings.phoenix.enabled = True
    with client_for(settings, FakeLLM(), probe_status={"healthz": "down (ConnectError)"}) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["services"]["phoenix"]["status"] == "down (ConnectError)"


def test_health_is_503_when_qdrant_is_down(docs: Path) -> None:
    with client_for(settings_for(docs), FakeLLM(), probe_status={"readyz": "down (ConnectError)"}) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


# --- /ingest ----------------------------------------------------------------------------------------


class FakePipeline:
    def __init__(self, gate: threading.Event | None = None, error: Exception | None = None) -> None:
        self.gate = gate
        self.error = error
        self.calls: list[tuple[Path, dict[str, object]]] = []

    def run(
        self,
        root: Path,
        *,
        retry_failed: bool = False,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IngestStats:
        self.calls.append((root, {"force": force, "retry_failed": retry_failed}))
        if self.gate is not None:
            self.gate.wait(5)
        if self.error is not None:
            raise self.error
        if progress:
            progress("parsed", 1)
            progress("indexed", 1)
        return IngestStats(discovered=1, indexed=1, chunks_written=3)


def wait_for_job(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(200):
        job = client.get(f"/ingest/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.01)
    raise AssertionError("ingest job did not finish")


def test_ingest_runs_in_the_background_and_reports_stats(docs: Path) -> None:
    pipeline = FakePipeline()
    with client_for(settings_for(docs), FakeLLM(), pipeline=pipeline) as client:
        response = client.post("/ingest", json={"path": "sub", "force": True, "contextualize": False})
        assert response.status_code == 202
        job = wait_for_job(client, response.json()["id"])
    assert job["status"] == "succeeded"
    assert job["stats"]["chunks_written"] == 3
    assert job["progress"] == {"parsed": 1, "indexed": 1}
    assert job["contextualize"] is False
    assert pipeline.calls == [((docs / "sub").resolve(), {"force": True, "retry_failed": False})]


def test_ingest_contextualize_override_does_not_leak_into_server_settings(docs: Path) -> None:
    seen: list[bool] = []
    settings = settings_for(docs)
    settings.ingestion.contextualize = True

    def factory(s: Settings) -> FakePipeline:
        seen.append(s.ingestion.contextualize)
        return FakePipeline()

    app = create_app(settings, graph_factory=lambda _: None, pipeline_factory=factory, check=lambda _: "ok")
    with TestClient(app) as client:
        wait_for_job(client, client.post("/ingest", json={"contextualize": False}).json()["id"])
    assert seen == [False]
    assert settings.ingestion.contextualize is True


def test_ingest_defaults_to_docs_dir(docs: Path) -> None:
    pipeline = FakePipeline()
    with client_for(settings_for(docs), FakeLLM(), pipeline=pipeline) as client:
        wait_for_job(client, client.post("/ingest", json={}).json()["id"])
    assert pipeline.calls[0][0] == docs.resolve()


@pytest.mark.parametrize(("path", "status"), [("../", 403), ("/etc", 403), ("missing", 404)])
def test_ingest_rejects_paths_outside_docs_dir_or_missing(docs: Path, path: str, status: int) -> None:
    with client_for(settings_for(docs), FakeLLM(), pipeline=FakePipeline()) as client:
        assert client.post("/ingest", json={"path": path}).status_code == status


def test_ingest_allows_configured_extra_roots(docs: Path, tmp_path: Path) -> None:
    extra = tmp_path / "corpus"
    extra.mkdir()
    settings = settings_for(docs)
    settings.api.ingest_roots = [extra]
    pipeline = FakePipeline()
    with client_for(settings, FakeLLM(), pipeline=pipeline) as client:
        wait_for_job(client, client.post("/ingest", json={"path": str(extra)}).json()["id"])
    assert pipeline.calls[0][0] == extra.resolve()


def test_only_one_ingest_job_runs_at_a_time(docs: Path) -> None:
    gate = threading.Event()
    with client_for(settings_for(docs), FakeLLM(), pipeline=FakePipeline(gate)) as client:
        first = client.post("/ingest", json={})
        assert client.post("/ingest", json={}).status_code == 409
        assert client.get("/health").json()["ingest_running"] is True
        gate.set()
        assert wait_for_job(client, first.json()["id"])["status"] == "succeeded"
        assert client.post("/ingest", json={}).status_code == 202


def test_failed_ingest_job_reports_the_error(docs: Path) -> None:
    pipeline = FakePipeline(error=ConnectionError("qdrant unreachable"))
    with client_for(settings_for(docs), FakeLLM(), pipeline=pipeline) as client:
        job = wait_for_job(client, client.post("/ingest", json={}).json()["id"])
    assert job["status"] == "failed"
    assert job["error"] == "ConnectionError: qdrant unreachable"


def test_unknown_job_is_404(docs: Path) -> None:
    with client_for(settings_for(docs), FakeLLM()) as client:
        assert client.get("/ingest/nope").status_code == 404


def test_http_spans_wrap_turns_but_health_probes_are_not_traced(docs: Path, spans: Any) -> None:
    with client_for(settings_for(docs), answering_llm("A [1].")) as client:
        client.get("/health")
        client.post("/ask", json={"question": "q"})
    names = [s.name for s in spans.get_finished_spans()]
    assert "POST /ask" in names
    assert not any("/health" in n for n in names)
