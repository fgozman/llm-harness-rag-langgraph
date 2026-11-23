"""FastAPI app: POST /ask (SSE), POST /ingest + GET /ingest/{id} (background jobs), GET /health.

SSE stream of /ask, one JSON object per `data:` line:
  event: status        {"node", "text"}        progress of the graph (analyzing, searching, verifying)
  event: node          {"node"}                a graph node finished
  event: answer_start  {"attempt"}             a (re)generation starts; discard earlier tokens on attempt > 1
  event: token         {"text"}                answer tokens as the model streams them
  event: final         FinalEvent              answer text (citation markers validated), status, citations
  event: error         ErrorEvent              timeout / LLM provider failure / internal error
Exactly one `final` or `error` event ends every stream.
"""

# No `from __future__ import annotations`: FastAPI resolves endpoint annotations at runtime, including the
# closure-local dependency in `_add_ask_route`.
import logging
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Generator, MutableMapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import anyio
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from starlette.concurrency import iterate_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag import logs, telemetry
from rag.api.jobs import IngestJobs, JobConflictError, PipelineFactory
from rag.api.schemas import (
    AskRequest,
    ErrorEvent,
    FinalEvent,
    HealthResponse,
    IngestJob,
    IngestRequest,
    ServiceStatus,
)
from rag.config import Settings
from rag.graph.build import turn_stream
from rag.graph.state import FinalAnswer
from rag.graph.stream import TurnStream, TurnTimeoutError
from rag.llm.errors import PROVIDER_ERRORS
from rag.services import Probe, probe, service_checks

log = logging.getLogger("rag.api")

GraphFactory = Callable[[Settings], Any]

# Graph events forwarded to clients (the graph's internal "final" marker is replaced by the full payload).
_FORWARDED = frozenset({"status", "node", "answer_start", "token"})


class RequestContextMiddleware:
    """Request id (X-Request-ID in/out) bound to every log line of the request, plus one access log line.

    Pure ASGI rather than BaseHTTPMiddleware so the context and timing span the whole streamed body.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        request_id = headers.get(b"x-request-id", b"").decode("latin-1")[:64] or uuid.uuid4().hex
        status = 500
        started = time.monotonic()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                message["headers"] = [*message.get("headers", []), (b"x-request-id", request_id.encode())]
            await send(message)

        with logs.bind(request_id=request_id):
            try:
                await self.app(scope, receive, send_with_id)
            finally:
                log.info(
                    "request",
                    extra={
                        "method": scope["method"],
                        "path": scope["path"],
                        "status": status,
                        "duration_ms": round((time.monotonic() - started) * 1000, 1),
                    },
                )


@dataclass(slots=True)
class Turn:
    """One /ask turn, owned by the request-scoped dependency (see `_add_ask_route`)."""

    thread_id: str
    stream: TurnStream
    started: float = field(default_factory=time.monotonic)
    # Set by the event stream when it reaches its final/error event; anything else means the client left.
    outcome: str = "cancelled"


class TurnSlots:
    """Caps concurrent turns and rejects a second turn on a thread that is still answering."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._lock = threading.Lock()
        self._active: set[str] = set()

    @property
    def active(self) -> int:
        return len(self._active)

    def acquire(self, thread_id: str) -> None:
        with self._lock:
            if thread_id in self._active:
                raise HTTPException(409, f"thread {thread_id} is still answering a previous question")
            if len(self._active) >= self._limit:
                raise HTTPException(429, "too many concurrent questions", headers={"Retry-After": "5"})
            self._active.add(thread_id)

    def release(self, thread_id: str) -> None:
        with self._lock:
            self._active.discard(thread_id)


def _error(kind: str, message: str, thread_id: str) -> ServerSentEvent:
    return ServerSentEvent(
        event="error",
        data=ErrorEvent.model_validate({"type": kind, "message": message, "thread_id": thread_id}),
    )


def _resolve_ingest_root(settings: Settings, requested: str | None) -> Path:
    docs = settings.docs_dir.resolve()
    root = docs if requested is None else (docs / Path(requested).expanduser()).resolve()
    allowed = [docs, *(r.resolve() for r in settings.api.ingest_roots)]
    # An absolute path replaces docs_dir in the join above; either way it must stay under an allowed root.
    if not any(root == a or root.is_relative_to(a) for a in allowed):
        raise HTTPException(403, f"{root} is outside the allowed ingest roots (docs_dir, api.ingest_roots)")
    if not root.exists():
        raise HTTPException(404, f"not found: {root}")
    return root


def _default_graph(settings: Settings) -> Any:
    from rag.factory import build_qa_graph

    return build_qa_graph(settings)


def _default_pipeline(settings: Settings) -> Any:
    from rag.factory import build_ingestion_pipeline

    return build_ingestion_pipeline(settings)


@dataclass(slots=True)
class _ApiState:
    settings: Settings
    slots: TurnSlots
    jobs: IngestJobs
    graph: Any = None


async def _answer_events(turn: Turn, timeout_s: float) -> AsyncGenerator[ServerSentEvent]:
    # No logs.bind() here: contextvars set in an async generator leak into the consumer across yields, and an
    # abandoned generator is finalised in another task. Fields are passed explicitly instead.
    fields = {"thread_id": turn.thread_id}
    stream = turn.stream
    try:
        async for event in iterate_in_threadpool(stream.events()):
            kind = event.get("type")
            if kind in _FORWARDED:
                yield ServerSentEvent(event=kind, data={k: v for k, v in event.items() if k != "type"})
        final = (stream.result or {}).get("final")
        if not isinstance(final, FinalAnswer):
            raise RuntimeError("the graph finished without a final answer")
        turn.outcome = final.status
        yield ServerSentEvent(event="final", data=FinalEvent(thread_id=turn.thread_id, **final.model_dump()))
    except TurnTimeoutError as exc:
        turn.outcome = "timeout"
        log.warning("turn timed out", extra={**fields, "timeout_s": timeout_s})
        yield _error("timeout", str(exc), turn.thread_id)
    except PROVIDER_ERRORS as exc:
        turn.outcome = "llm_error"
        message = f"{type(exc).__name__}: {exc}"
        log.warning("LLM provider error", extra={**fields, "error": message})
        yield _error("llm_error", message, turn.thread_id)
    except Exception:
        turn.outcome = "internal_error"
        log.exception("turn failed", extra=fields)
        yield _error("internal_error", "internal error; see the server log", turn.thread_id)


def _add_ask_route(app: FastAPI, api: _ApiState) -> None:
    settings = api.settings

    def start_turn(body: AskRequest, response: Response) -> Generator[Turn]:
        limit = settings.api.max_question_chars
        if len(body.question) > limit:
            raise HTTPException(422, f"question longer than {limit} characters")
        thread_id = body.thread_id or uuid.uuid4().hex
        api.slots.acquire(thread_id)
        stream = turn_stream(
            api.graph,
            body.question,
            thread_id=thread_id,
            timeout_s=settings.graph.turn_timeout_s,
            user_id=settings.phoenix.user_id or "api",
            metadata={
                "interface": "api",
                "provider": settings.llm.provider,
                "generation_model": settings.llm.generation_model,
            },
            # The conversation stays busy until the worker has really stopped, not just until the response
            # ends: a cancelled worker finishes its in-flight LLM call first, and two workers on one thread
            # would interleave checkpoint writes.
            on_finish=lambda: api.slots.release(thread_id),
        )
        stream.start()
        turn = Turn(thread_id=thread_id, stream=stream)
        response.headers["X-Thread-ID"] = thread_id
        log.info("turn started", extra={"thread_id": thread_id, "question_chars": len(body.question)})
        try:
            yield turn
        finally:
            # Runs in the request's own context once the response is done, including client disconnects, so
            # cancellation doesn't depend on when the abandoned generator gets garbage-collected.
            stream.cancel()
            latency_ms = round((time.monotonic() - turn.started) * 1000, 1)
            log.info(
                "turn finished",
                extra={"thread_id": thread_id, "outcome": turn.outcome, "latency_ms": latency_ms},
            )

    # scope="request": teardown runs after the streamed response ends, not when the handler returns.
    @app.post("/ask", response_class=EventSourceResponse)
    async def ask(
        turn: Annotated[Turn, Depends(start_turn, scope="request")],
    ) -> AsyncGenerator[ServerSentEvent]:
        """Answer a question; streams progress, tokens, then the final answer with citations (SSE)."""
        async for event in _answer_events(turn, settings.graph.turn_timeout_s):
            yield event


def _add_ingest_routes(app: FastAPI, api: _ApiState) -> None:
    @app.post("/ingest", status_code=202)
    def ingest(body: IngestRequest) -> IngestJob:
        """Start indexing a folder (or file) under docs_dir in the background; poll GET /ingest/{id}."""
        root = _resolve_ingest_root(api.settings, body.path)
        try:
            return api.jobs.submit(
                root, force=body.force, retry_failed=body.retry_failed, contextualize=body.contextualize
            )
        except JobConflictError as exc:
            raise HTTPException(409, f"ingest job {exc} is still running") from exc

    @app.get("/ingest/{job_id}")
    def ingest_status(job_id: str) -> IngestJob:
        job = api.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown ingest job {job_id}")
        return job


def _add_health_route(app: FastAPI, api: _ApiState, check: Probe) -> None:
    @app.get("/health", responses={503: {"model": HealthResponse}})
    def health(response: Response) -> HealthResponse:
        """Qdrant (required) and Phoenix (optional) reachability; 503 when questions can't be answered."""
        checks = service_checks(api.settings, check)
        degraded = any(c.failed for c in checks)
        if degraded:
            response.status_code = 503
        return HealthResponse(
            status="degraded" if degraded else "ok",
            services={c.name: ServiceStatus(url=c.url, status=c.status, required=c.required) for c in checks},
            active_turns=api.slots.active,
            ingest_running=api.jobs.running,
        )


def _is_health_probe(scope: MutableMapping[str, Any]) -> bool:
    return scope.get("path") == "/health"


def create_app(
    settings: Settings,
    *,
    graph_factory: GraphFactory = _default_graph,
    pipeline_factory: PipelineFactory = _default_pipeline,
    check: Probe = probe,
) -> FastAPI:
    api = _ApiState(
        settings=settings,
        slots=TurnSlots(settings.api.max_concurrent_turns),
        jobs=IngestJobs(settings, pipeline_factory),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        telemetry.init_tracing(settings.phoenix)
        started = time.monotonic()
        # Models load once, before the server accepts requests, so the first question isn't slow.
        api.graph = await anyio.to_thread.run_sync(graph_factory, settings)
        log.info("graph ready", extra={"load_s": round(time.monotonic() - started, 1)})
        yield

    app = FastAPI(
        title="rag",
        version="0.1.0",
        summary="Grounded, cited Q&A over a local document folder.",
        lifespan=lifespan,
        # FastAPI's native OpenTelemetry spans (global provider = Phoenix when tracing is on) give each turn
        # an HTTP root span; health probes would only flood the trace list.
        telemetry={"exclude": _is_health_probe},
    )
    app.add_middleware(RequestContextMiddleware)
    _add_ask_route(app, api)
    _add_ingest_routes(app, api)
    _add_health_route(app, api, check)
    return app
