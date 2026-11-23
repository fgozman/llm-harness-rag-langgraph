"""Phoenix / OpenTelemetry tracing.

`init_tracing` registers Phoenix once per process with OpenInference auto-instrumentation (LangChain /
LangGraph, Anthropic, OpenAI). Manual spans use the global OTel API, which is a no-op until a provider is
registered, so tracing can be disabled (config) or unavailable (Phoenix down) without affecting the app.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from rag.config import PhoenixSettings

log = logging.getLogger(__name__)

_TRACER_NAME = "rag"
_PROBE_TIMEOUT_S = 0.5
# Bound what a stalled/down collector can cost: OTLP exporter retries otherwise block flushes for seconds.
_EXPORT_TIMEOUT_S = "3"
_MAX_CONTENT_CHARS = 2000


class _State:
    enabled: bool | None = None  # None = not initialised yet in this process


_STATE = _State()

KIND = "openinference.span.kind"


def _probe(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/healthz", timeout=_PROBE_TIMEOUT_S).status_code == 200
    except httpx.HTTPError:
        return False


def _register(**kwargs: Any) -> None:
    from phoenix.otel import register

    register(**kwargs)


def init_tracing(settings: PhoenixSettings, *, probe: Callable[[str], bool] = _probe) -> bool:
    """Enable tracing if configured and Phoenix is reachable. Idempotent; returns whether tracing is on."""
    if _STATE.enabled is not None:
        return _STATE.enabled
    base = settings.collector_endpoint.rstrip("/")
    if not settings.enabled:
        _STATE.enabled = False
        return False
    if not probe(base):
        log.warning("Phoenix is not reachable at %s; tracing disabled for this run", base)
        _STATE.enabled = False
        return False
    os.environ.setdefault("OTEL_EXPORTER_OTLP_TIMEOUT", _EXPORT_TIMEOUT_S)
    # If Phoenix goes away mid-run, one error per failed batch is enough; retry warnings are noise.
    logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter").setLevel(logging.ERROR)
    kwargs: dict[str, Any] = {
        "endpoint": f"{base}/v1/traces",
        "project_name": settings.project_name,
        "batch": True,
        "auto_instrument": True,
        "verbose": False,
    }
    if settings.api_key is not None:  # only needed when Phoenix authentication is enabled
        kwargs["api_key"] = settings.api_key.get_secret_value()
    _register(**kwargs)
    _STATE.enabled = True
    return True


def reset_for_tests() -> None:
    _STATE.enabled = None


def tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)


@contextmanager
def span(name: str, kind: str, attributes: Mapping[str, Any] | None = None) -> Generator[Span]:
    """A span with an OpenInference kind; exceptions are recorded and re-raised."""
    with tracer().start_as_current_span(name, record_exception=True, set_status_on_exception=True) as current:
        current.set_attribute(KIND, kind)
        for key, value in (attributes or {}).items():
            set_attr(current, key, value)
        yield current


def set_attr(target: Span, key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool | int | float | str):
        target.set_attribute(key, value)
    else:
        target.set_attribute(key, json.dumps(value, default=str))


@dataclass(frozen=True, slots=True)
class TracedDocument:
    id: str
    score: float | None
    content: str
    metadata: Mapping[str, Any]


def set_documents(target: Span, prefix: str, documents: Sequence[TracedDocument]) -> None:
    """OpenInference document list, e.g. retrieval.documents.{i}.document.{id,score,content,metadata}."""
    for i, doc in enumerate(documents):
        base = f"{prefix}.{i}.document"
        target.set_attribute(f"{base}.id", doc.id)
        if doc.score is not None:
            target.set_attribute(f"{base}.score", float(doc.score))
        target.set_attribute(f"{base}.content", doc.content[:_MAX_CONTENT_CHARS])
        target.set_attribute(f"{base}.metadata", json.dumps(dict(doc.metadata), default=str))


def mark_ok(target: Span) -> None:
    target.set_status(Status(StatusCode.OK))
