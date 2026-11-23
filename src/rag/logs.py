"""Logging setup: human-readable text (CLI default) or one JSON object per line (server / log shippers).

Request-scoped fields (request_id, thread_id, ...) are bound with `bind()` and stored in a contextvar, so they
follow the request into worker threads started with a copied context (see `rag.graph.stream`).
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import traceback
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Literal

LogFormat = Literal["text", "json"]

_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "rag_log_context", default=None
)

# Attributes every LogRecord has; anything else on a record came from `extra=` or the bound context.
# uvicorn adds an ANSI-coloured copy of its message as `color_message`.
_STANDARD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "message",
    "asctime",
    "color_message",
}

NOISY_LOGGERS = (
    "httpx",
    "httpx2",  # HTTP stack of some provider SDKs behind LangChain: one INFO line per LLM call
    "httpcore",
    "docling",
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "RapidOCR",
    "uvicorn.access",
)


@contextmanager
def bind(**fields: Any) -> Generator[None]:
    """Add fields to every log record emitted in this context (nested binds override, then restore)."""
    token = _CONTEXT.set({**(_CONTEXT.get() or {}), **fields})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


class ContextFilter(logging.Filter):
    """Copies the bound context onto each record (handler-level, so third-party loggers get it too)."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in (_CONTEXT.get() or {}).items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **_extras(record),
        }
        if record.exc_info:
            line["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(levelname)s %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        extras = _extras(record)
        if not extras:
            return text
        head, sep, tail = text.partition("\n")  # keep key=value on the message line, before any traceback
        fields = " ".join(f"{k}={v}" for k, v in extras.items())
        return f"{head} [{fields}]{sep}{tail}"


class _AbandonedNodeFilter(logging.Filter):
    """Drops the error a cancelled turn's still-running graph node logs when it finishes.

    A timed-out/cancelled turn closes LangGraph's stream, which shuts its executor down; a node that was
    mid-call (threads can't be killed) later fails to schedule its writes and concurrent.futures logs that.
    The turn has already been reported as timed out/cancelled, so this traceback is noise.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        # Identified by where it was raised, not by its message: a plain RuntimeError from
        # ThreadPoolExecutor.submit (BrokenThreadPool is a subclass, so `type(...) is` excludes it), with a
        # LangGraph frame on the stack (only the graph's abandoned callbacks, not other executors).
        if type(exc) is not RuntimeError or exc.__traceback__ is None:
            return True
        frames = [frame for frame, _ in traceback.walk_tb(exc.__traceback__)]
        raised_by_submit = frames[-1].f_code is ThreadPoolExecutor.submit.__code__
        from_langgraph = any(f.f_globals.get("__name__", "").startswith("langgraph.") for f in frames)
        return not (raised_by_submit and from_langgraph)


# Installed at import (the graph imports this module), so it applies to the CLI, the server and tests alike.
logging.getLogger("concurrent.futures").addFilter(_AbandonedNodeFilter())


class _StderrHandler(logging.StreamHandler[Any]):
    """Writes to the *current* sys.stderr (test/CLI runners swap it), like logging.lastResort."""

    @property
    def stream(self) -> Any:  # pyright: ignore[reportIncompatibleVariableOverride]
        return sys.stderr

    @stream.setter
    def stream(self, _: Any) -> None:
        pass


def configure(*, level: str = "INFO", fmt: LogFormat = "text", verbose: bool = False) -> None:
    """Replace root handlers with one stderr handler; quiet chatty libraries unless verbose."""
    handler = _StderrHandler()
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    handler.addFilter(ContextFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose else level.upper())
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG if verbose else logging.WARNING)
    # Uvicorn's loggers propagate to the root handler instead of their own formatters.
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True
