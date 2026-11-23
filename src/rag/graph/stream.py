"""Run one conversational turn in a worker thread with an end-to-end deadline and cancellation.

The graph is synchronous, and a single stalled provider stream once hung a turn for >15 minutes (SDK read
timeouts x retries). Consumers (CLI, SSE endpoint) iterate events from a queue with a deadline for the whole
turn, so they always get an answer or a timeout in bounded time. Python threads can't be killed: the worker is
stopped cooperatively, at its next event, which is frequent (every node and every token).
"""

from __future__ import annotations

import contextvars
import queue
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

Event = dict[str, Any]
EventCallback = Callable[[Event], None]
TurnFn = Callable[[EventCallback], dict[str, Any]]


class TurnTimeoutError(TimeoutError):
    """The turn did not finish within its deadline (it has been cancelled)."""


class TurnCancelledError(Exception):
    """Raised inside the worker to abort a cancelled turn at its next event."""


_DONE = object()


class TurnStream:
    def __init__(self, run: TurnFn, *, timeout_s: float, on_finish: Callable[[], None] | None = None) -> None:
        """`on_finish` runs in the worker when it really exits (after completion, error or cancellation)."""
        self._run = run
        self._timeout_s = timeout_s
        self._on_finish = on_finish
        self._queue: queue.Queue[object] = queue.Queue()
        self._cancelled = threading.Event()
        self._start_lock = threading.Lock()
        self._deadline: float | None = None
        self._error: BaseException | None = None
        self.result: dict[str, Any] | None = None

    def start(self) -> None:
        """Start the worker (idempotent); the deadline counts from here."""
        with self._start_lock:
            if self._deadline is not None:
                return
            self._deadline = time.monotonic() + self._timeout_s
            # Copy the caller's contextvars (log context, active trace span) into the worker.
            context = contextvars.copy_context()
            threading.Thread(target=context.run, args=(self._work,), name="rag-turn", daemon=True).start()

    def cancel(self) -> None:
        self._cancelled.set()
        self._queue.put(_DONE)  # wakes a consumer blocked on the queue

    def _on_event(self, event: Event) -> None:
        if self._cancelled.is_set():
            raise TurnCancelledError
        self._queue.put(event)

    def _work(self) -> None:
        try:
            result = self._run(self._on_event)
            if not self._cancelled.is_set():
                self.result = result
        except TurnCancelledError:
            pass
        except BaseException as exc:
            self._error = exc
        finally:
            self._queue.put(_DONE)
            if self._on_finish is not None:
                self._on_finish()

    def events(self) -> Iterator[Event]:
        """Yield the turn's events; raises the worker's exception, or TurnTimeoutError at the deadline."""
        self.start()
        assert self._deadline is not None
        deadline = self._deadline
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TurnTimeoutError(f"no answer within {self._timeout_s:g} s")
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    continue
                if item is _DONE:
                    break
                assert isinstance(item, dict)
                yield item
            if self._error is not None and not self._cancelled.is_set():
                raise self._error
        finally:
            # Timeout, consumer stopped early (client disconnected), or an error: stop the worker too.
            if self.result is None:
                self._cancelled.set()
