import contextvars
import threading
import time
from typing import Any

import pytest

from rag.graph.stream import TurnStream, TurnTimeoutError

EventFn = Any


def test_events_are_yielded_in_order_and_result_is_kept() -> None:
    def run(on_event: EventFn) -> dict[str, Any]:
        for i in range(3):
            on_event({"type": "token", "text": str(i)})
        return {"final": "done"}

    stream = TurnStream(run, timeout_s=5)
    assert [e["text"] for e in stream.events()] == ["0", "1", "2"]
    assert stream.result == {"final": "done"}


def test_worker_exception_is_reraised_in_the_consumer() -> None:
    def run(on_event: EventFn) -> dict[str, Any]:
        on_event({"type": "status"})
        raise ValueError("provider down")

    stream = TurnStream(run, timeout_s=5)
    events: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="provider down"):
        events.extend(stream.events())
    assert events == [{"type": "status"}]


def test_deadline_covers_the_whole_turn_not_each_event() -> None:
    # Events keep arriving, but the turn as a whole must still end at the deadline.
    aborted = threading.Event()

    def run(on_event: EventFn) -> dict[str, Any]:
        try:
            while True:
                on_event({"type": "token", "text": "x"})
                time.sleep(0.02)
        except BaseException:
            aborted.set()
            raise

    stream = TurnStream(run, timeout_s=0.3)
    started = time.monotonic()
    with pytest.raises(TurnTimeoutError):
        for _ in stream.events():
            pass
    assert time.monotonic() - started < 1.0
    # The worker is stopped at its next event instead of running on in the background.
    assert aborted.wait(1.0)


def test_deadline_fires_while_the_worker_is_blocked() -> None:
    release = threading.Event()

    def run(on_event: EventFn) -> dict[str, Any]:
        release.wait(5)  # a stalled provider stream: no events at all
        on_event({"type": "token"})
        return {}

    stream = TurnStream(run, timeout_s=0.2)
    started = time.monotonic()
    with pytest.raises(TurnTimeoutError):
        list(stream.events())
    assert time.monotonic() - started < 1.0
    release.set()


def test_cancel_stops_iteration_and_aborts_the_worker() -> None:
    aborted = threading.Event()

    def run(on_event: EventFn) -> dict[str, Any]:
        try:
            while True:
                on_event({"type": "token"})
                time.sleep(0.01)
        except BaseException:
            aborted.set()
            raise

    stream = TurnStream(run, timeout_s=5)
    seen = 0
    for _ in stream.events():
        seen += 1
        if seen == 3:
            stream.cancel()
    assert seen >= 3
    assert aborted.wait(1.0)
    assert stream.result is None


def test_cancel_from_another_thread_unblocks_a_waiting_consumer() -> None:
    release = threading.Event()

    def run(on_event: EventFn) -> dict[str, Any]:
        release.wait(5)
        return {}

    stream = TurnStream(run, timeout_s=5)
    threading.Timer(0.1, stream.cancel).start()
    started = time.monotonic()
    assert list(stream.events()) == []
    assert time.monotonic() - started < 1.0
    release.set()


_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def test_worker_inherits_the_callers_context() -> None:
    # Log context (request id) and the active trace context are contextvars; worker threads don't inherit
    # them by default.
    _REQUEST_ID.set("req-42")

    def run(on_event: EventFn) -> dict[str, Any]:
        return {"request_id": _REQUEST_ID.get()}

    stream = TurnStream(run, timeout_s=5)
    list(stream.events())
    assert stream.result == {"request_id": "req-42"}


def test_on_finish_runs_when_the_worker_exits_not_when_the_consumer_stops() -> None:
    # A cancelled turn's worker keeps running until its next event; whoever guards the conversation thread
    # must only release it once the worker is really gone.
    release = threading.Event()
    finished = threading.Event()

    def run(on_event: EventFn) -> dict[str, Any]:
        release.wait(5)
        on_event({"type": "token"})  # raises: cancelled
        return {}

    stream = TurnStream(run, timeout_s=5, on_finish=finished.set)
    stream.start()
    stream.cancel()
    assert list(stream.events()) == []
    assert not finished.is_set()
    release.set()
    assert finished.wait(1.0)


def test_start_runs_the_worker_before_anyone_consumes_and_buffers_events() -> None:
    def run(on_event: EventFn) -> dict[str, Any]:
        on_event({"type": "status"})
        return {"ok": True}

    finished = threading.Event()
    stream = TurnStream(run, timeout_s=5, on_finish=finished.set)
    stream.start()
    assert finished.wait(1.0)
    assert list(stream.events()) == [{"type": "status"}]
    assert stream.result == {"ok": True}


def test_deadline_counts_from_start() -> None:
    def run(on_event: EventFn) -> dict[str, Any]:
        time.sleep(5)
        return {}

    stream = TurnStream(run, timeout_s=0.3)
    stream.start()
    time.sleep(0.3)
    started = time.monotonic()
    with pytest.raises(TurnTimeoutError):
        list(stream.events())
    assert time.monotonic() - started < 0.2
