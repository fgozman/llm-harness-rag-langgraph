import json
import logging
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pytest

from rag import logs


@pytest.fixture
def records() -> Iterator[list[logging.LogRecord]]:
    captured: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = Capture()
    handler.addFilter(logs.ContextFilter())
    logger = logging.getLogger("test.logs")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield captured
    logger.removeHandler(handler)


def _json(record: logging.LogRecord) -> dict[str, object]:
    return json.loads(logs.JsonFormatter().format(record))


def test_json_line_has_standard_fields_and_extras(records: list[logging.LogRecord]) -> None:
    logging.getLogger("test.logs").info("turn finished", extra={"status": "answered", "latency_ms": 12.5})
    line = _json(records[0])
    assert line["level"] == "INFO"
    assert line["logger"] == "test.logs"
    assert line["message"] == "turn finished"
    assert line["status"] == "answered"
    assert line["latency_ms"] == 12.5
    assert str(line["ts"]).endswith("Z")


def test_bound_context_is_added_and_restored(records: list[logging.LogRecord]) -> None:
    log = logging.getLogger("test.logs")
    with logs.bind(request_id="r1", thread_id="t1"):
        log.info("inside")
        with logs.bind(request_id="r2"):
            log.info("nested")
    log.info("outside")
    inside, nested, outside = (_json(r) for r in records)
    assert (inside["request_id"], inside["thread_id"]) == ("r1", "t1")
    assert (nested["request_id"], nested["thread_id"]) == ("r2", "t1")
    assert "request_id" not in outside


def test_exceptions_are_serialised(records: list[logging.LogRecord]) -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("test.logs").exception("failed")
    line = _json(records[0])
    assert "ValueError: boom" in str(line["exc_info"])


def test_non_json_values_do_not_break_logging(records: list[logging.LogRecord]) -> None:
    logging.getLogger("test.logs").info("odd", extra={"obj": object()})
    assert "object object" in str(_json(records[0])["obj"])


def test_text_format_appends_context(records: list[logging.LogRecord]) -> None:
    with logs.bind(request_id="r9"):
        logging.getLogger("test.logs").info("hello", extra={"status": "ok"})
    text = logs.TextFormatter().format(records[0])
    assert text.startswith("INFO test.logs: hello")
    assert "request_id=r9" in text and "status=ok" in text


def _callback_in_module(module: str, body: Callable[[], None]) -> Callable[[Future[int]], None]:
    """A done-callback whose frame belongs to `module` (stands in for LangGraph's executor callback)."""
    namespace: dict[str, Any] = {"__name__": module, "body": body}
    exec("def callback(future):\n    body()\n", namespace)
    return namespace["callback"]


def test_abandoned_node_shutdown_error_is_dropped_other_errors_kept() -> None:
    seen: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(str(record.exc_info[1]) if record.exc_info else record.getMessage())

    pool = ThreadPoolExecutor(max_workers=1)
    done = pool.submit(lambda: 1)
    done.result()
    pool.shutdown()

    def resubmit() -> None:
        pool.submit(lambda: 2)

    def other() -> None:
        raise RuntimeError("something else")

    handler = Capture()
    logging.getLogger().addHandler(handler)
    try:
        # A finished future runs callbacks immediately; concurrent.futures logs what they raise.
        done.add_done_callback(_callback_in_module("langgraph.pregel.fake", resubmit))  # dropped
        done.add_done_callback(_callback_in_module("someone.else", resubmit))  # not LangGraph: kept
        done.add_done_callback(_callback_in_module("langgraph.pregel.fake", other))  # other error: kept
    finally:
        logging.getLogger().removeHandler(handler)
    assert seen == ["cannot schedule new futures after shutdown", "something else"]
