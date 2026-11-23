import json
from collections.abc import Iterator
from typing import Any, Literal

import httpx
import pytest
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from chat_fakes import ScriptedChatModel, reply, usage
from rag.llm.base import LLMError, Message, strict_json_schema
from rag.llm.chat_model import ChatModelLLM, TaskModel, provider_error
from rag.llm.usage import METER, UsageMeter


class Inner(BaseModel):
    claim: str
    citations: list[int] = Field(description="numbers")


class Out(BaseModel):
    intent: Literal["a", "b"]
    note: str | None = Field(default=None, max_length=10)
    count: int = Field(ge=1)
    items: list[Inner]


MESSAGES = [Message("user", "first"), Message("assistant", "reply"), Message("user", "question")]
OUT_JSON = {"intent": "a", "note": None, "count": 2, "items": [{"claim": "c", "citations": [1]}]}


def make_llm(chat: ScriptedChatModel, max_tokens: int = 300, **structured: Any) -> ChatModelLLM:
    tasks = {
        "grade": TaskModel(chat=chat, name="cheap-model", max_tokens=max_tokens),
        "generate": TaskModel(chat=chat, name="gen-model", max_tokens=max_tokens),
    }
    return ChatModelLLM(tasks, structured_method="json_schema", structured_params=structured)


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.fixture
def meter() -> Iterator[UsageMeter]:
    METER.reset()
    yield METER
    METER.reset()


# --- schema ----------------------------------------------------------------------------------------


def test_strict_schema_requires_everything_and_drops_unsupported_keywords() -> None:
    schema = strict_json_schema(Out)
    assert schema["required"] == ["intent", "note", "count", "items"]
    assert schema["additionalProperties"] is False
    inner = schema["$defs"]["Inner"]
    assert inner["required"] == ["claim", "citations"] and inner["additionalProperties"] is False
    text = json.dumps(schema)
    for keyword in ('"minimum"', '"maxLength"', '"default"'):
        assert keyword not in text
    assert {"type": "null"} in schema["properties"]["note"]["anyOf"]


# --- structured outputs ----------------------------------------------------------------------------


def test_structured_uses_native_schema_and_validates_the_result() -> None:
    chat = ScriptedChatModel(replies=[reply(json.dumps(OUT_JSON))])
    out = make_llm(chat, strict=True).structured("grade", system="S", messages=MESSAGES, schema=Out)
    assert out == Out.model_validate(OUT_JSON)
    (call,) = chat.structured_calls
    assert call == {
        "schema": strict_json_schema(Out),
        "include_raw": True,
        "method": "json_schema",
        "strict": True,
    }
    (messages,) = chat.calls
    assert [type(m) for m in messages] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert [m.text for m in messages] == ["S", "first", "reply", "question"]


@pytest.mark.parametrize(
    ("message", "error"),
    [
        (reply("{}", finish="refusal"), "refused"),
        (reply('{"intent": "a"', finish="length"), "truncated"),
        (reply('{"intent": "a"', finish="content_filter"), "content_filter"),
        (reply("not json"), "not valid JSON"),
        (reply(json.dumps({**OUT_JSON, "intent": "zzz"})), "did not match Out"),
        # No normalized finish reason (e.g. Ollama): output at the cap counts as truncation.
        (reply(json.dumps(OUT_JSON), finish=None, usage=usage(5, 300)), "truncated"),
    ],
)
def test_unusable_structured_outputs_raise(message: AIMessage, error: str) -> None:
    llm = make_llm(ScriptedChatModel(replies=[message]))
    with pytest.raises(LLMError, match=error):
        llm.structured("grade", system="S", messages=MESSAGES, schema=Out)


# --- streaming -------------------------------------------------------------------------------------


def test_generate_streams_tokens_and_returns_the_answer() -> None:
    chat = ScriptedChatModel(replies=[reply("Revenue was 10 [1].")])
    tokens: list[str] = []
    text = make_llm(chat).generate("generate", system="S", messages=MESSAGES, on_token=tokens.append)
    assert text == "Revenue was 10 [1]."
    assert "".join(tokens) == text and len(tokens) > 1


def test_generate_raises_when_the_answer_was_truncated() -> None:
    chat = ScriptedChatModel(replies=[reply("Half an", finish="length")])
    with pytest.raises(LLMError, match="truncated"):
        make_llm(chat).generate("generate", system="S", messages=MESSAGES)


def test_errors_raised_by_the_token_callback_are_not_provider_errors() -> None:
    # A cancelled turn aborts from on_token: that must propagate as-is, not become an LLM error.
    class CancelledTurnError(Exception):
        pass

    def cancel(_: str) -> None:
        raise CancelledTurnError

    chat = ScriptedChatModel(replies=[reply("a b c")])
    with pytest.raises(CancelledTurnError):
        make_llm(chat).generate("generate", system="S", messages=MESSAGES, on_token=cancel)


# --- provider errors -------------------------------------------------------------------------------


@pytest.mark.parametrize(("status", "transient"), [(401, False), (404, False), (429, True), (503, True)])
def test_provider_errors_keep_the_http_status(status: int, transient: bool) -> None:
    chat = ScriptedChatModel(error=StatusError(status))
    with pytest.raises(LLMError) as info:
        make_llm(chat).structured("grade", system="S", messages=MESSAGES, schema=Out)
    assert info.value.status_code == status and info.value.transient is transient
    assert isinstance(info.value.__cause__, StatusError)


def test_streaming_provider_errors_are_translated() -> None:
    chat = ScriptedChatModel(error=ConnectionError("refused"))
    with pytest.raises(LLMError, match="ConnectionError: refused") as info:
        make_llm(chat).generate("generate", system="S", messages=MESSAGES)
    assert info.value.status_code is None and not info.value.transient


def test_generate_raises_when_the_answer_was_content_filtered() -> None:
    chat = ScriptedChatModel(replies=[reply("Partial", finish="content_filter")])
    with pytest.raises(LLMError, match="content_filter"):
        make_llm(chat).generate("generate", system="S", messages=MESSAGES)


def test_non_provider_errors_propagate_unchanged() -> None:
    # A bug (or an option the integration rejects) must not be reported as a provider outage.
    llm = make_llm(ScriptedChatModel(error=TypeError("unexpected keyword argument")))
    with pytest.raises(TypeError):
        llm.structured("grade", system="S", messages=MESSAGES, schema=Out)
    with pytest.raises(TypeError):
        llm.generate("generate", system="S", messages=MESSAGES)


def test_timeouts_are_transient_and_unreachable_providers_are_not() -> None:
    request = httpx.Request("POST", "http://llm")

    class SDKTimeoutError(Exception):  # SDKs re-raise httpx errors as their own types
        pass

    try:
        raise SDKTimeoutError("timed out") from httpx.ReadTimeout("read", request=request)
    except SDKTimeoutError as exc:
        timeout = provider_error(exc)
    assert timeout is not None and timeout.transient and timeout.status_code is None
    unreachable = provider_error(httpx.ConnectError("refused", request=request))
    assert unreachable is not None and not unreachable.transient


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (ModelRateLimitError("slow down"), True),
        (ModelTimeoutError("timed out"), True),
        (ModelConnectionError("unreachable"), False),
        (ModelAuthenticationError("bad key"), False),
    ],
)
def test_langchain_model_errors_are_classified(error: Exception, transient: bool) -> None:
    translated = provider_error(error)
    assert translated is not None and translated.transient is transient


def test_status_code_is_also_read_from_an_http_response() -> None:
    class Response:
        status_code = 500

    class HTTPStyleError(Exception):
        response: Any = Response()

    error = provider_error(HTTPStyleError("boom"))
    assert error is not None and error.status_code == 500


# --- usage -----------------------------------------------------------------------------------------


def test_calls_record_usage_with_cached_tokens_separated(meter: UsageMeter) -> None:
    chat = ScriptedChatModel(
        replies=[
            reply(json.dumps(OUT_JSON), usage=usage(400, 20, cache_read=300, cache_creation=50)),
            reply("Hi there", usage=usage(12, 3)),
        ]
    )
    llm = make_llm(chat)
    llm.structured("grade", system="S", messages=MESSAGES, schema=Out)
    llm.generate("generate", system="S", messages=MESSAGES)
    snap = meter.snapshot()
    cheap = snap["cheap-model"]
    assert (cheap.input_tokens, cheap.cache_read_tokens, cheap.cache_write_tokens) == (50, 300, 50)
    assert (snap["gen-model"].input_tokens, snap["gen-model"].output_tokens) == (12, 3)
