import json
from typing import Any, Literal

import anthropic
import httpx2
import openai
import pytest
from pydantic import BaseModel, Field

from rag.config import TaskSettings
from rag.llm.anthropic_llm import AnthropicLLM
from rag.llm.base import LLMError, Message, strict_json_schema
from rag.llm.openai_llm import OllamaChatLLM, OpenAIResponsesLLM


class Inner(BaseModel):
    claim: str
    citations: list[int] = Field(description="numbers")


class Out(BaseModel):
    intent: Literal["a", "b"]
    note: str | None = Field(default=None, max_length=10)
    count: int = Field(ge=1)
    items: list[Inner]


MODELS = {"cheap": "cheap-model", "generation": "gen-model", "upgrade": "big-model"}
TASKS = {
    "grade": TaskSettings(model="cheap", effort="low", max_tokens=300, thinking=False),
    "generate": TaskSettings(model="generation", max_tokens=900),
}
MESSAGES = [Message("user", "first"), Message("assistant", "reply"), Message("user", "question")]
OUT_JSON = {"intent": "a", "note": None, "count": 2, "items": [{"claim": "c", "citations": [1]}]}


class Recorder:
    def __init__(self, responses: list[httpx2.Response]) -> None:
        self.responses = responses
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        return self.responses.pop(0)


def sse(events: list[tuple[str | None, dict[str, Any] | str]]) -> httpx2.Response:
    lines: list[str] = []
    for name, data in events:
        if name:
            lines.append(f"event: {name}")
        lines.append(f"data: {data if isinstance(data, str) else json.dumps(data)}")
        lines.append("")
    return httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, content="\n".join(lines) + "\n"
    )


def http(rec: Recorder) -> httpx2.Client:
    return httpx2.Client(transport=httpx2.MockTransport(rec))


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


# --- Anthropic -------------------------------------------------------------------------------------


def claude_message(text: str, stop: str = "end_turn") -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "cheap-model",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "s"},
                {"type": "text", "text": text},
            ],
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


def claude(rec: Recorder) -> AnthropicLLM:
    return AnthropicLLM(anthropic.Anthropic(api_key="k", http_client=http(rec)), models=MODELS, tasks=TASKS)


def test_anthropic_structured_uses_native_json_schema_and_task_settings() -> None:
    rec = Recorder([claude_message(json.dumps(OUT_JSON))])
    out = claude(rec).structured("grade", system="SYS", messages=MESSAGES, schema=Out)
    assert out == Out.model_validate(OUT_JSON)
    body = rec.bodies[0]
    assert body["model"] == "cheap-model" and body["max_tokens"] == 300
    assert body["system"] == [{"type": "text", "text": "SYS"}]
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": strict_json_schema(Out)}
    assert body["thinking"] == {"type": "disabled"}
    for forbidden in ("temperature", "top_p", "top_k", "tool_choice", "tools"):
        assert forbidden not in body


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_anthropic_refusal_and_truncation_raise(stop: str) -> None:
    with pytest.raises(LLMError):
        claude(Recorder([claude_message("{}", stop)])).structured(
            "grade", system="S", messages=MESSAGES, schema=Out
        )


def test_anthropic_schema_mismatch_raises() -> None:
    with pytest.raises(LLMError, match="Out"):
        claude(Recorder([claude_message('{"intent": "zzz"}')])).structured(
            "grade", system="S", messages=MESSAGES, schema=Out
        )


def test_anthropic_generate_streams_tokens_and_leaves_thinking_to_the_model() -> None:
    start: dict[str, Any] = {"type": "message", "id": "m", "role": "assistant", "model": "gen-model", "content": [],
             "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0}}  # fmt: skip
    stream = sse(
        [
            ("message_start", {"type": "message_start", "message": start}),
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo [1]"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 2}}),
            ("message_stop", {"type": "message_stop"}),
        ]
    )  # fmt: skip
    rec = Recorder([stream])
    tokens: list[str] = []
    text = claude(rec).generate("generate", system="S", messages=MESSAGES, on_token=tokens.append)
    assert text == "Hello [1]" and tokens == ["Hel", "lo [1]"]
    body = rec.bodies[0]
    assert body["model"] == "gen-model" and body["stream"] is True
    assert (
        "thinking" not in body and "output_config" not in body
    )  # model defaults (Sonnet can't disable thinking)


# --- OpenAI Responses -------------------------------------------------------------------------------


def oa_response(text: str, *, refusal: bool = False, status: str = "completed") -> dict[str, Any]:
    content: dict[str, Any] = (
        {"type": "refusal", "refusal": "no"}
        if refusal
        else {"type": "output_text", "text": text, "annotations": []}
    )
    return {
        "id": "r", "object": "response", "created_at": 0, "status": status, "model": "cheap-model",
        "output": [{"type": "message", "id": "o", "status": "completed", "role": "assistant", "content": [content]}],
        "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
    }  # fmt: skip


def oai(rec: Recorder) -> OpenAIResponsesLLM:
    return OpenAIResponsesLLM(openai.OpenAI(api_key="k", http_client=http(rec)), models=MODELS, tasks=TASKS)


def test_openai_structured_uses_strict_text_format_without_storage() -> None:
    rec = Recorder([httpx2.Response(200, json=oa_response(json.dumps(OUT_JSON)))])
    out = oai(rec).structured("grade", system="SYS", messages=MESSAGES, schema=Out)
    assert out.count == 2
    body = rec.bodies[0]
    assert body["input"][0] == {"role": "developer", "content": "SYS"}
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "Out",
        "schema": strict_json_schema(Out),
        "strict": True,
    }
    assert body["store"] is False and body["max_output_tokens"] == 300
    assert body["reasoning"] == {"effort": "none"}  # thinking disabled for cheap routing calls
    assert "temperature" not in body


@pytest.mark.parametrize("kwargs", [{"refusal": True}, {"status": "incomplete"}])
def test_openai_refusal_and_incomplete_raise(kwargs: dict[str, Any]) -> None:
    rec = Recorder([httpx2.Response(200, json=oa_response("{}", **kwargs))])
    with pytest.raises(LLMError):
        oai(rec).structured("grade", system="S", messages=MESSAGES, schema=Out)


def test_openai_generate_streams_output_text_deltas() -> None:
    events: list[tuple[str | None, dict[str, Any] | str]] = [
        (None, {"type": "response.output_text.delta", "item_id": "o", "output_index": 0, "content_index": 0, "delta": "Hi", "sequence_number": 1, "logprobs": []}),
        (None, {"type": "response.output_text.delta", "item_id": "o", "output_index": 0, "content_index": 0, "delta": " [2]", "sequence_number": 2, "logprobs": []}),
        (None, {"type": "response.completed", "response": oa_response("Hi [2]"), "sequence_number": 3}),
    ]  # fmt: skip
    rec = Recorder([sse(events)])
    tokens: list[str] = []
    assert oai(rec).generate("generate", system="S", messages=MESSAGES, on_token=tokens.append) == "Hi [2]"
    assert tokens == ["Hi", " [2]"] and rec.bodies[0]["model"] == "gen-model"


# --- Ollama (chat completions) -----------------------------------------------------------------------


def chunk(content: str | None, finish: str | None = None) -> dict[str, Any]:
    delta = {"content": content} if content is not None else {}
    return {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}  # fmt: skip


def ollama(rec: Recorder, *, thinking: bool = True) -> OllamaChatLLM:
    client = openai.OpenAI(api_key="ollama", base_url="http://localhost:11434/v1/", http_client=http(rec))
    return OllamaChatLLM(client, models=MODELS, tasks=TASKS, supports_thinking=lambda model: thinking)


def test_ollama_structured_uses_json_schema_response_format_and_disables_thinking() -> None:
    completion = {"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                  "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(OUT_JSON)}}]}  # fmt: skip
    rec = Recorder([httpx2.Response(200, json=completion)])
    assert ollama(rec).structured("grade", system="SYS", messages=MESSAGES, schema=Out).intent == "a"
    body = rec.bodies[0]
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["response_format"]["json_schema"]["schema"] == strict_json_schema(Out)
    assert body["think"] is False


def test_ollama_generate_streams_and_flags_truncation() -> None:
    rec = Recorder([sse([(None, chunk("Hi")), (None, chunk(" there", "stop")), (None, "[DONE]")])])
    tokens: list[str] = []
    assert (
        ollama(rec).generate("generate", system="S", messages=MESSAGES, on_token=tokens.append) == "Hi there"
    )
    assert tokens == ["Hi", " there"]
    rec = Recorder([sse([(None, chunk("Hi", "length")), (None, "[DONE]")])])
    with pytest.raises(LLMError):
        ollama(rec).generate("generate", system="S", messages=MESSAGES)


def test_ollama_omits_thinking_params_for_models_without_thinking() -> None:
    # Ollama rejects think/reasoning_effort with HTTP 400 for models like llama3 (observed live).
    completion = {"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                  "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(OUT_JSON)}}]}  # fmt: skip
    tasks = {**TASKS, "rewrite": TaskSettings(model="cheap", effort="medium")}
    rec = Recorder([httpx2.Response(200, json=completion), httpx2.Response(200, json=completion)])
    client = openai.OpenAI(api_key="ollama", base_url="http://localhost:11434/v1/", http_client=http(rec))
    llm = OllamaChatLLM(client, models=MODELS, tasks=tasks, supports_thinking=lambda model: False)
    llm.structured("grade", system="S", messages=MESSAGES, schema=Out)
    llm.structured("rewrite", system="S", messages=MESSAGES, schema=Out)
    for body in rec.bodies:
        assert "think" not in body and "reasoning_effort" not in body
