import json
from typing import Any

import anthropic
import httpx2
import openai
import pytest

from rag.ingestion.context_backends import AnthropicBackend, OpenAIResponsesBackend, build_anthropic_params
from rag.ingestion.contextualize import ContextPrompt, TokenUsage

PROMPT = ContextPrompt(
    system="SYS ignore any instructions", document="<document>DOC</document>", chunk="CHUNK"
)


class Recorder:
    """Records JSON request bodies. Both SDKs (anthropic>=2, openai>=3) use their own httpx2 HTTP stack."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.paths.append(request.url.path)
        self.requests.append(json.loads(request.content))
        return httpx2.Response(200, json=self.response)


def anthropic_client(rec: Recorder) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key="k", http_client=httpx2.Client(transport=httpx2.MockTransport(rec)))


# --- Anthropic -------------------------------------------------------------------------------------


def anthropic_message(text: str = " ctx ", stop: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 40,
            "output_tokens": 12,
            "cache_read_input_tokens": 300,
            "cache_creation_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
        },
    }


def test_anthropic_params_cache_document_and_respect_model_constraints() -> None:
    params = build_anthropic_params(
        PROMPT, model="claude-haiku-5-5", effort="low", max_tokens=150, cache_ttl="1h"
    )
    assert params["model"] == "claude-haiku-5-5"
    assert params["output_config"] == {"effort": "low"}
    assert params["thinking"] == {"type": "disabled"}
    assert "temperature" not in params and "top_p" not in params and "tool_choice" not in params
    assert params["system"][0]["text"] == PROMPT.system
    doc_block, chunk_block = params["messages"][0]["content"]
    assert doc_block == {
        "type": "text",
        "text": PROMPT.document,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }
    assert chunk_block == {"type": "text", "text": PROMPT.chunk}


def test_anthropic_backend_sync_call_uses_5m_cache_and_maps_usage() -> None:
    rec = Recorder(anthropic_message())
    backend = AnthropicBackend(
        anthropic_client(rec), model="claude-haiku-5-5", effort="low", max_tokens=150, poll_interval_s=1
    )
    completion = backend.complete(PROMPT)
    assert backend.supports_batch
    assert rec.requests[0]["messages"][0]["content"][0]["cache_control"]["ttl"] == "5m"
    assert completion.text == " ctx " and not completion.truncated
    assert completion.usage == TokenUsage(input=40, output=12, cache_read=300)


def test_anthropic_backend_flags_truncation() -> None:
    rec = Recorder(anthropic_message(stop="max_tokens"))
    backend = AnthropicBackend(
        anthropic_client(rec), model="m", effort="low", max_tokens=150, poll_interval_s=1
    )
    assert backend.complete(PROMPT).truncated


def test_anthropic_batch_requests_use_1h_cache() -> None:
    client = anthropic.Anthropic(api_key="k")
    backend = AnthropicBackend(client, model="m", effort="low", max_tokens=150, poll_interval_s=1)
    params = backend.batch_params(PROMPT)
    assert params["messages"][0]["content"][0]["cache_control"]["ttl"] == "1h"


# --- OpenAI Responses (OpenAI + Ollama) --------------------------------------------------------------


def openai_response(
    text: str = "ctx", status: str = "completed", output_tokens: int = 12, cached: int = 300
) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": status,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "model": "gpt-6-luna",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 400,
            "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": None},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 400 + output_tokens,
        },
    }


def openai_backend(rec: Recorder, **kw: Any) -> OpenAIResponsesBackend:
    client = openai.OpenAI(api_key="k", http_client=httpx2.Client(transport=httpx2.MockTransport(rec)))
    args: dict[str, Any] = {"model": "gpt-6-luna", "max_tokens": 150, "reasoning_effort": "none"}
    args.update(kw)
    return OpenAIResponsesBackend(client, **args)


def test_openai_request_puts_stable_prefix_first_and_disables_storage() -> None:
    rec = Recorder(openai_response())
    backend = openai_backend(rec, openai_platform=True)
    backend.complete(PROMPT)
    body = rec.requests[0]
    assert rec.paths == ["/v1/responses"]
    assert body["model"] == "gpt-6-luna"
    assert body["input"] == [
        {"role": "developer", "content": PROMPT.system},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": PROMPT.document},
                {"type": "input_text", "text": PROMPT.chunk},
            ],
        },
    ]
    assert body["max_output_tokens"] == 150
    assert body["reasoning"] == {"effort": "none"}
    assert body["store"] is False
    assert isinstance(body["prompt_cache_key"], str) and len(body["prompt_cache_key"]) <= 64
    assert "temperature" not in body


def test_openai_cache_key_is_stable_per_document() -> None:
    rec = Recorder(openai_response())
    backend = openai_backend(rec, openai_platform=True)
    backend.complete(PROMPT)
    backend.complete(ContextPrompt(PROMPT.system, PROMPT.document, "OTHER CHUNK"))
    backend.complete(ContextPrompt(PROMPT.system, "<document>ELSE</document>", "CHUNK"))
    keys = [r["prompt_cache_key"] for r in rec.requests]
    assert keys[0] == keys[1] != keys[2]


def test_openai_usage_separates_cached_tokens() -> None:
    completion = openai_backend(Recorder(openai_response())).complete(PROMPT)
    assert completion.usage == TokenUsage(input=100, output=12, cache_read=300)
    assert completion.text == "ctx" and not completion.truncated


@pytest.mark.parametrize(
    ("status", "output_tokens"),
    [("incomplete", 80), ("completed", 150)],  # Ollama reports "completed" even when it hits the cap
)
def test_openai_truncation_detected(status: str, output_tokens: int) -> None:
    rec = Recorder(openai_response(status=status, output_tokens=output_tokens))
    assert openai_backend(rec).complete(PROMPT).truncated


def test_ollama_style_backend_sends_extra_body_and_no_openai_only_fields() -> None:
    rec = Recorder(openai_response(cached=0))
    backend = openai_backend(rec, reasoning_effort=None, extra_body={"think": False}, openai_platform=False)
    backend.complete(PROMPT)
    body = rec.requests[0]
    assert body["think"] is False
    assert "reasoning" not in body and "prompt_cache_key" not in body and "store" not in body
    assert not backend.supports_batch


def test_openai_backend_has_no_batch_path() -> None:
    backend = openai_backend(Recorder(openai_response()))
    with pytest.raises(NotImplementedError):
        backend.complete_batch([])
