"""Provider backends for contextual enrichment.

- Anthropic: Messages API with explicit prompt caching; Message Batches for large runs.
- OpenAI-compatible Responses API: OpenAI (automatic prefix caching) and Ollama (local); synchronous only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Sequence
from typing import Any, Literal, cast

import anthropic
import openai
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from openai.types.responses import Response

from rag.config import Effort
from rag.ingestion.contextualize import Completion, ContextPrompt, ContextRequest, TokenUsage

log = logging.getLogger(__name__)

CacheTTL = Literal["5m", "1h"]

_BATCH_MAX_REQUESTS = 100_000
# The API limit is 256 MB per batch; stay well below it.
_BATCH_MAX_BYTES = 200 * 1024 * 1024


# --- Anthropic -------------------------------------------------------------------------------------


def build_anthropic_params(
    prompt: ContextPrompt, *, model: str, effort: Effort, max_tokens: int, cache_ttl: CacheTTL
) -> dict[str, Any]:
    # No temperature/top_p/tool_choice: Claude 5.5 models reject non-default sampling params. Thinking is
    # disabled (allowed at effort <= high) because a 1-2 sentence blurb gains nothing from it.
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": prompt.system}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt.document,
                        "cache_control": {"type": "ephemeral", "ttl": cache_ttl},
                    },
                    {"type": "text", "text": prompt.chunk},
                ],
            }
        ],
        "output_config": {"effort": effort},
        "thinking": {"type": "disabled"},
    }


def _anthropic_completion(message: anthropic.types.Message) -> Completion:
    text = "".join(block.text for block in message.content if block.type == "text")
    u = message.usage
    created = u.cache_creation
    return Completion(
        text=text,
        truncated=message.stop_reason == "max_tokens",
        usage=TokenUsage(
            input=u.input_tokens,
            output=u.output_tokens,
            cache_read=u.cache_read_input_tokens or 0,
            cache_write_5m=created.ephemeral_5m_input_tokens
            if created
            else (u.cache_creation_input_tokens or 0),
            cache_write_1h=created.ephemeral_1h_input_tokens if created else 0,
        ),
    )


def _split_batches(requests: Sequence[Request]) -> list[list[Request]]:
    batches: list[list[Request]] = [[]]
    size = 0
    for req in requests:
        req_size = len(json.dumps(req["params"]))
        if batches[-1] and (size + req_size > _BATCH_MAX_BYTES or len(batches[-1]) >= _BATCH_MAX_REQUESTS):
            batches.append([])
            size = 0
        batches[-1].append(req)
        size += req_size
    return batches


class AnthropicBackend:
    supports_batch = True

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        model: str,
        effort: Effort,
        max_tokens: int,
        poll_interval_s: float,
    ) -> None:
        self._client = client
        self._model = model
        self._effort: Effort = effort
        self._max_tokens = max_tokens
        self._poll_interval_s = poll_interval_s

    def _params(self, prompt: ContextPrompt, cache_ttl: CacheTTL) -> dict[str, Any]:
        return build_anthropic_params(
            prompt, model=self._model, effort=self._effort, max_tokens=self._max_tokens, cache_ttl=cache_ttl
        )

    def batch_params(self, prompt: ContextPrompt) -> dict[str, Any]:
        # Batches can run longer than 5 minutes, so the 1h cache keeps shared prefixes alive across them.
        return self._params(prompt, "1h")

    def complete(self, prompt: ContextPrompt) -> Completion:
        # Params are a plain dict (shared with the batch path), so pin the non-streaming overload.
        message: anthropic.types.Message = self._client.messages.create(
            **self._params(prompt, "5m"), stream=False
        )
        return _anthropic_completion(message)

    def complete_batch(self, requests: Sequence[ContextRequest]) -> dict[str, Completion]:
        payload = [
            Request(
                custom_id=r.custom_id,
                params=cast(MessageCreateParamsNonStreaming, self.batch_params(r.prompt)),
            )
            for r in requests
        ]
        batch_ids = [
            self._client.messages.batches.create(requests=group).id for group in _split_batches(payload)
        ]
        results: dict[str, Completion] = {}
        for batch_id in batch_ids:
            self._wait(batch_id)
            for item in self._client.messages.batches.results(batch_id):
                if item.result.type == "succeeded":
                    results[item.custom_id] = _anthropic_completion(item.result.message)
                else:
                    log.warning("batch request %s ended with %s", item.custom_id, item.result.type)
        return results

    def _wait(self, batch_id: str) -> None:
        while self._client.messages.batches.retrieve(batch_id).processing_status != "ended":
            log.info("waiting for contextualization batch %s", batch_id)
            time.sleep(self._poll_interval_s)


# --- OpenAI-compatible Responses API (OpenAI, Ollama) --------------------------------------------------


class OpenAIResponsesBackend:
    """Synchronous Responses API backend. OpenAI caches identical prefixes automatically (>= 1024 tokens)."""

    supports_batch = False

    def __init__(
        self,
        client: openai.OpenAI,
        *,
        model: str,
        max_tokens: int,
        reasoning_effort: str | None,
        extra_body: dict[str, object] | None = None,
        openai_platform: bool = False,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._extra_body = extra_body
        self._openai_platform = openai_platform

    def _request(self, prompt: ContextPrompt) -> dict[str, Any]:
        # Stable content first (developer instructions, then the document), varying chunk last: maximizes the
        # cacheable prefix.
        request: dict[str, Any] = {
            "model": self._model,
            "input": [
                {"role": "developer", "content": prompt.system},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt.document},
                        {"type": "input_text", "text": prompt.chunk},
                    ],
                },
            ],
            "max_output_tokens": self._max_tokens,
        }
        if self._reasoning_effort is not None:
            request["reasoning"] = {"effort": self._reasoning_effort}
        if self._openai_platform:
            # Same document -> same key, so OpenAI routes those requests to the same cache. Document text is
            # untrusted; never keep it server-side.
            request["prompt_cache_key"] = hashlib.sha256(prompt.document.encode()).hexdigest()[:32]
            request["store"] = False
        if self._extra_body:
            request["extra_body"] = self._extra_body
        return request

    def complete(self, prompt: ContextPrompt) -> Completion:
        # The request is a plain dict, so pin the non-streaming overload explicitly.
        response: Response = self._client.responses.create(**self._request(prompt), stream=False)
        usage = response.usage
        cached = usage.input_tokens_details.cached_tokens if usage else 0
        output = usage.output_tokens if usage else 0
        reason = response.incomplete_details.reason if response.incomplete_details else None
        return Completion(
            text=response.output_text,
            # Ollama reports "completed" even when it stops at the cap, so also compare against the limit.
            truncated=reason == "max_output_tokens" or output >= self._max_tokens,
            usage=TokenUsage(
                input=(usage.input_tokens - cached) if usage else 0,
                output=output,
                cache_read=cached,
            ),
        )

    def complete_batch(self, requests: Sequence[ContextRequest]) -> dict[str, Completion]:
        raise NotImplementedError("batching is only implemented for Anthropic")
