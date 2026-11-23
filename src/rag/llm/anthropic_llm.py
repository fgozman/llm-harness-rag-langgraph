"""Anthropic Messages API: native JSON-schema structured outputs (no tool forcing) and streaming."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import anthropic
from pydantic import BaseModel

from rag.config import LLMTask, TaskSettings
from rag.llm.base import LLMError, Message, TokenCallback, parse_structured, strict_json_schema


class AnthropicLLM:
    def __init__(
        self, client: anthropic.Anthropic, *, models: dict[str, str], tasks: dict[str, TaskSettings]
    ) -> None:
        self._client = client
        self._models = models
        self._tasks = tasks

    def _params(self, task: LLMTask, system: str, messages: Sequence[Message]) -> dict[str, Any]:
        cfg = self._tasks[task]
        # Claude 5.5 rejects non-default temperature/top_p/top_k, so none are ever sent.
        params: dict[str, Any] = {
            "model": self._models[cfg.model],
            "max_tokens": cfg.max_tokens,
            "system": [{"type": "text", "text": system}],
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        output_config: dict[str, Any] = {}
        if cfg.effort is not None:
            output_config["effort"] = cfg.effort
        if output_config:
            params["output_config"] = output_config
        if cfg.thinking is False:
            params["thinking"] = {"type": "disabled"}
        return params

    def structured[T: BaseModel](
        self, task: LLMTask, *, system: str, messages: Sequence[Message], schema: type[T]
    ) -> T:
        params = self._params(task, system, messages)
        params.setdefault("output_config", {})["format"] = {
            "type": "json_schema",
            "schema": strict_json_schema(schema),
        }
        message: anthropic.types.Message = self._client.messages.create(**params, stream=False)
        _raise_for_stop(message.stop_reason, task)
        # Thinking blocks (if any) are ignored: only the final text carries the JSON.
        text = "".join(b.text for b in message.content if b.type == "text")
        return parse_structured(text, schema)

    def generate(
        self,
        task: LLMTask,
        *,
        system: str,
        messages: Sequence[Message],
        on_token: TokenCallback | None = None,
    ) -> str:
        parts: list[str] = []
        with self._client.messages.stream(**self._params(task, system, messages)) as stream:
            for text in stream.text_stream:
                parts.append(text)
                if on_token:
                    on_token(text)
            final = stream.get_final_message()
        _raise_for_stop(final.stop_reason, task)
        return "".join(parts)


def _raise_for_stop(stop_reason: str | None, task: str) -> None:
    if stop_reason == "refusal":
        raise LLMError(f"{task}: the model refused to answer")
    if stop_reason == "max_tokens":
        raise LLMError(f"{task}: output truncated at max_tokens")
