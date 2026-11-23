"""OpenAI-compatible providers.

- OpenAI: Responses API (`text.format` strict JSON schema; streamed output_text deltas).
- Ollama: Chat Completions (`response_format` json_schema; verified live), because Ollama's Responses
  endpoint does not support `text.format`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import openai
from openai.types.chat import ChatCompletion
from openai.types.responses import Response
from pydantic import BaseModel

from rag.config import LLMTask, TaskSettings
from rag.llm.base import LLMError, Message, TokenCallback, parse_structured, strict_json_schema


class OpenAIResponsesLLM:
    def __init__(
        self, client: openai.OpenAI, *, models: dict[str, str], tasks: dict[str, TaskSettings]
    ) -> None:
        self._client = client
        self._models = models
        self._tasks = tasks

    def _params(self, task: LLMTask, system: str, messages: Sequence[Message]) -> dict[str, Any]:
        cfg = self._tasks[task]
        params: dict[str, Any] = {
            "model": self._models[cfg.model],
            "input": [
                {"role": "developer", "content": system},
                *({"role": m.role, "content": m.content} for m in messages),
            ],
            "max_output_tokens": cfg.max_tokens,
            "store": False,  # documents are untrusted/private: don't retain them server-side
        }
        effort = "none" if cfg.thinking is False else cfg.effort
        if effort is not None:
            params["reasoning"] = {"effort": effort}
        return params

    def structured[T: BaseModel](
        self, task: LLMTask, *, system: str, messages: Sequence[Message], schema: type[T]
    ) -> T:
        params = self._params(task, system, messages)
        params["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema.__name__,
                "schema": strict_json_schema(schema),
                "strict": True,
            }
        }
        response: Response = self._client.responses.create(**params, stream=False)
        _raise_for_response(response, task)
        return parse_structured(response.output_text, schema)

    def generate(
        self,
        task: LLMTask,
        *,
        system: str,
        messages: Sequence[Message],
        on_token: TokenCallback | None = None,
    ) -> str:
        parts: list[str] = []
        for event in self._client.responses.create(**self._params(task, system, messages), stream=True):
            if event.type == "response.output_text.delta":
                parts.append(event.delta)
                if on_token:
                    on_token(event.delta)
            elif event.type == "response.completed":
                _raise_for_response(event.response, task)
            elif event.type in ("response.failed", "response.incomplete"):
                raise LLMError(f"{task}: response {event.type.split('.')[-1]}")
        return "".join(parts)


def _raise_for_response(response: Response, task: str) -> None:
    for item in response.output:
        if item.type == "message" and any(c.type == "refusal" for c in item.content):
            raise LLMError(f"{task}: the model refused to answer")
    if response.status == "incomplete":
        reason = response.incomplete_details.reason if response.incomplete_details else "unknown"
        raise LLMError(f"{task}: incomplete response ({reason})")


class OllamaChatLLM:
    def __init__(
        self,
        client: openai.OpenAI,
        *,
        models: dict[str, str],
        tasks: dict[str, TaskSettings],
        supports_thinking: Callable[[str], bool],
    ) -> None:
        self._client = client
        self._models = models
        self._tasks = tasks
        # Ollama returns HTTP 400 for think/reasoning_effort on models without the "thinking" capability.
        self._supports_thinking = supports_thinking

    def _params(self, task: LLMTask, system: str, messages: Sequence[Message]) -> dict[str, Any]:
        cfg = self._tasks[task]
        model = self._models[cfg.model]
        params: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                *({"role": m.role, "content": m.content} for m in messages),
            ],
            "max_tokens": cfg.max_tokens,
        }
        if not self._supports_thinking(model):
            return params
        if cfg.thinking is False:
            params["extra_body"] = {"think": False}
        elif cfg.effort is not None:
            params["reasoning_effort"] = cfg.effort
        return params

    def structured[T: BaseModel](
        self, task: LLMTask, *, system: str, messages: Sequence[Message], schema: type[T]
    ) -> T:
        params = self._params(task, system, messages)
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": strict_json_schema(schema), "strict": True},
        }
        completion: ChatCompletion = self._client.chat.completions.create(**params, stream=False)
        choice = completion.choices[0]
        if choice.finish_reason == "length":
            raise LLMError(f"{task}: output truncated at max_tokens")
        return parse_structured(choice.message.content or "", schema)

    def generate(
        self,
        task: LLMTask,
        *,
        system: str,
        messages: Sequence[Message],
        on_token: TokenCallback | None = None,
    ) -> str:
        parts: list[str] = []
        for chunk in self._client.chat.completions.create(
            **self._params(task, system, messages), stream=True
        ):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                parts.append(delta)
                if on_token:
                    on_token(delta)
            if chunk.choices[0].finish_reason == "length":
                raise LLMError(f"{task}: output truncated at max_tokens")
        return "".join(parts)
