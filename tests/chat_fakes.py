"""A scripted LangChain chat model: the provider-neutral LLM code is tested without any provider package."""

import json
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field


def reply(
    text: str,
    *,
    finish: str | None = "stop",
    usage: UsageMetadata | None = None,
) -> AIMessage:
    metadata: dict[str, Any] = {"normalized_finish_reason": finish} if finish else {}
    return AIMessage(content=text, response_metadata=metadata, usage_metadata=usage)


def usage(
    input_tokens: int, output_tokens: int, *, cache_read: int = 0, cache_creation: int = 0
) -> UsageMetadata:
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_token_details={"cache_read": cache_read, "cache_creation": cache_creation},
    )


class ScriptedChatModel(BaseChatModel):
    """Returns the scripted replies in order; streams them word by word; records every call."""

    model: str = "scripted"
    max_tokens: int | None = None
    base_url: str | None = None
    replies: list[AIMessage] = Field(default_factory=list[AIMessage])
    error: Exception | None = None
    calls: list[list[BaseMessage]] = Field(default_factory=list[list[BaseMessage]])
    structured_calls: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self, messages: Sequence[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return self.replies.pop(0)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message = self._next(messages)
        words = message.text.split(" ")
        for i, word in enumerate(words):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word if i == 0 else f" {word}"))
        # Like real integrations: finish reason and usage arrive on the last chunk.
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="", response_metadata=message.response_metadata, usage_metadata=message.usage_metadata
            )
        )

    def with_structured_output(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, schema: dict[str, Any] | type, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[Any, Any]:
        self.structured_calls.append({"schema": schema, "include_raw": include_raw, **kwargs})

        def run(messages: list[BaseMessage]) -> dict[str, Any]:
            raw = self._next(messages)
            try:
                return {"raw": raw, "parsed": json.loads(raw.text), "parsing_error": None}
            except ValueError as exc:
                return {"raw": raw, "parsed": None, "parsing_error": exc}

        return RunnableLambda(run)
