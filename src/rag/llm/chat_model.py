"""Any LangChain chat model as the pipeline's LLM: structured outputs + token streaming, provider-neutral.

The provider (Anthropic, OpenAI, Ollama, Gemini, Mistral, Bedrock, ...) is chosen by configuration and built
with LangChain's `init_chat_model` (see `rag.factory`); nothing here knows about a specific provider. Provider
failures are translated into `LLMError` at this boundary, keeping the provider's HTTP status when it has one.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from langchain_core.exceptions import ModelConnectionError, ModelError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from rag.config import LLMTask
from rag.llm.base import LLMError, Message, TokenCallback, strict_json_schema
from rag.llm.usage import METER

StructuredMethod = Literal["json_schema", "function_calling", "json_mode"]

# LangChain's normalized finish reasons (response_metadata["normalized_finish_reason"]) that mean the output
# is unusable as an answer.
_FAILED_FINISH = frozenset({"length", "refusal", "context_exceeded", "content_filter", "error"})

# HTTP statuses worth another attempt later (rate limits, timeouts, server errors); everything else is a
# configuration or request problem that every further call would hit too.
_TRANSIENT_STATUSES = frozenset({408, 409, 429})


# --- provider boundary --------------------------------------------------------------------------------


def _status_code(exc: Exception) -> int | None:
    # Provider SDK errors carry the HTTP status as `status_code` (Anthropic, OpenAI, Ollama, ...) or on
    # `response` (httpx-style); duck-typed so no provider package is needed here.
    for candidate in (exc, getattr(exc, "response", None)):
        status = getattr(candidate, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def _cause_chain(exc: BaseException) -> Iterator[BaseException]:
    # SDKs re-raise transport failures as their own types (`raise APITimeoutError(...) from httpx_error`).
    current: BaseException | None = exc
    while current is not None:
        yield current
        current = current.__cause__


def provider_error(exc: Exception) -> LLMError | None:
    """The LLMError for a failure reported by the provider or on the way to it; None for anything else."""
    status = _status_code(exc)
    chain = list(_cause_chain(exc))
    if isinstance(exc, ModelError):
        # LangChain's provider-neutral classification (Anthropic, OpenAI integrations). An unreachable
        # provider is not worth skipping past: every further call would fail the same way.
        transient = exc.is_retryable and not isinstance(exc, ModelConnectionError)
    elif status is not None:
        transient = status in _TRANSIENT_STATUSES or status >= 500
    elif any(isinstance(e, (httpx.TimeoutException, TimeoutError)) for e in chain):
        transient = True  # a slow response (after the integration's retries), e.g. a busy local Ollama
    elif any(isinstance(e, (httpx.TransportError, ConnectionError)) for e in chain):
        transient = False  # provider unreachable
    else:
        return None
    return LLMError(f"{type(exc).__name__}: {exc}", status_code=status, transient=transient)


def call_provider[T](fn: Callable[[], T]) -> T:
    """Run one provider call; provider failures become an LLMError (the original stays as __cause__).

    Anything else (a bug, an invalid option caught by the integration) propagates unchanged, so it is not
    reported as a provider outage.
    """
    try:
        return fn()
    except Exception as exc:
        error = provider_error(exc)
        if error is None:
            raise
        raise error from exc


def stream_provider(chat: BaseChatModel, messages: Sequence[BaseMessage]) -> Iterator[AIMessageChunk]:
    """Stream chunks; provider failures of the stream are translated (as in `call_provider`), the consumer's
    own errors and non-provider errors are not."""
    stream = chat.stream(list(messages))
    try:
        while True:
            try:
                chunk = next(stream)
            except StopIteration:
                return
            except Exception as exc:
                error = provider_error(exc)
                if error is None:
                    raise
                raise error from exc
            yield chunk
    finally:
        # A consumer that stops early (cancelled turn) must not leave the HTTP stream open until GC.
        if isinstance(stream, Generator):
            stream.close()


# --- responses ----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts of one response; `input` excludes cached tokens (they are priced differently)."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0


def usage_of(message: AIMessage) -> Usage | None:
    usage = message.usage_metadata
    if usage is None:
        return None
    details: Mapping[str, Any] = usage.get("input_token_details") or {}
    cache_read = 0
    cache_write = 0
    for key, value in details.items():
        if not isinstance(value, int):
            continue
        if key == "cache_read":
            cache_read += value
        # Standard key, plus per-TTL keys some integrations report instead (e.g. ephemeral_5m_input_tokens).
        elif key == "cache_creation" or key.startswith("ephemeral_"):
            cache_write += value
    # LangChain's input_tokens include cached tokens.
    return Usage(
        input=usage["input_tokens"] - cache_read - cache_write,
        output=usage["output_tokens"],
        cache_read=cache_read,
        cache_write=cache_write,
    )


def record_usage(model: str, message: AIMessage) -> None:
    usage = usage_of(message)
    if usage is not None:
        METER.record(
            model,
            input_tokens=usage.input,
            output_tokens=usage.output,
            cache_read_tokens=usage.cache_read,
            cache_write_tokens=usage.cache_write,
        )


def finish_problem(message: AIMessage, *, max_tokens: int) -> str | None:
    """'length', 'refusal', ... when the response is unusable; None when it finished normally."""
    reason = message.response_metadata.get("normalized_finish_reason")
    if reason is None:
        # Integrations without normalized finish reasons (e.g. Ollama): output at the cap means truncation.
        usage = message.usage_metadata
        return "length" if usage is not None and usage["output_tokens"] >= max_tokens else None
    return str(reason) if reason in _FAILED_FINISH else None


def _raise_for_finish(message: AIMessage, task: str, max_tokens: int) -> None:
    problem = finish_problem(message, max_tokens=max_tokens)
    if problem == "refusal":
        raise LLMError(f"{task}: the model refused to answer")
    if problem == "length":
        raise LLMError(f"{task}: output truncated at max_tokens")
    if problem is not None:
        raise LLMError(f"{task}: response ended with {problem}")


def to_messages(system: str, messages: Sequence[Message]) -> list[BaseMessage]:
    converted: list[BaseMessage] = [SystemMessage(system)]
    for m in messages:
        converted.append(HumanMessage(m.content) if m.role == "user" else AIMessage(m.content))
    return converted


# --- the LLM ------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskModel:
    """The chat model a task runs on (configured with the task's output budget and options)."""

    chat: BaseChatModel
    name: str  # model id, for usage accounting and pricing
    max_tokens: int


class ChatModelLLM:
    def __init__(
        self,
        tasks: Mapping[str, TaskModel],
        *,
        structured_method: StructuredMethod,
        structured_params: Mapping[str, Any] | None = None,
    ) -> None:
        self._tasks = tasks
        self._method: StructuredMethod = structured_method
        self._structured_params = dict(structured_params or {})

    def structured[T: BaseModel](
        self, task: LLMTask, *, system: str, messages: Sequence[Message], schema: type[T]
    ) -> T:
        model = self._tasks[task]
        # Provider-neutral schema (all properties required, no unsupported keywords); validated below.
        runnable = model.chat.with_structured_output(
            strict_json_schema(schema), method=self._method, include_raw=True, **self._structured_params
        )
        result: Any = call_provider(lambda: runnable.invoke(to_messages(system, messages)))
        raw = result["raw"]
        if not isinstance(raw, AIMessage):
            raise LLMError(f"{task}: unexpected response {type(raw).__name__}")
        record_usage(model.name, raw)
        _raise_for_finish(raw, task, model.max_tokens)
        if result["parsing_error"] is not None:
            raise LLMError(f"{task}: structured output is not valid JSON: {result['parsing_error']}")
        try:
            return schema.model_validate(result["parsed"])
        except ValidationError as exc:
            raise LLMError(f"structured output did not match {schema.__name__}: {exc}") from exc

    def generate(
        self,
        task: LLMTask,
        *,
        system: str,
        messages: Sequence[Message],
        on_token: TokenCallback | None = None,
    ) -> str:
        model = self._tasks[task]
        parts: list[str] = []
        full: AIMessageChunk | None = None
        for chunk in stream_provider(model.chat, to_messages(system, messages)):
            full = chunk if full is None else full + chunk
            # `.text` is the answer only: reasoning/thinking blocks are left to the model.
            if text := chunk.text:
                parts.append(text)
                if on_token:
                    on_token(text)
        if full is None:
            raise LLMError(f"{task}: the model returned no response")
        record_usage(model.name, full)
        _raise_for_finish(full, task, model.max_tokens)
        return "".join(parts)
