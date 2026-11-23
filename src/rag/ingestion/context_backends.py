"""Contextual-enrichment backend on any LangChain chat model (provider chosen by configuration)."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from rag.ingestion.contextualize import Completion, ContextPrompt, TokenUsage
from rag.llm.chat_model import call_provider, finish_problem, usage_of


class ChatModelContextBackend:
    """One chat call per chunk: system prompt, then the shared document prefix, then the chunk.

    The stable content comes first so providers with automatic prefix caching (OpenAI, Gemini, ...) reuse it.
    `cache_marker` additionally marks the document block with `cache_control` for providers that need an
    explicit breakpoint (Anthropic-style); LangChain passes it through as part of the content block.
    """

    def __init__(self, chat: BaseChatModel, *, max_tokens: int, cache_marker: bool) -> None:
        self._chat = chat
        self._max_tokens = max_tokens
        self._cache_marker = cache_marker

    def messages(self, prompt: ContextPrompt) -> list[BaseMessage]:
        document: dict[str, Any] = {"type": "text", "text": prompt.document}
        if self._cache_marker:
            document["cache_control"] = {"type": "ephemeral"}
        return [
            SystemMessage(prompt.system),
            HumanMessage(content=[document, {"type": "text", "text": prompt.chunk}]),
        ]

    def complete(self, prompt: ContextPrompt) -> Completion:
        message = call_provider(lambda: self._chat.invoke(self.messages(prompt)))
        usage = usage_of(message)
        return Completion(
            text=message.text,
            truncated=finish_problem(message, max_tokens=self._max_tokens) == "length",
            usage=TokenUsage(
                input=usage.input,
                output=usage.output,
                cache_read=usage.cache_read,
                cache_write=usage.cache_write,
            )
            if usage
            else TokenUsage(),
        )
