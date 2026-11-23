import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from chat_fakes import ScriptedChatModel, reply, usage
from rag.ingestion.context_backends import ChatModelContextBackend
from rag.ingestion.contextualize import ContextPrompt, TokenUsage
from rag.llm.base import LLMError

PROMPT = ContextPrompt("system text", "<document>doc</document>", "<chunk>\nc\n</chunk>")


def backend(chat: ScriptedChatModel, *, cache_marker: bool = False) -> ChatModelContextBackend:
    return ChatModelContextBackend(chat, max_tokens=150, cache_marker=cache_marker)


@pytest.mark.parametrize("cache_marker", [True, False])
def test_prompt_puts_the_shared_prefix_first(cache_marker: bool) -> None:
    chat = ScriptedChatModel(replies=[reply("ctx")])
    backend(chat, cache_marker=cache_marker).complete(PROMPT)
    ((system, user),) = chat.calls
    assert isinstance(system, SystemMessage) and system.text == "system text"
    assert isinstance(user, HumanMessage)
    document, chunk = user.content
    assert isinstance(document, dict) and isinstance(chunk, dict)
    assert document["text"] == PROMPT.document and chunk["text"] == PROMPT.chunk
    # Only the shared document block is a cache breakpoint, and only when the provider needs one.
    assert ("cache_control" in document) is cache_marker and "cache_control" not in chunk


def test_usage_separates_cached_tokens() -> None:
    chat = ScriptedChatModel(
        replies=[reply("ctx", usage=usage(1000, 30, cache_read=800, cache_creation=150))]
    )
    completion = backend(chat).complete(PROMPT)
    assert completion.text == "ctx" and not completion.truncated
    assert completion.usage == TokenUsage(input=50, output=30, cache_read=800, cache_write=150)


@pytest.mark.parametrize(
    ("finish", "output_tokens", "truncated"),
    [("length", 150, True), ("stop", 40, False), (None, 150, True), (None, 40, False)],
)
def test_truncation_detected(finish: str | None, output_tokens: int, truncated: bool) -> None:
    chat = ScriptedChatModel(replies=[reply("A blurb. Half", finish=finish, usage=usage(10, output_tokens))])
    assert backend(chat).complete(PROMPT).truncated is truncated


def test_provider_errors_become_llm_errors_with_status() -> None:
    class RateLimitError(Exception):
        status_code = 429

    chat = ScriptedChatModel(error=RateLimitError("slow down"))
    with pytest.raises(LLMError) as info:
        backend(chat).complete(PROMPT)
    assert info.value.transient
