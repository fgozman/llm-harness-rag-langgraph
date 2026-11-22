from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import SecretStr

from rag.config import Settings
from rag.factory import ConfigurationError, build_contextualizer
from rag.ingestion.context_backends import AnthropicBackend, OpenAIResponsesBackend
from rag.ingestion.contextualize import ContextPrompt

PROMPT = ContextPrompt("sys", "doc", "chunk")


def _word_counter(text: str) -> int:
    return len(text.split())


def _word_counter_factory(model: str) -> Callable[[str], int]:
    return _word_counter


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    monkeypatch.chdir(tmp_path)  # ignore the repo's .env / config.yaml
    # Keep unit tests offline: the real token counter downloads the embedding tokenizer.
    monkeypatch.setattr("rag.factory.token_counter", _word_counter_factory)
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM__PROVIDER", "RAG_CONFIG_FILE"):
        monkeypatch.delenv(var, raising=False)
    return Settings()


def with_provider(settings: Settings, provider: str) -> Settings:
    return Settings.model_validate({**settings.model_dump(), "llm": {"provider": provider}})


def test_anthropic_requires_key(settings: Settings) -> None:
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        build_contextualizer(settings)


def test_anthropic_backend_supports_batch(settings: Settings) -> None:
    settings.anthropic_api_key = SecretStr("k")
    ctx = build_contextualizer(settings)
    assert isinstance(ctx.backend, AnthropicBackend) and ctx.backend.supports_batch


def test_anthropic_rejects_effort_it_does_not_support(settings: Settings) -> None:
    settings.anthropic_api_key = SecretStr("k")
    settings.ingestion.contextualize_effort = "none"
    with pytest.raises(ConfigurationError, match="effort"):
        build_contextualizer(settings)


def test_openai_requires_key(settings: Settings) -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_contextualizer(with_provider(settings, "openai"))


def test_openai_backend_uses_cheap_model_and_no_reasoning_by_default(settings: Settings) -> None:
    s = with_provider(settings, "openai")
    s.openai_api_key = SecretStr("k")
    ctx = build_contextualizer(s)
    backend = ctx.backend
    assert isinstance(backend, OpenAIResponsesBackend) and not backend.supports_batch
    request = backend._request(PROMPT)  # pyright: ignore[reportPrivateUsage]
    assert request["model"] == "gpt-6-luna"
    # Reasoning tokens count against max_output_tokens; a 1-2 sentence blurb needs none.
    assert request["reasoning"] == {"effort": "none"}
    assert request["store"] is False


def test_ollama_needs_no_key_and_disables_thinking(settings: Settings) -> None:
    s = with_provider(settings, "ollama")
    ctx = build_contextualizer(s)
    backend = ctx.backend
    assert isinstance(backend, OpenAIResponsesBackend)
    request = backend._request(PROMPT)  # pyright: ignore[reportPrivateUsage]
    assert request["model"] == "qwen3:8b"
    assert request["extra_body"] == {"think": False}
    assert "reasoning" not in request and "store" not in request
    assert str(backend._client.base_url) == "http://localhost:11434/v1/"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_explicit_batch_mode_is_rejected_without_batch_support(settings: Settings, provider: str) -> None:
    s = with_provider(settings, provider)
    s.openai_api_key = SecretStr("k")
    s.ingestion.contextualize_mode = "batch"
    with pytest.raises(ConfigurationError, match="batch"):
        build_contextualizer(s)
