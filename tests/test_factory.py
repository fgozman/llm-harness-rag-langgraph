from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chat_fakes import ScriptedChatModel
from rag.config import Settings
from rag.factory import ConfigurationError, build_chat_model, build_contextualizer, build_llm
from rag.ingestion.context_backends import ChatModelContextBackend
from rag.llm.chat_model import ChatModelLLM


def _word_counter(text: str) -> int:
    return len(text.split())


def _word_counter_factory(model: str) -> Callable[[str], int]:
    return _word_counter


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    monkeypatch.chdir(tmp_path)  # ignore the repo's .env / config.yaml
    # Keep unit tests offline: the real token counter downloads the embedding tokenizer.
    monkeypatch.setattr("rag.factory.token_counter", _word_counter_factory)
    for var in ("LLM__PROVIDER", "RAG_CONFIG_FILE"):
        monkeypatch.delenv(var, raising=False)
    # Integrations validate that a key is configured; no request is ever sent.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return Settings()


def with_llm(settings: Settings, **llm: Any) -> Settings:
    return Settings.model_validate({**settings.model_dump(exclude={"llm", "ingestion"}), "llm": llm})


# --- provider-neutral wiring (no integration package needed) ------------------------------------------


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def init_chat_model(model: str, *, model_provider: str, **kwargs: Any) -> ScriptedChatModel:
        calls.append({"model": model, "model_provider": model_provider, **kwargs})
        return ScriptedChatModel(model=model, **kwargs)

    monkeypatch.setattr("rag.factory.init_chat_model", init_chat_model)
    return calls


def scripted_settings(settings: Settings, **llm: Any) -> Settings:
    models = {"generation_model": "big", "upgrade_model": "bigger", "cheap_model": "small"}
    return with_llm(settings, provider="acme", **models, **llm)


def test_any_langchain_provider_works_with_explicit_models(
    settings: Settings, scripted: list[dict[str, Any]]
) -> None:
    s = scripted_settings(settings, base_url="http://gateway:8080")
    assert isinstance(build_llm(s), ChatModelLLM)
    by_model = {c["model"]: c for c in scripted}
    assert set(by_model) == {"small", "big"}  # analyze/grade/... on cheap, generate/golden on generation
    assert all(c["model_provider"] == "acme" and c["base_url"] == "http://gateway:8080" for c in scripted)
    assert {c["max_tokens"] for c in scripted if c["model"] == "big"} == {8192, 4096}


def test_providers_without_a_preset_need_explicit_models(settings: Settings) -> None:
    with pytest.raises(ValueError, match=r"llm\.generation_model, llm\.upgrade_model, llm\.cheap_model"):
        with_llm(settings, provider="acme")


def test_task_params_reach_the_chat_model(settings: Settings, scripted: list[dict[str, Any]]) -> None:
    s = scripted_settings(settings, tasks={"generate": {"model": "upgrade", "params": {"replies": []}}})
    build_llm(s)
    generate = next(c for c in scripted if c["model"] == "bigger")
    assert generate["replies"] == [] and generate["max_tokens"] == 8192


def test_unknown_params_are_rejected(settings: Settings, scripted: list[dict[str, Any]]) -> None:
    s = scripted_settings(settings, params={"temprature": 0.2})
    with pytest.raises(ConfigurationError, match="does not accept temprature"):
        build_llm(s)


def test_missing_integration_package_is_a_configuration_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def init_chat_model(model: str, **_: Any) -> Any:
        raise ImportError("Unable to import langchain_acme. Please install with `pip install langchain-acme`")

    monkeypatch.setattr("rag.factory.init_chat_model", init_chat_model)
    s = scripted_settings(settings)
    with pytest.raises(ConfigurationError, match="langchain-acme"):
        build_chat_model(s.llm, "small", max_tokens=100, params={})


def test_contextualizer_uses_the_cheap_model_and_its_own_budget(
    settings: Settings, scripted: list[dict[str, Any]]
) -> None:
    s = scripted_settings(settings, prompt_cache_marker=True)
    ctx = build_contextualizer(s)
    assert isinstance(ctx.backend, ChatModelContextBackend)
    (call,) = scripted
    assert (call["model"], call["max_tokens"]) == ("small", 150)
    messages = ctx.backend.messages(ctx_prompt())
    assert "cache_control" in messages[1].content[0]  # pyright: ignore[reportIndexIssue, reportOperatorIssue]


def ctx_prompt() -> Any:
    from rag.ingestion.contextualize import ContextPrompt

    return ContextPrompt("sys", "doc", "chunk")


# --- provider presets (need the optional integration packages) ----------------------------------------


def test_anthropic_preset(settings: Settings) -> None:
    pytest.importorskip("langchain_anthropic")
    llm = build_llm(settings)
    assert isinstance(llm, ChatModelLLM)
    analyze = llm._tasks["analyze"].chat  # pyright: ignore[reportPrivateUsage]
    assert type(analyze).__name__ == "ChatAnthropic"
    fields = analyze.model_dump()
    assert fields["model"] == "claude-haiku-5-5" and fields["max_tokens"] == 512
    assert fields["thinking"] == {"type": "disabled"} and fields["reasoning_effort"] == "low"
    ctx = build_contextualizer(settings)
    assert isinstance(ctx.backend, ChatModelContextBackend)
    assert "cache_control" in ctx.backend.messages(ctx_prompt())[1].content[0]  # pyright: ignore[reportIndexIssue, reportOperatorIssue]


def test_openai_preset(settings: Settings) -> None:
    pytest.importorskip("langchain_openai")
    llm = build_llm(with_llm(settings, provider="openai"))
    assert isinstance(llm, ChatModelLLM)
    fields = llm._tasks["analyze"].chat.model_dump()  # pyright: ignore[reportPrivateUsage]
    assert fields["model_name"] == "gpt-6-luna" and fields["max_tokens"] == 512
    # Responses API without server-side storage; no reasoning for cheap routing calls.
    assert fields["use_responses_api"] is True and fields["store"] is False
    assert fields["reasoning_effort"] == "none"
    assert llm._structured_params == {"strict": True}  # pyright: ignore[reportPrivateUsage]


def test_ollama_preset_uses_num_predict_and_no_key(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("langchain_ollama")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY")
    s = with_llm(settings, provider="ollama", base_url="http://gpu-box:11434")
    llm = build_llm(s)
    assert isinstance(llm, ChatModelLLM)
    fields = llm._tasks["analyze"].chat.model_dump()  # pyright: ignore[reportPrivateUsage]
    assert fields["model"] == "qwen3:8b" and fields["num_predict"] == 512
    assert fields["reasoning"] is False and fields["base_url"] == "http://gpu-box:11434"
    ctx = build_contextualizer(s)
    assert isinstance(ctx.backend, ChatModelContextBackend)
    assert "cache_control" not in ctx.backend.messages(ctx_prompt())[1].content[0]  # pyright: ignore[reportIndexIssue, reportOperatorIssue]
