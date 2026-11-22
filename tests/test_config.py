from pathlib import Path

import pytest

from rag.config import ChunkingSettings, HardwareProfile, Settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Keep tests independent of the developer's real .env / config.yaml / shell.
    monkeypatch.chdir(tmp_path)
    for var in (
        "HARDWARE_PROFILE",
        "RAG_CONFIG_FILE",
        "EMBEDDING__MODEL",
        "LLM__GENERATION_MODEL",
        "DEVICE",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_defaults_use_mac_profile() -> None:
    s = Settings()
    assert s.hardware_profile is HardwareProfile.MAC
    assert s.embedding.model == "Qwen/Qwen3-Embedding-0.6B"
    assert s.reranker.model == "Qwen/Qwen3-Reranker-0.6B"
    assert s.reranker.fallback_model == "BAAI/bge-reranker-v2-m3"
    assert s.device == "auto"
    assert s.dtype == "float16"


def test_gpu_profile_selects_4b_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARDWARE_PROFILE", "gpu")
    s = Settings()
    assert s.embedding.model == "Qwen/Qwen3-Embedding-4B"
    assert s.reranker.model == "Qwen/Qwen3-Reranker-4B"
    assert s.dtype == "bfloat16"


def test_cpu_profile_uses_bge_reranker_top_20(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARDWARE_PROFILE", "cpu")
    s = Settings()
    assert s.embedding.model == "Qwen/Qwen3-Embedding-0.6B"
    assert s.reranker.model == "BAAI/bge-reranker-v2-m3"
    assert s.reranker.candidates == 20
    assert s.dtype == "float32"


def test_explicit_model_overrides_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARDWARE_PROFILE", "gpu")
    monkeypatch.setenv("EMBEDDING__MODEL", "custom/embedder")
    s = Settings()
    assert s.embedding.model == "custom/embedder"
    assert s.reranker.model == "Qwen/Qwen3-Reranker-4B"


def test_explicit_device_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVICE", "cpu")
    assert Settings().device == "cpu"


def test_llm_defaults() -> None:
    s = Settings()
    assert s.llm.provider == "anthropic"
    assert s.llm.generation_model == "claude-sonnet-5-5"
    assert s.llm.cheap_model == "claude-haiku-5-5"


def test_yaml_file_is_read_and_env_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "custom.yaml"
    cfg.write_text("hardware_profile: cpu\nllm:\n  generation_model: from-yaml\n  cheap_model: cheap-yaml\n")
    monkeypatch.setenv("RAG_CONFIG_FILE", str(cfg))
    monkeypatch.setenv("LLM__GENERATION_MODEL", "from-env")
    s = Settings()
    assert s.hardware_profile is HardwareProfile.CPU
    assert s.llm.generation_model == "from-env"
    assert s.llm.cheap_model == "cheap-yaml"


def test_dotenv_is_read(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("HARDWARE_PROFILE=gpu\n")
    assert Settings().hardware_profile is HardwareProfile.GPU


def test_chunk_overlap_must_be_a_fraction() -> None:
    with pytest.raises(ValueError):
        ChunkingSettings(overlap_ratio=0.9)


def test_standard_api_key_env_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    s = Settings()
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-test"
    assert "sk-ant-test" not in repr(s)


def test_profile_defaults_do_not_leak_between_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARDWARE_PROFILE", "gpu")
    Settings()
    monkeypatch.setenv("HARDWARE_PROFILE", "mac")
    assert Settings().embedding.model == "Qwen/Qwen3-Embedding-0.6B"


@pytest.mark.parametrize(("profile", "batch"), [("mac", 16), ("gpu", 64), ("cpu", 8)])
def test_embedding_batch_size_comes_from_profile(
    monkeypatch: pytest.MonkeyPatch, profile: str, batch: int
) -> None:
    monkeypatch.setenv("HARDWARE_PROFILE", profile)
    assert Settings().embedding.batch_size == batch


def test_openai_provider_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM__PROVIDER", "openai")
    llm = Settings().llm
    assert (llm.generation_model, llm.upgrade_model, llm.cheap_model) == (
        "gpt-6.2-sol",
        "gpt-6.2-astra",
        "gpt-6-luna",
    )


def test_ollama_provider_defaults_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM__PROVIDER", "ollama")
    monkeypatch.setenv("LLM__CHEAP_MODEL", "llama3:latest")
    llm = Settings().llm
    assert llm.generation_model == "qwen3:8b"
    assert llm.cheap_model == "llama3:latest"


def test_empty_env_values_count_as_unset(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=\nANTHROPIC_API_KEY=\n")
    s = Settings()
    assert s.openai_api_key is None and s.anthropic_api_key is None
