import socket
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_CONFIG_FILE", raising=False)


def _closed_port() -> int:
    # Bind then release, so the port is very likely unused and connections are refused fast.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_config_prints_resolved_settings_without_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0, result.output
    assert "Qwen/Qwen3-Embedding-0.6B" in result.output
    assert "claude-sonnet-5-5" in result.output
    assert "sk-ant-supersecret" not in result.output


def test_doctor_fails_when_services_are_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT__URL", f"http://127.0.0.1:{_closed_port()}")
    monkeypatch.setenv("PHOENIX__COLLECTOR_ENDPOINT", f"http://127.0.0.1:{_closed_port()}")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "qdrant" in result.output.lower()
    assert "down" in result.output.lower()


def test_doctor_skips_phoenix_when_tracing_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT__URL", f"http://127.0.0.1:{_closed_port()}")
    monkeypatch.setenv("PHOENIX__ENABLED", "false")
    result = runner.invoke(app, ["doctor"])
    assert "disabled" in result.output.lower()


@pytest.mark.integration
def test_doctor_passes_against_running_services(monkeypatch: pytest.MonkeyPatch) -> None:
    # Uses the repo's .env/config.yaml so it targets the docker compose services.
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
