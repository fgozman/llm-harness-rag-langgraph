import socket
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from graph_fakes import THREE, FakeLLM, FakeRetriever, analysis, grades, make, ok, plan
from rag.cli import app
from rag.llm.base import LLMError, Message, TokenCallback

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


class RecordingPipeline:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def run(self, root: Path, **kwargs: object) -> object:
        from rag.ingestion.pipeline import IngestStats

        self.kwargs = kwargs
        return IngestStats()


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], {"force": False, "retry_failed": False}),
        (["--force", "--retry-failed"], {"force": True, "retry_failed": True}),
    ],
)
def test_ingest_passes_flags_to_the_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flags: list[str], expected: dict[str, bool]
) -> None:
    pipeline = RecordingPipeline()

    def fake_build(settings: object) -> RecordingPipeline:
        return pipeline

    monkeypatch.setattr("rag.factory.build_ingestion_pipeline", fake_build)
    result = runner.invoke(app, ["ingest", str(tmp_path), "--no-contextualize", *flags])
    assert result.exit_code == 0, result.output
    assert {k: pipeline.kwargs[k] for k in expected} == expected


class _SlowLLM(FakeLLM):
    def generate(
        self, task: Any, *, system: str, messages: Sequence[Message], on_token: TokenCallback | None = None
    ) -> str:
        time.sleep(1.0)
        return super().generate(task, system=system, messages=messages, on_token=on_token)


def _patch_graph(monkeypatch: pytest.MonkeyPatch, llm: FakeLLM) -> None:
    graph = make(llm, FakeRetriever([THREE]))

    def build(settings: object) -> Any:
        return graph

    monkeypatch.setattr("rag.factory.build_qa_graph", build)


def test_ask_prints_the_answer_and_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_graph(monkeypatch, FakeLLM([analysis(), plan(), grades(True, False, True), ok()], ["Rev 10 [1]."]))
    result = runner.invoke(app, ["ask", "What was revenue?"])
    assert result.exit_code == 0, result.output
    assert "Rev 10 [1]." in result.output
    assert "a.pdf" in result.output


def test_ask_times_out_at_the_turn_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH__TURN_TIMEOUT_S", "0.3")
    _patch_graph(monkeypatch, _SlowLLM([analysis(), plan(), grades(True, False, True), ok()], ["late"]))
    started = time.monotonic()
    result = runner.invoke(app, ["ask", "What was revenue?"])
    assert result.exit_code == 1
    assert "Timed out" in result.output
    assert time.monotonic() - started < 1.0


class _FailOnceLLM(FakeLLM):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.failed = False

    def generate(
        self, task: Any, *, system: str, messages: Sequence[Message], on_token: TokenCallback | None = None
    ) -> str:
        if not self.failed:
            self.failed = True
            raise LLMError("generate: output truncated at max_tokens")
        return super().generate(task, system=system, messages=messages, on_token=on_token)


def test_chat_reports_a_failed_turn_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    turn = [analysis(), plan(), grades(True, False, True)]
    _patch_graph(monkeypatch, _FailOnceLLM([*turn, *turn, ok()], ["Rev 10 [1]."]))
    result = runner.invoke(app, ["chat"], input="What was revenue?\nWhat was revenue?\nexit\n")
    assert result.exit_code == 0, result.output
    assert "LLM error:" in result.output
    assert "Rev 10 [1]." in result.output


def test_serve_runs_uvicorn_without_its_own_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def run(app_: object, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("uvicorn.run", run)
    result = runner.invoke(app, ["serve", "--port", "8123"])
    assert result.exit_code == 0, result.output
    assert calls == [{"host": "127.0.0.1", "port": 8123, "log_config": None, "access_log": False}]
