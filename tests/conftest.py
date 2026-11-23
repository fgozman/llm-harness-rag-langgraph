from collections.abc import Iterator
from pathlib import Path

import pytest
from dotenv import load_dotenv
from openinference.instrumentation import TracerProvider
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

REPO = Path(__file__).resolve().parents[1]

_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))


@pytest.fixture(scope="session")
def tracer_provider() -> TracerProvider:
    # The global provider can be set only once per process; spans from other tests are harmless.
    trace.set_tracer_provider(_PROVIDER)
    return _PROVIDER


@pytest.fixture
def spans(tracer_provider: TracerProvider) -> Iterator[InMemorySpanExporter]:
    _EXPORTER.clear()
    yield _EXPORTER
    _EXPORTER.clear()


@pytest.fixture(autouse=True)
def _no_phoenix_export(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Unit tests must never export to a real Phoenix (another local project may be listening on :6006).
    from rag import telemetry

    monkeypatch.setenv("PHOENIX__ENABLED", "false")
    telemetry.reset_for_tests()
    yield
    telemetry.reset_for_tests()


@pytest.fixture(scope="session", autouse=True)
def _provider_keys_for_integration_tests(request: pytest.FixtureRequest) -> None:
    # Like the CLI: LangChain integrations read API keys from the environment, and real-LLM tests take them
    # from the repo .env. Unit-only runs (integration tests deselected) never see real keys.
    if any(item.get_closest_marker("integration") for item in request.session.items):
        load_dotenv(REPO / ".env", override=False)
