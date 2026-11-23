from collections.abc import Iterator

import pytest
from openinference.instrumentation import TracerProvider
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

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
