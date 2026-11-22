"""Real Anthropic calls. Needs ANTHROPIC_API_KEY in the repo .env; costs a fraction of a cent."""

from pathlib import Path

import pytest

from rag.config import Settings
from rag.factory import build_contextualizer, token_counter
from rag.ingestion.chunking import chunk_document
from rag.ingestion.models import Chunk, ParsedDocument
from rag.ingestion.parsing import DocumentParser

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "docs" / "engineering" / "servo_tuning.md"

pytestmark = pytest.mark.integration


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.chdir(REPO)
    s = Settings()
    if s.anthropic_api_key is None:
        pytest.fail("ANTHROPIC_API_KEY missing from .env")
    return s


def _doc_and_chunks(settings: Settings) -> tuple[ParsedDocument, list[Chunk]]:
    assert settings.embedding.model is not None
    doc = DocumentParser(pdf_ocr=False).parse(FIXTURE)
    _, chunks = chunk_document(doc, "doc-1", token_counter(settings.embedding.model), settings.chunking)
    return doc, chunks


def test_sync_contextualization_produces_short_grounded_context(settings: Settings) -> None:
    settings.ingestion.contextualize_mode = "sync"
    doc, chunks = _doc_and_chunks(settings)
    result = build_contextualizer(settings).contextualize([(doc, chunks)])
    contexts = [c.context for c in result.chunks[0]]
    assert result.failed_chunks == 0
    assert all(ctx and 0 < len(ctx.split()) < 80 for ctx in contexts), contexts
    assert result.input_tokens > 0 and result.output_tokens > 0
    assert 0 < result.cost_usd < 0.01


@pytest.mark.slow
def test_batch_contextualization_round_trip(settings: Settings) -> None:
    settings.ingestion.contextualize_mode = "batch"
    settings.ingestion.batch_poll_interval_s = 5
    doc, chunks = _doc_and_chunks(settings)
    result = build_contextualizer(settings).contextualize([(doc, chunks[:2])])
    assert result.failed_chunks == 0
    assert all(c.context for c in result.chunks[0])
