"""Real LLM calls (Anthropic: ANTHROPIC_API_KEY in the repo .env, fractions of a cent; Ollama: local)."""

import os
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
    return Settings()


@pytest.fixture
def anthropic_settings(settings: Settings) -> Settings:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.fail("ANTHROPIC_API_KEY missing from .env")
    return Settings.model_validate(
        {**settings.model_dump(exclude={"llm", "ingestion"}), "llm": {"provider": "anthropic"}}
    )


def _doc_and_chunks(settings: Settings) -> tuple[ParsedDocument, list[Chunk]]:
    assert settings.embedding.model is not None
    doc = DocumentParser(pdf_ocr=False).parse(FIXTURE)
    _, chunks = chunk_document(doc, "doc-1", token_counter(settings.embedding.model), settings.chunking)
    return doc, chunks


def test_contextualization_produces_short_grounded_context(anthropic_settings: Settings) -> None:
    doc, chunks = _doc_and_chunks(anthropic_settings)
    result = build_contextualizer(anthropic_settings).contextualize([(doc, chunks)])
    contexts = [c.context for c in result.chunks[0]]
    assert result.failed_chunks == 0
    assert all(ctx and 0 < len(ctx.split()) < 80 for ctx in contexts), contexts
    assert result.input_tokens > 0 and result.output_tokens > 0
    assert 0 < result.cost_usd < 0.01


@pytest.mark.ollama
def test_ollama_contextualization(settings: Settings) -> None:
    # Uses a locally pulled model; override with OLLAMA_TEST_MODEL (default llama3:latest).
    model = os.environ.get("OLLAMA_TEST_MODEL", "llama3:latest")
    llm = {"provider": "ollama", "cheap_model": model}
    s = Settings.model_validate({**settings.model_dump(exclude={"llm", "ingestion"}), "llm": llm})
    doc, chunks = _doc_and_chunks(s)
    result = build_contextualizer(s).contextualize([(doc, chunks[:3])])
    assert result.failed_chunks == 0
    assert all(c.context for c in result.chunks[0])
    assert result.cost_usd == 0.0  # local model: no pricing
