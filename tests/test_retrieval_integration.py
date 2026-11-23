"""Phase 2 acceptance, automated: real embedders, Qdrant and reranker (no LLM)."""

import re
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag.cli import app
from rag.config import Settings
from rag.factory import build_index, build_ingestion_pipeline, build_reranker, build_retriever
from rag.retrieval.retriever import Retriever, SearchMode

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "tests" / "fixtures" / "retrieval"
CODES = re.findall(r"^## Valve (HX-\d{4}-[A-Z])$", (CATALOG / "valve_catalog.md").read_text(), re.M)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Settings]:
    mp = pytest.MonkeyPatch()
    mp.chdir(REPO)
    s = Settings()
    s.qdrant.collection = f"test_retrieval_{uuid.uuid4().hex[:8]}"
    s.state_dir = tmp_path_factory.mktemp("state")
    s.ingestion.contextualize = False
    stats = build_ingestion_pipeline(s).run(CATALOG)
    assert stats.indexed == 1 and stats.failed == 0
    yield s
    build_index(s).client.delete_collection(s.qdrant.collection)
    mp.undo()


@pytest.fixture(scope="module")
def retriever(settings: Settings) -> Retriever:
    return build_retriever(settings)


# Two keyword-heavy question types: by part code, and by seal-kit number (appears once, mid-sentence).
QUESTIONS = [
    *((code, f"What is the maximum working pressure of valve {code}?") for code in CODES),
    *((code, f"Which valve uses seal kit K-{code[3:]}?") for code in CODES),
]


def rank_of(retriever: Retriever, code: str, question: str, mode: SearchMode, *, rerank: bool = False) -> int:
    """Rank of the code's own section in the top 12 (99 = not retrieved)."""
    result = retriever.retrieve([question], mode=mode, rerank=rerank, top_k=12)
    for rank, chunk in enumerate(result.ranked, start=1):
        if f"The {code} pressure-relief valve" in str(chunk.payload["text"]):
            return rank
    return 99


def mrr(ranks: list[int]) -> float:
    return sum(1 / r for r in ranks) / len(ranks)


def test_hybrid_beats_dense_only_on_exact_codes(retriever: Retriever) -> None:
    assert len(CODES) == 12
    dense = [rank_of(retriever, code, q, "dense") for code, q in QUESTIONS]
    hybrid = [rank_of(retriever, code, q, "hybrid") for code, q in QUESTIONS]
    print(f"dense ranks={dense} MRR={mrr(dense):.3f}\nhybrid ranks={hybrid} MRR={mrr(hybrid):.3f}")
    assert all(h <= d for h, d in zip(hybrid, dense, strict=True)), "hybrid is never worse than dense-only"
    assert mrr(hybrid) > mrr(dense)
    assert sum(r == 1 for r in hybrid) > sum(r == 1 for r in dense)


def test_reranking_resolves_the_remaining_code_confusions(retriever: Retriever) -> None:
    reranked = [rank_of(retriever, code, q, "hybrid", rerank=True) for code, q in QUESTIONS]
    print(f"hybrid+rerank ranks={reranked} MRR={mrr(reranked):.3f}")
    assert all(r == 1 for r in reranked)


def test_qwen3_reranker_reproduces_model_card_example(settings: Settings) -> None:
    reranker = build_reranker(settings)
    assert reranker is not None
    yes, no = reranker.rerank(
        "What is the capital of China?", ["The capital of China is Beijing.", "Gravity is a force."]
    )
    assert yes > 0.9 and no < 0.1


def test_reranker_puts_the_exact_valve_first(retriever: Retriever) -> None:
    result = retriever.retrieve(["What is the maximum working pressure of valve HX-2219-B?"], top_k=3)
    assert "The HX-2219-B pressure-relief valve" in str(result.ranked[0].payload["text"])
    (passage, *_) = result.passages
    assert passage.file_name == "valve_catalog.md" and "HX-2219-B" in passage.text


def test_search_cli_prints_ranked_sources(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT__COLLECTION", settings.qdrant.collection)
    monkeypatch.setenv("STATE_DIR", str(settings.state_dir))
    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(
        app, ["search", "service interval of HX-2912-B", "-k", "3", "--file", "valve_catalog.md"]
    )
    assert result.exit_code == 0, result.output
    assert "valve_catalog.md" in result.output and "HX-2912-B" in result.output
    assert "rerank" in result.output and "candidates" in result.output
