"""Phase 5 plumbing against the running Phoenix and the real LLM (closed-book, 2 questions, ~$0.01)."""

from pathlib import Path

import pytest

from rag.config import Settings
from rag.evaluation.runner import run_evaluation

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.integration


def test_closed_book_experiment_runs_in_phoenix_and_saves_a_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(REPO)
    monkeypatch.setenv("PHOENIX__ENABLED", "true")
    settings = Settings()
    settings.evaluation.results_dir = tmp_path
    settings.evaluation.dataset_prefix = "rag-golden-itest"
    settings.evaluation.project_name = "rag-tests"

    report = run_evaluation(settings, closed_book=True, categories=["exact_figure", "unanswerable"], limit=2)

    assert len(report.rows) == 2
    assert set(report.evaluators()) == {"correctness", "abstention", "leaky", "latency_s", "cost_usd"}
    assert (tmp_path / f"{report.name}.json").exists()
    assert report.metadata["closed_book"] is True and report.metadata["examples"] == 2

    from phoenix.client import Client

    datasets = Client(base_url=settings.phoenix.collector_endpoint).datasets.list()
    assert any(str(d["name"]).startswith("rag-golden-itest-2q-") for d in datasets)
