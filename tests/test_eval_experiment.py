from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from rag.config import Settings
from rag.evaluation.evaluators import (
    CitationCheck,
    CitationReport,
    ClosedBookAnswer,
    Correctness,
    Faithfulness,
    Relevance,
    build_evaluators,
)
from rag.evaluation.experiment import (
    ExperimentReport,
    apply_overrides,
    closed_book_task,
    dataset_records,
    parse_variant,
    run_experiment,
)
from rag.evaluation.golden import GoldenItem, SourceRef
from test_graph import FakeLLM  # pyright: ignore[reportImplicitRelativeImport]


def item(i: int, *, answerable: bool = True, category: str = "exact_figure") -> GoldenItem:
    sources = [
        SourceRef(file_name=f"gold{i}.pdf", file_path=f"/c/gold{i}.pdf", chunk_id=f"c{i}", location="p. 1")
    ]
    return GoldenItem(
        id=f"q{i}",
        category=category,  # type: ignore[arg-type]
        question=f"Question {i}?",
        reference_answer=f"Answer {i}" if answerable else "Not found in the documents.",
        answerable=answerable,
        sources=sources if answerable else [],
        supporting_quote="",
        generator="t",
        verification={},
    )


def rag_output(
    *, abstained: bool = False, files: Sequence[str] = ("x.pdf", "gold1.pdf"), citations: bool = True
) -> dict[str, Any]:
    return {
        "answer": "Not found in the documents." if abstained else "Answer 1 [1].",
        "status": "abstained" if abstained else "answered",
        "abstained": abstained,
        "retrieved_files": list(files),
        "passages": [{"index": 1, "file_name": "gold1.pdf", "text": "Answer 1 is stated here."}],
        "citations": [{"index": 1, "file_name": "gold1.pdf"}] if citations else [],
        "latency_s": 2.5,
        "cost_usd": 0.01,
    }


META: dict[str, Any] = {
    "id": "q1",
    "category": "exact_figure",
    "answerable": True,
    "gold_files": ["gold1.pdf"],
}
UNANSWERABLE_META: dict[str, Any] = {
    "id": "q9",
    "category": "unanswerable",
    "answerable": False,
    "gold_files": [],
}
INPUT = {"question": "Question 1?"}
EXPECTED = {"reference_answer": "Answer 1"}


def evaluate(fn: Any, **kwargs: Any) -> dict[str, Any]:
    result: dict[str, Any] = fn.evaluate(**kwargs)
    return result


# --- evaluators ---------------------------------------------------------------------------------------


def test_retrieval_evaluators_use_gold_documents_and_skip_unanswerable() -> None:
    ev = build_evaluators(FakeLLM(), k=5, closed_book=False)
    assert evaluate(ev["hit_at_k"], output=rag_output(), metadata=META)["score"] == 1.0
    assert evaluate(ev["mrr"], output=rag_output(), metadata=META)["score"] == 0.5
    assert 0 < evaluate(ev["ndcg_at_k"], output=rag_output(), metadata=META)["score"] < 1
    na = evaluate(ev["hit_at_k"], output=rag_output(), metadata=UNANSWERABLE_META)
    assert na["score"] is None and na["label"] == "n/a"


def test_abstention_latency_and_cost_evaluators() -> None:
    ev = build_evaluators(FakeLLM(), k=5, closed_book=False)
    assert (
        evaluate(ev["abstention"], output=rag_output(abstained=True), metadata=UNANSWERABLE_META)["score"]
        == 1.0
    )
    assert evaluate(ev["abstention"], output=rag_output(abstained=True), metadata=META)["score"] == 0.0
    assert evaluate(ev["latency_s"], output=rag_output())["score"] == 2.5
    assert evaluate(ev["cost_usd"], output=rag_output())["score"] == 0.01


def test_correctness_judges_answerable_and_scores_abstention_for_unanswerable() -> None:
    judge = FakeLLM([Correctness(label="partially_correct", explanation="missing unit")])
    ev = build_evaluators(judge, k=5, closed_book=False)
    graded = evaluate(ev["correctness"], input=INPUT, output=rag_output(), expected=EXPECTED, metadata=META)
    assert (graded["score"], graded["label"]) == (0.5, "partially_correct")
    assert "Answer 1" in judge.prompts("Correctness")[0]
    unanswerable = evaluate(
        ev["correctness"],
        input=INPUT,
        output=rag_output(abstained=True),
        expected=EXPECTED,
        metadata=UNANSWERABLE_META,
    )
    assert unanswerable["score"] == 1.0 and judge.count("Correctness") == 1


def test_faithfulness_relevance_and_citation_judges() -> None:
    judge = FakeLLM(
        [
            Faithfulness(supported_claims=["a", "b", "c"], unsupported_claims=["d"], explanation=""),
            Relevance(relevant=True, explanation=""),
            CitationReport(
                citations=[
                    CitationCheck(index=1, supported=True, note=""),
                    CitationCheck(index=2, supported=False, note=""),
                ]
            ),
        ]
    )
    ev = build_evaluators(judge, k=5, closed_book=False)
    assert evaluate(ev["faithfulness"], output=rag_output())["score"] == 0.75
    assert evaluate(ev["answer_relevance"], input=INPUT, output=rag_output())["score"] == 1.0
    assert evaluate(ev["citation_support"], output=rag_output())["score"] == 0.5
    assert "Answer 1 is stated here." in judge.prompts("Faithfulness")[0]


def test_judges_are_skipped_when_there_is_nothing_to_judge() -> None:
    judge = FakeLLM()
    ev = build_evaluators(judge, k=5, closed_book=False)
    assert evaluate(ev["citation_support"], output=rag_output(citations=False))["label"] == "n/a"
    assert evaluate(ev["faithfulness"], output=rag_output(abstained=True, citations=False))["label"] == "n/a"
    assert (
        evaluate(ev["correctness"], input=INPUT, output={"error": "boom"}, expected=EXPECTED, metadata=META)[
            "label"
        ]
        == "error"
    )
    assert judge.calls == []


def test_unconfident_correct_guesses_are_still_leaky() -> None:
    judge = FakeLLM([Correctness(label="correct", explanation="")])
    ev = build_evaluators(judge, k=5, closed_book=True)
    out = {"answer": "Answer 1", "abstained": True, "latency_s": 1.0, "cost_usd": 0.0}
    assert (
        evaluate(ev["leaky"], input=INPUT, output=out, expected=EXPECTED, metadata=META)["label"] == "leaky"
    )


def test_closed_book_evaluators_flag_leaky_questions() -> None:
    judge = FakeLLM(
        [Correctness(label="correct", explanation=""), Correctness(label="incorrect", explanation="")]
    )
    ev = build_evaluators(judge, k=5, closed_book=True)
    assert set(ev) == {"correctness", "abstention", "leaky", "latency_s", "cost_usd"}
    out = {"answer": "Answer 1", "abstained": False, "latency_s": 1.0, "cost_usd": 0.0}
    assert evaluate(ev["leaky"], input=INPUT, output=out, expected=EXPECTED, metadata=META)["score"] == 1.0
    assert evaluate(ev["leaky"], input=INPUT, output=out, expected=EXPECTED, metadata=META)["score"] == 0.0
    assert (
        evaluate(ev["leaky"], input=INPUT, output=out, expected=EXPECTED, metadata=UNANSWERABLE_META)["label"]
        == "n/a"
    )


# --- variants ---------------------------------------------------------------------------------------


def test_parse_variant_and_apply_overrides() -> None:
    name, overrides = parse_variant(
        "ctx:qdrant.collection=test_corpus_ctx,reranker.enabled=false,reranker.top_k=3"
    )
    assert name == "ctx"
    assert overrides == {
        "qdrant.collection": "test_corpus_ctx",
        "reranker.enabled": "false",
        "reranker.top_k": "3",
    }
    s = apply_overrides(Settings(), overrides)
    assert (s.qdrant.collection, s.reranker.enabled, s.reranker.top_k) == ("test_corpus_ctx", False, 3)
    assert parse_variant("baseline") == ("baseline", {})
    with pytest.raises(ValueError):
        parse_variant("bad:novalue")
    with pytest.raises(ValueError):
        apply_overrides(Settings(), {"nope.field": "1"})


# --- experiment runner -------------------------------------------------------------------------------


def test_dataset_records_carry_gold_files_and_categories() -> None:
    inputs, outputs, metadata = dataset_records([item(1), item(9, answerable=False, category="unanswerable")])
    assert inputs[0] == {"question": "Question 1?"} and outputs[0] == {"reference_answer": "Answer 1"}
    assert metadata[0] == {
        "id": "q1",
        "category": "exact_figure",
        "answerable": True,
        "gold_files": ["gold1.pdf"],
    }
    assert metadata[1]["gold_files"] == []


@dataclass
class FakeDataset:
    examples: list[dict[str, Any]]


class FakeExperiments:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def run_experiment(
        self, *, dataset: FakeDataset, task: Callable[..., Any], evaluators: Mapping[str, Any], **kwargs: Any
    ) -> None:
        self.kwargs = kwargs
        for ex in dataset.examples:
            out = task(input=ex["input"], metadata=ex["metadata"])
            for ev in evaluators.values():
                ev.evaluate(input=ex["input"], output=out, expected=ex["output"], metadata=ex["metadata"])


class FakeClient:
    def __init__(self) -> None:
        self.experiments = FakeExperiments()


def test_run_experiment_collects_a_local_report() -> None:
    items = [item(1), item(9, answerable=False, category="unanswerable")]
    inputs, outputs, metadata = dataset_records(items)
    dataset = FakeDataset(
        [{"input": i, "output": o, "metadata": m} for i, o, m in zip(inputs, outputs, metadata, strict=True)]
    )

    def task(input: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        answerable = bool(metadata["answerable"])
        return rag_output(abstained=not answerable, citations=answerable)

    judge = FakeLLM(
        [
            Correctness(label="correct", explanation=""),
            Faithfulness(supported_claims=["a"], unsupported_claims=[], explanation=""),
            Relevance(relevant=True, explanation=""),
            CitationReport(citations=[CitationCheck(index=1, supported=True, note="")]),
            Relevance(relevant=True, explanation=""),
        ]
    )
    client = FakeClient()
    report = run_experiment(
        client, dataset, task=task, evaluators=build_evaluators(judge, k=5, closed_book=False),
        name="exp-1", metadata={"variant": "baseline"},
    )  # fmt: skip
    assert isinstance(report, ExperimentReport)
    assert client.experiments.kwargs["experiment_name"] == "exp-1"
    assert report.mean("correctness") == 1.0 and report.mean("abstention") == 1.0
    assert report.mean("hit_at_k") == 1.0  # the unanswerable example is excluded, not counted as 0
    assert report.by_category("abstention") == {"exact_figure": 1.0, "unanswerable": 1.0}
    assert report.rows["q9"]["output"]["status"] == "abstained"


def test_closed_book_task_always_keeps_the_best_guess() -> None:
    # The model must not opt out: a guess that happens to be right is exactly what "leaky" detects.
    llm = FakeLLM([ClosedBookAnswer(best_guess="$0.008 per share", confident=False)])
    out = closed_book_task(llm, pricing={})(input={"question": "Q?"}, metadata=META)
    assert out["answer"] == "$0.008 per share" and out["abstained"] is True
    assert llm.calls[0][0] == "closed_book"
