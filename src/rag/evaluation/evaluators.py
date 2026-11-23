"""Experiment evaluators: deterministic retrieval/abstention/latency/cost metrics and LLM-as-judge scores.

Each evaluator returns {"score", "label", "explanation"}; score None means "not applicable" (e.g. retrieval
metrics on unanswerable questions) so averages aren't diluted. Parameter names (input/output/expected/
metadata)
are how Phoenix binds dataset fields to evaluator arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from phoenix.client.experiments import create_evaluator
from pydantic import BaseModel, Field

from rag.evaluation.metrics import abstention_score, hit_at_k, ndcg_at_k, reciprocal_rank
from rag.llm.base import LLM, Message
from rag.prompts import judge as prompt

Score = dict[str, Any]
_CORRECTNESS_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


class Correctness(BaseModel):
    label: Literal["correct", "partially_correct", "incorrect"]
    explanation: str


class Faithfulness(BaseModel):
    supported_claims: list[str]
    unsupported_claims: list[str]
    explanation: str


class Relevance(BaseModel):
    relevant: bool
    explanation: str


class CitationCheck(BaseModel):
    index: int = Field(description="Cited passage number.")
    supported: bool
    note: str


class CitationReport(BaseModel):
    citations: list[CitationCheck]


class ClosedBookAnswer(BaseModel):
    best_guess: str = Field(description="Your best answer, even if unsure; never empty.")
    confident: bool = Field(description="True only if you are confident the answer is correct.")


def score(value: float | None, label: str, explanation: str = "") -> Score:
    return {"score": value, "label": label, "explanation": explanation}


def not_applicable(reason: str) -> Score:
    return score(None, "n/a", reason)


def _failed(output: dict[str, Any]) -> Score | None:
    return score(None, "error", str(output["error"])) if output.get("error") else None


def _passages(output: dict[str, Any]) -> list[tuple[int, str]]:
    return [(int(p["index"]), str(p["text"])) for p in output.get("passages", [])]


def _user(text: str) -> list[Message]:
    return [Message("user", text)]


class _Judges:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def correctness(self, question: str, reference: str, answer: str) -> Score:
        verdict = self._llm.structured(
            "judge",
            system=prompt.CORRECTNESS,
            messages=_user(prompt.correctness_message(question, reference, answer)),
            schema=Correctness,
        )
        return score(_CORRECTNESS_SCORE[verdict.label], verdict.label, verdict.explanation)

    def faithfulness(self, output: dict[str, Any]) -> Score:
        verdict = self._llm.structured(
            "judge",
            system=prompt.FAITHFULNESS,
            messages=_user(prompt.faithfulness_message(_passages(output), str(output["answer"]))),
            schema=Faithfulness,
        )
        total = len(verdict.supported_claims) + len(verdict.unsupported_claims)
        value = len(verdict.supported_claims) / total if total else 1.0
        label = "faithful" if not verdict.unsupported_claims else "hallucination"
        return score(value, label, "; ".join(verdict.unsupported_claims) or verdict.explanation)

    def relevance(self, question: str, answer: str) -> Score:
        verdict = self._llm.structured(
            "judge",
            system=prompt.RELEVANCE,
            messages=_user(prompt.relevance_message(question, answer)),
            schema=Relevance,
        )
        return score(
            1.0 if verdict.relevant else 0.0,
            "relevant" if verdict.relevant else "irrelevant",
            verdict.explanation,
        )

    def citations(self, output: dict[str, Any]) -> Score:
        report = self._llm.structured(
            "judge",
            system=prompt.CITATIONS,
            messages=_user(prompt.citations_message(_passages(output), str(output["answer"]))),
            schema=CitationReport,
        )
        if not report.citations:
            return not_applicable("judge found no citations")
        supported = sum(c.supported for c in report.citations)
        notes = "; ".join(f"[{c.index}] {c.note}" for c in report.citations if not c.supported)
        return score(
            supported / len(report.citations), f"{supported}/{len(report.citations)} supported", notes
        )


def build_evaluators(judge_llm: LLM, *, k: int, closed_book: bool) -> dict[str, Any]:
    judges = _Judges(judge_llm)

    def retrieval(metric: Callable[[list[str], set[str]], float]) -> Callable[..., Score]:
        def evaluate(output: dict[str, Any], metadata: dict[str, Any]) -> Score:
            if (error := _failed(output)) is not None:
                return error
            gold = set(metadata.get("gold_files") or [])
            if not metadata.get("answerable") or not gold:
                return not_applicable("no gold documents (unanswerable)")
            value = metric(list(output.get("retrieved_files", [])), gold)
            return score(value, f"{value:.2f}")

        return evaluate

    def abstention(output: dict[str, Any], metadata: dict[str, Any]) -> Score:
        if (error := _failed(output)) is not None:
            return error
        answerable, abstained = bool(metadata.get("answerable")), bool(output.get("abstained"))
        value = abstention_score(answerable=answerable, abstained=abstained)
        if answerable:
            return score(value, "false_abstention" if abstained else "answered")
        return score(value, "abstained" if abstained else "hallucinated_answer")

    def correctness(
        input: dict[str, Any], output: dict[str, Any], expected: dict[str, Any], metadata: dict[str, Any]
    ) -> Score:
        if (error := _failed(output)) is not None:
            return error
        if not metadata.get("answerable"):
            # For unanswerable questions the correct answer is an abstention.
            abstained = bool(output.get("abstained"))
            return score(
                1.0 if abstained else 0.0, "correct" if abstained else "incorrect", "unanswerable question"
            )
        if output.get("abstained") and not output.get("answer"):
            return score(0.0, "incorrect", "abstained on an answerable question")
        return judges.correctness(
            str(input["question"]), str(expected["reference_answer"]), str(output["answer"])
        )

    def faithfulness(output: dict[str, Any]) -> Score:
        if (error := _failed(output)) is not None:
            return error
        if output.get("abstained") and not output.get("citations"):
            return not_applicable("abstained without making claims")
        return judges.faithfulness(output)

    def answer_relevance(input: dict[str, Any], output: dict[str, Any]) -> Score:
        if (error := _failed(output)) is not None:
            return error
        return judges.relevance(str(input["question"]), str(output.get("answer", "")))

    def citation_support(output: dict[str, Any]) -> Score:
        if (error := _failed(output)) is not None:
            return error
        if not output.get("citations"):
            return not_applicable("no citations")
        return judges.citations(output)

    def leaky(
        input: dict[str, Any], output: dict[str, Any], expected: dict[str, Any], metadata: dict[str, Any]
    ) -> Score:
        """Closed-book: answered correctly without retrieval -> the question leaks from training data."""
        if not metadata.get("answerable"):
            return not_applicable("unanswerable question")
        if (error := _failed(output)) is not None:
            return error
        if not output.get("answer"):
            return score(0.0, "not_leaky", "no answer")
        # Judged regardless of confidence: an unconfident but correct guess still leaks the answer.
        graded = judges.correctness(
            str(input["question"]), str(expected["reference_answer"]), str(output["answer"])
        )
        is_leaky = graded["label"] == "correct"
        return score(1.0 if is_leaky else 0.0, "leaky" if is_leaky else "not_leaky", graded["explanation"])

    def latency_s(output: dict[str, Any]) -> Score:
        value = output.get("latency_s")
        return (
            score(float(value), f"{value:.1f}s")
            if isinstance(value, int | float)
            else not_applicable("no latency")
        )

    def cost_usd(output: dict[str, Any]) -> Score:
        value = output.get("cost_usd")
        return (
            score(float(value), f"${value:.4f}")
            if isinstance(value, int | float)
            else not_applicable("no cost")
        )

    def code(name: str, fn: Callable[..., Score]) -> Any:
        return create_evaluator(kind="CODE", name=name)(fn)

    def llm(name: str, fn: Callable[..., Score]) -> Any:
        return create_evaluator(kind="LLM", name=name)(fn)

    if closed_book:
        return {
            "correctness": llm("correctness", correctness),
            "abstention": code("abstention", abstention),
            "leaky": llm("leaky", leaky),
            "latency_s": code("latency_s", latency_s),
            "cost_usd": code("cost_usd", cost_usd),
        }
    return {
        "hit_at_k": code("hit_at_k", retrieval(lambda r, g: hit_at_k(r, g, k))),
        "mrr": code("mrr", retrieval(reciprocal_rank)),
        "ndcg_at_k": code("ndcg_at_k", retrieval(lambda r, g: ndcg_at_k(r, g, k))),
        "abstention": code("abstention", abstention),
        "correctness": llm("correctness", correctness),
        "faithfulness": llm("faithfulness", faithfulness),
        "answer_relevance": llm("answer_relevance", answer_relevance),
        "citation_support": llm("citation_support", citation_support),
        "latency_s": code("latency_s", latency_s),
        "cost_usd": code("cost_usd", cost_usd),
    }
