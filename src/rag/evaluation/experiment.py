"""Run the QA graph (or the closed-book baseline) over the golden set as a Phoenix experiment.

Results are also collected locally (eval/results/<experiment>.json) so reports, comparisons and leaky-question
flags don't depend on reading Phoenix back.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from phoenix.client.experiments import create_evaluator

from rag.config import ModelPricing, Settings
from rag.evaluation.evaluators import ClosedBookAnswer, Score
from rag.evaluation.golden import GoldenItem
from rag.graph.build import run_turn
from rag.llm.base import LLM, Message
from rag.llm.usage import METER
from rag.prompts import judge as judge_prompts

Task = Callable[..., dict[str, Any]]
_MAX_PASSAGE_CHARS = 6000  # judges see what the generator saw, trimmed to keep experiment payloads small

# --- variants ----------------------------------------------------------------------------------------


def parse_variant(spec: str) -> tuple[str, dict[str, str]]:
    """'name:a.b=1,c=x' -> ('name', {'a.b': '1', 'c': 'x'}); a bare name means no overrides."""
    name, _, rest = spec.partition(":")
    overrides: dict[str, str] = {}
    for part in filter(None, (p.strip() for p in rest.split(","))):
        key, sep, value = part.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"invalid override {part!r}; expected key=value")
        overrides[key.strip()] = value.strip()
    return name.strip(), overrides


def apply_overrides(settings: Settings, overrides: Mapping[str, str]) -> Settings:
    data: dict[str, Any] = settings.model_dump()
    for dotted, value in overrides.items():
        target = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            if not isinstance(target.get(key), dict):
                raise ValueError(f"unknown setting {dotted!r}")
            target = target[key]
        if leaf not in target:
            raise ValueError(f"unknown setting {dotted!r}")
        target[leaf] = value  # pydantic coerces "false"/"3" to the field's type
    return Settings.model_validate(data)


# --- dataset -----------------------------------------------------------------------------------------


def dataset_records(
    items: Sequence[GoldenItem],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = [{"question": i.question} for i in items]
    outputs = [{"reference_answer": i.reference_answer} for i in items]
    metadata = [
        {
            "id": i.id,
            "category": i.category,
            "answerable": i.answerable,
            "gold_files": sorted({s.file_name for s in i.sources}),
        }
        for i in items
    ]
    return inputs, outputs, metadata


class _Datasets(Protocol):
    def list(self) -> list[Any]: ...

    def get_dataset(self, *, dataset: str) -> Any: ...

    def create_dataset(self, **kwargs: Any) -> Any: ...


def ensure_dataset(datasets: _Datasets, items: Sequence[GoldenItem], prefix: str) -> Any:
    """Content-addressed name: an edited golden set becomes a new dataset; an unchanged one is reused."""
    digest = hashlib.sha256("".join(i.model_dump_json() for i in items).encode()).hexdigest()[:8]
    name = f"{prefix}-{len(items)}q-{digest}"
    # Existence is checked by listing: get_dataset reports "not found" only as a generic ValueError, and any
    # other failure (Phoenix down, auth) must surface instead of triggering a create.
    if any(d["name"] == name for d in datasets.list()):
        return datasets.get_dataset(dataset=name)
    inputs, outputs, metadata = dataset_records(items)
    return datasets.create_dataset(
        name=name,
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        dataset_description="Golden QA set over the post-cutoff test corpus (see eval/golden.jsonl).",
    )


# --- tasks -------------------------------------------------------------------------------------------


def rag_task(graph: Any, pricing: Mapping[str, ModelPricing]) -> Task:
    def task(input: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        METER.reset()
        started = time.perf_counter()
        try:
            state = run_turn(
                graph, str(input["question"]), thread_id=f"eval-{metadata['id']}-{uuid.uuid4().hex[:6]}"
            )
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "latency_s": time.perf_counter() - started}
        final = state["final"]
        return {
            "answer": final.text,
            "status": final.status,
            "abstained": final.status == "abstained",
            "citations": [
                c.model_dump(include={"index", "file_name", "page_start", "section"}) for c in final.citations
            ],
            "retrieved_files": [str(c.payload.get("file_name", "")) for c in state.get("reranked", [])],
            "passages": [
                {"index": p.index, "file_name": p.file_name, "text": p.text[:_MAX_PASSAGE_CHARS]}
                for p in state.get("answer_passages", [])
            ],
            "queries": state.get("queries", []),
            "retrieval_attempts": state.get("retrieval_attempts", 0),
            "generation_count": state.get("generation_count", 0),
            "latency_s": time.perf_counter() - started,
            "cost_usd": METER.cost_usd(pricing),
        }

    return task


def closed_book_task(llm: LLM, pricing: Mapping[str, ModelPricing]) -> Task:
    """Same questions, no retrieval: the generation model answers from its own knowledge."""

    def task(input: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        METER.reset()
        started = time.perf_counter()
        try:
            reply = llm.structured(
                "closed_book",
                system=judge_prompts.CLOSED_BOOK,
                messages=[Message("user", str(input["question"]))],
                schema=ClosedBookAnswer,
            )
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "latency_s": time.perf_counter() - started}
        return {
            # Always the best guess, so leaky questions are detectable; confidence drives abstention.
            "answer": reply.best_guess,
            "abstained": not reply.confident,
            "latency_s": time.perf_counter() - started,
            "cost_usd": METER.cost_usd(pricing),
        }

    return task


# --- running + reporting -----------------------------------------------------------------------------


@dataclass
class ExperimentReport:
    name: str
    metadata: dict[str, Any]
    rows: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])

    def _row(self, example_id: str) -> dict[str, Any]:
        return self.rows.setdefault(example_id, {"scores": {}})

    def scores(self, evaluator: str) -> list[tuple[str, float]]:
        return [
            (str(row.get("category")), float(row["scores"][evaluator]["score"]))
            for row in self.rows.values()
            if evaluator in row["scores"] and row["scores"][evaluator].get("score") is not None
        ]

    def mean(self, evaluator: str) -> float | None:
        values = [v for _, v in self.scores(evaluator)]
        return statistics.fmean(values) if values else None

    def percentile(self, evaluator: str, q: float) -> float | None:
        values = sorted(v for _, v in self.scores(evaluator))
        return values[min(len(values) - 1, round(q * (len(values) - 1)))] if values else None

    def by_category(self, evaluator: str) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for category, value in self.scores(evaluator):
            grouped[category].append(value)
        return {c: statistics.fmean(v) for c, v in sorted(grouped.items())}

    def failed(self) -> list[str]:
        """Examples whose task raised; they have no scores, so the means above exclude them."""
        failed: list[str] = []
        for example_id, row in self.rows.items():
            # Every row has an output: the task runs (and records it) before any evaluator.
            if "error" in row["output"]:
                failed.append(example_id)
        return sorted(failed)

    def evaluators(self) -> list[str]:
        return sorted({name for row in self.rows.values() for name in row["scores"]})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"name": self.name, "metadata": self.metadata, "rows": self.rows}, indent=1, default=str
            )
        )

    @classmethod
    def load(cls, path: Path) -> ExperimentReport:
        data = json.loads(path.read_text())
        return cls(name=data["name"], metadata=data["metadata"], rows=data["rows"])


class _Experiments(Protocol):
    def run_experiment(self, **kwargs: Any) -> Any: ...


class _Client(Protocol):
    @property
    def experiments(self) -> _Experiments: ...


def run_experiment(
    client: _Client,
    dataset: Any,
    *,
    task: Task,
    evaluators: Mapping[str, Any],
    name: str,
    metadata: dict[str, Any],
    description: str = "",
) -> ExperimentReport:
    report = ExperimentReport(name=name, metadata=metadata)

    def recorded_task(input: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        output = task(input=input, metadata=metadata)
        row = report._row(str(metadata["id"]))  # pyright: ignore[reportPrivateUsage]
        row.update(category=metadata.get("category"), answerable=metadata.get("answerable"), output=output)
        return output

    def recorded(evaluator: Any) -> Any:
        def evaluate(
            input: dict[str, Any], output: dict[str, Any], expected: dict[str, Any], metadata: dict[str, Any]
        ) -> Score:
            result: Score = evaluator.evaluate(
                input=input, output=output, expected=expected, metadata=metadata
            )
            report._row(str(metadata["id"]))["scores"][evaluator.name] = result  # pyright: ignore[reportPrivateUsage]
            return result

        return create_evaluator(kind=evaluator.kind, name=evaluator.name)(evaluate)

    client.experiments.run_experiment(
        dataset=dataset,
        task=recorded_task,
        evaluators={key: recorded(ev) for key, ev in evaluators.items()},
        experiment_name=name,
        experiment_description=description,
        experiment_metadata=metadata,
        timeout=900,  # one QA turn can take a minute or more on the full corpus
        print_summary=False,
    )
    return report
