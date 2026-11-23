"""Orchestration behind `rag golden` / `rag eval` / `rag compare` (wires settings, models and Phoenix)."""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag import telemetry
from rag.config import Settings
from rag.evaluation.evaluators import build_evaluators
from rag.evaluation.experiment import (
    ExperimentReport,
    apply_overrides,
    closed_book_task,
    ensure_dataset,
    parse_variant,
    rag_task,
    run_experiment,
)
from rag.evaluation.golden import (
    DEFAULT_QUOTAS,
    GoldenGenerator,
    GoldenItem,
    canary_items,
    coverage_summary,
    load_golden,
    load_pool,
    plan_candidates,
    save_golden,
)
from rag.graph.build import PROMPT_VERSIONS
from rag.llm.usage import METER
from rag.prompts import golden as golden_prompt
from rag.prompts import judge as judge_prompt

log = logging.getLogger(__name__)
CLOSED_BOOK = "closed_book"


def _phoenix_client(settings: Settings) -> Any:
    from phoenix.client import Client

    key = settings.phoenix.api_key.get_secret_value() if settings.phoenix.api_key else None
    return Client(base_url=settings.phoenix.collector_endpoint, api_key=key)


# --- golden set --------------------------------------------------------------------------------------


def generate_golden(settings: Settings) -> tuple[list[GoldenItem], float]:
    """Build the golden set from the indexed corpus; returns items and LLM cost."""
    from rag.factory import build_index, build_llm, build_retriever

    cfg = settings.evaluation
    METER.reset()
    llm = build_llm(settings)
    pool = load_pool(build_index(settings).client, settings.qdrant.collection)
    plan = plan_candidates(pool, random.Random(cfg.seed))
    generator = GoldenGenerator(
        llm, generator_name=f"{settings.llm.generation_model} ({golden_prompt.VERSION})"
    )
    items: list[GoldenItem] = []
    for category, quota in DEFAULT_QUOTAS.items():
        produced = generator.generate_category(category, plan[category], quota)
        log.info("%s: %d/%d items from %d candidates", category, len(produced), quota, len(plan[category]))
        items += produced

    retriever = build_retriever(settings)

    def retrieve(question: str) -> list[tuple[str, str]]:
        return [(p.file_name, p.text) for p in retriever.retrieve([question]).passages]

    items += generator.unanswerable(coverage_summary(pool), cfg.unanswerable, retrieve=retrieve)
    canaries = canary_items(cfg.canary_facts_path, corpus_dir=cfg.corpus_dir)
    items += [c for c in canaries if c.answerable][: cfg.canary_answerable]
    items += [c for c in canaries if not c.answerable][: cfg.canary_unanswerable]
    save_golden(items, cfg.golden_path)
    return items, METER.cost_usd(settings.llm.pricing)


def upload_golden(settings: Settings) -> tuple[str, int]:
    items = load_golden(settings.evaluation.golden_path)
    dataset = ensure_dataset(_phoenix_client(settings).datasets, items, settings.evaluation.dataset_prefix)
    return str(dataset.name), len(items)


# --- experiments -------------------------------------------------------------------------------------


def _select(items: Sequence[GoldenItem], *, categories: Sequence[str], limit: int | None) -> list[GoldenItem]:
    chosen = [i for i in items if not categories or i.category in categories]
    return chosen[:limit] if limit else chosen


def run_evaluation(
    base: Settings,
    *,
    variant: str = "baseline",
    closed_book: bool = False,
    categories: Sequence[str] = (),
    limit: int | None = None,
) -> ExperimentReport:
    from langgraph.checkpoint.memory import InMemorySaver

    from rag.factory import build_llm, build_retriever
    from rag.graph.build import GraphDeps, build_graph, state_serializer

    name, overrides = parse_variant(variant)
    settings = apply_overrides(base, overrides)
    settings.phoenix.project_name = settings.evaluation.project_name
    telemetry.init_tracing(settings.phoenix)
    items = _select(load_golden(settings.evaluation.golden_path), categories=categories, limit=limit)
    client = _phoenix_client(settings)
    dataset = ensure_dataset(client.datasets, items, settings.evaluation.dataset_prefix)
    llm = build_llm(settings)
    if closed_book:
        name = CLOSED_BOOK
        task = closed_book_task(llm, settings.llm.pricing)
    else:
        graph = build_graph(
            GraphDeps(llm=llm, retriever=build_retriever(settings), settings=settings.graph),
            checkpointer=InMemorySaver(serde=state_serializer()),
        )
        task = rag_task(graph, settings.llm.pricing)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    metadata = {
        "variant": name,
        "overrides": overrides,
        "closed_book": closed_book,
        "collection": settings.qdrant.collection,
        "generation_model": settings.llm.generation_model,
        "cheap_model": settings.llm.cheap_model,
        "reranker": settings.reranker.model if settings.reranker.enabled else None,
        "top_k": settings.reranker.top_k,
        "retrieval_mode": settings.retrieval.mode,
        "examples": len(items),
        "categories": list(categories),
        "prompt_versions": {**PROMPT_VERSIONS, "judge": judge_prompt.VERSION},
    }
    report = run_experiment(
        client,
        dataset,
        task=task,
        evaluators=build_evaluators(llm, k=settings.reranker.top_k, closed_book=closed_book),
        name=f"{name}-{stamp}",
        metadata=metadata,
        description=f"{'closed-book baseline' if closed_book else 'RAG pipeline'} ({name})",
    )
    report.save(settings.evaluation.results_dir / f"{report.name}.json")
    return report


def latest_report(results_dir: Path, prefix: str) -> ExperimentReport | None:
    paths = sorted(results_dir.glob(f"{prefix}-*.json"))
    return ExperimentReport.load(paths[-1]) if paths else None


def leaky_questions(closed_book: ExperimentReport) -> list[str]:
    return sorted(
        example_id
        for example_id, row in closed_book.rows.items()
        if row["scores"].get("leaky", {}).get("label") == "leaky"
    )
