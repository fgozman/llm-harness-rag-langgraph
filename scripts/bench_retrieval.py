"""Latency benchmark for retrieve + rerank on an indexed collection (spec: record p50/p95 on the Mac profile).

QDRANT__COLLECTION=test_corpus STATE_DIR=.rag/test_corpus uv run python scripts/bench_retrieval.py
"""

from __future__ import annotations

import statistics
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rag.config import Settings
from rag.factory import build_retriever

QUERIES = [
    "What was Uber's revenue for the quarter ended September 30, 2026?",
    "How did Airbnb's nights and experiences booked change in Q3 2026?",
    "What guidance did AMD give for the fourth quarter of 2026?",
    "Palantir US commercial revenue growth in the second quarter of 2026",
    "Cloudflare dollar-based net retention rate",
    "Lyft gross bookings third quarter 2026",
    "Datadog number of customers with ARR over $100,000",
    "Booking Holdings room nights growth",
    "What share of Commerce office locations were underutilized?",
    "How many service members used tuition assistance according to GAO?",
    "ICE detention costs per bed per day",
    "AI-powered malware examples from 2026",
    "Which LangGraph release added a new checkpointer feature?",
    "Phoenix 22.0 breaking changes",
    "qdrant-client 1.20 release notes",
    "beam search approximation for DiskANN",
    "generative recommendation with semantic IDs",
    "Federal Register rule on National Instant Criminal Background Check System criteria",
    "effective date of the horse protection rule postponement",
    "Vellmark warranty claims for the VR-9 actuator",
]


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def main(
    candidates: Annotated[list[int], typer.Option(help="Rerank candidate counts to compare.")] = [20, 25, 30],  # noqa: B006
    repeats: Annotated[int, typer.Option(help="Passes over the query set.")] = 2,
) -> None:
    console = Console()
    settings = Settings()
    retriever = build_retriever(settings)
    retriever.retrieve(QUERIES[:2])  # warm-up: model load, MPS kernels, Qdrant caches
    table = Table(
        "candidates",
        "search p50",
        "rerank p50",
        "rerank p95",
        "total p50",
        "total p95",
        title=settings.qdrant.collection,
    )
    for n in candidates:
        settings.reranker.candidates = n
        search: list[float] = []
        rerank: list[float] = []
        total: list[float] = []
        for _ in range(repeats):
            for q in QUERIES:
                t = retriever.retrieve([q]).timings_ms
                console.print(f"[dim]n={n} {t['total']:.0f} ms (rerank {t['rerank']:.0f}) {q[:50]}[/dim]")
                search.append(t["search"])
                rerank.append(t["rerank"])
                total.append(t["total"])
        table.add_row(
            str(n),
            f"{statistics.median(search):.0f} ms",
            f"{statistics.median(rerank):.0f} ms",
            f"{_pct(rerank, 0.95):.0f} ms",
            f"{statistics.median(total):.0f} ms",
            f"{_pct(total, 0.95):.0f} ms",
        )
    console.print(table)
    console.print(
        f"[dim]reranker={settings.reranker.model}, device={settings.device}, "
        f"{len(QUERIES)} queries x {repeats}[/dim]"
    )


if __name__ == "__main__":
    typer.run(main)
