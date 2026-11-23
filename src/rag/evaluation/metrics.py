"""Deterministic metrics: document-level retrieval quality and abstention correctness."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def ranked_unique(items: Sequence[str]) -> list[str]:
    """Collapse chunk-level results to documents, keeping each document's best rank."""
    return list(dict.fromkeys(items))


def _check(gold: Collection[str]) -> None:
    if not gold:
        raise ValueError("retrieval metrics need at least one gold document")


def hit_at_k(retrieved: Sequence[str], gold: Collection[str], k: int) -> float:
    _check(gold)
    return 1.0 if any(doc in gold for doc in ranked_unique(retrieved)[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], gold: Collection[str]) -> float:
    _check(gold)
    for rank, doc in enumerate(ranked_unique(retrieved), start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Collection[str], k: int) -> float:
    """Binary-relevance NDCG over the first k distinct documents."""
    _check(gold)
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, doc in enumerate(ranked_unique(retrieved)[:k], 1) if doc in gold
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal


def abstention_score(*, answerable: bool, abstained: bool) -> float:
    """1 when the system abstains exactly on unanswerable questions (false abstentions also score 0)."""
    return 1.0 if abstained != answerable else 0.0
