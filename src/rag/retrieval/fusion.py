"""Reciprocal Rank Fusion across any number of rankings (dense/sparse x query variants)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def rrf_fuse[T: Hashable](rankings: Sequence[Sequence[T]], *, k: int = 60) -> list[tuple[T, float]]:
    """score(d) = sum over rankings of 1 / (k + rank), rank starting at 1.

    A document repeated within one ranking counts once, at its best rank. Ties keep first-appearance
    order so results are deterministic.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[T, float] = {}
    for ranking in rankings:
        seen: set[T] = set()
        for rank, doc in enumerate(ranking, start=1):
            if doc in seen:
                continue
            seen.add(doc)
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    # dict preserves first-appearance order and sorted() is stable, which gives the tie-break.
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
