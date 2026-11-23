"""Retrieve -> fuse (RRF) -> rerank -> expand to parents within the context budget."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from rag.config import RerankerSettings, RetrievalSettings
from rag.ingestion.models import ParentSection
from rag.retrieval.context import build_context
from rag.retrieval.filters import SearchFilters
from rag.retrieval.fusion import rrf_fuse
from rag.retrieval.types import ContextPassage, RankedChunk, ScoredChunk, payload_section

SearchMode = Literal["hybrid", "dense", "sparse"]


class ChunkSearcher(Protocol):
    def search(
        self, queries: Sequence[str], *, mode: SearchMode, limit: int, filters: SearchFilters | None
    ) -> list[list[ScoredChunk]]:
        """One ranking per (query, vector type)."""
        ...


class Reranker(Protocol):
    def rerank(self, query: str, documents: Sequence[str]) -> list[float]: ...


class ParentStore(Protocol):
    def get_parents(self, parent_ids: Sequence[str]) -> dict[str, ParentSection]: ...


@dataclass(slots=True)
class RetrievalResult:
    queries: list[str]
    candidates: list[RankedChunk]  # fused order, before reranking
    ranked: list[RankedChunk]  # final top-k
    passages: list[ContextPassage]
    timings_ms: dict[str, float] = field(default_factory=dict[str, float])


def rerank_text(payload: dict[str, object]) -> str:
    """What the reranker judges: location + contextual blurb + chunk text (mirrors what was embedded)."""
    location = " > ".join([str(payload.get("title", "")), *payload_section(payload)])
    parts = [location, payload.get("context"), payload.get("text")]
    return "\n".join(str(p) for p in parts if p)


class Retriever:
    def __init__(
        self,
        *,
        searcher: ChunkSearcher,
        reranker: Reranker | None,
        parents: ParentStore,
        count_tokens: Callable[[str], int],
        retrieval: RetrievalSettings,
        reranker_settings: RerankerSettings,
    ) -> None:
        self._searcher = searcher
        self._reranker = reranker
        self._parents = parents
        self._count_tokens = count_tokens
        self._cfg = retrieval
        self._rr = reranker_settings

    def retrieve(
        self,
        queries: Sequence[str],
        *,
        filters: SearchFilters | None = None,
        mode: SearchMode | None = None,
        rerank: bool = True,
        top_k: int | None = None,
    ) -> RetrievalResult:
        if not queries:
            raise ValueError("at least one query is required")
        timings: dict[str, float] = {}
        start = time.perf_counter()
        rankings = self._searcher.search(
            queries, mode=mode or self._cfg.mode, limit=self._cfg.candidates_per_ranking, filters=filters
        )
        timings["search"] = _ms(start)

        candidates = self._fuse(rankings)
        k = top_k or self._rr.top_k
        t = time.perf_counter()
        if rerank and self._reranker is not None and candidates:
            ranked = self._rerank(queries[0], candidates)[:k]
        else:
            ranked = candidates[:k]
        timings["rerank"] = _ms(t)

        t = time.perf_counter()
        parent_ids = list(dict.fromkeys(str(c.payload["parent_id"]) for c in ranked))
        passages = build_context(
            ranked, self._parents.get_parents(parent_ids), self._count_tokens, self._cfg.context_token_budget
        )
        timings["context"] = _ms(t)
        timings["total"] = _ms(start)
        return RetrievalResult(list(queries), candidates, ranked, passages, timings)

    def _fuse(self, rankings: list[list[ScoredChunk]]) -> list[RankedChunk]:
        payloads = {hit.chunk_id: hit.payload for ranking in rankings for hit in ranking}
        fused = rrf_fuse([[hit.chunk_id for hit in ranking] for ranking in rankings], k=self._cfg.rrf_k)
        return [RankedChunk(cid, score, score, payloads[cid]) for cid, score in fused]

    def _rerank(self, query: str, candidates: list[RankedChunk]) -> list[RankedChunk]:
        assert self._reranker is not None
        pool = candidates[: self._rr.candidates]
        scores = self._reranker.rerank(query, [rerank_text(c.payload) for c in pool])
        rescored = [
            RankedChunk(c.chunk_id, s, c.fused_score, c.payload, rerank_score=s)
            for c, s in zip(pool, scores, strict=True)
        ]
        # Stable sort: equal rerank scores keep fused order.
        return sorted(rescored, key=lambda c: c.score, reverse=True)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
