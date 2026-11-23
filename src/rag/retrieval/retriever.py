"""Retrieve -> fuse (RRF) -> rerank -> expand to parents within the context budget."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from rag import telemetry
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
    model_name: str

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

    def search(
        self,
        queries: Sequence[str],
        *,
        filters: SearchFilters | None = None,
        mode: SearchMode | None = None,
    ) -> list[RankedChunk]:
        """Retrieve for every query variant and fuse all rankings (RRF)."""
        if not queries:
            raise ValueError("at least one query is required")
        mode = mode or self._cfg.mode
        query_info = {"queries": list(queries), "mode": mode, "filters": _filters_json(filters)}
        with telemetry.span("hybrid_search", "RETRIEVER", {"input.value": query_info}) as current:
            rankings = self._searcher.search(
                queries, mode=mode, limit=self._cfg.candidates_per_ranking, filters=filters
            )
            fused = self._fuse(rankings)
            # Record the pool that can reach the reranker; the long tail adds noise, not insight.
            shown = fused[: max(self._rr.candidates or 0, self._rr.top_k)]
            telemetry.set_documents(
                current, "retrieval.documents", [_traced(c, c.fused_score) for c in shown]
            )
            current.set_attribute("rag.retrieve.candidates", len(fused))
            return fused

    def rerank_and_expand(
        self,
        query: str,
        candidates: Sequence[RankedChunk],
        *,
        rerank: bool = True,
        top_k: int | None = None,
        timings: dict[str, float] | None = None,
    ) -> tuple[list[RankedChunk], list[ContextPassage]]:
        """Rerank the top candidates against `query`, keep top-k, expand to parent passages within budget."""
        k = top_k or self._rr.top_k
        t = time.perf_counter()
        if rerank and self._reranker is not None and candidates:
            ranked = self._traced_rerank(query, list(candidates), k)
        else:
            ranked = list(candidates[:k])
        rerank_ms = _ms(t)
        t = time.perf_counter()
        with telemetry.span("build_context", "CHAIN") as current:
            parent_ids = list(dict.fromkeys(str(c.payload["parent_id"]) for c in ranked))
            passages = build_context(
                ranked,
                self._parents.get_parents(parent_ids),
                self._count_tokens,
                self._cfg.context_token_budget,
            )
            current.set_attribute("rag.context.passages", len(passages))
            current.set_attribute("rag.context.budget", self._cfg.context_token_budget)
            current.set_attribute("rag.context.tokens", sum(self._count_tokens(p.text) for p in passages))
        if timings is not None:
            timings["rerank"] = rerank_ms
            timings["context"] = _ms(t)
        return ranked, passages

    def retrieve(
        self,
        queries: Sequence[str],
        *,
        filters: SearchFilters | None = None,
        mode: SearchMode | None = None,
        rerank: bool = True,
        top_k: int | None = None,
    ) -> RetrievalResult:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        candidates = self.search(queries, filters=filters, mode=mode)
        timings["search"] = _ms(start)
        ranked, passages = self.rerank_and_expand(
            queries[0], candidates, rerank=rerank, top_k=top_k, timings=timings
        )
        timings["total"] = _ms(start)
        return RetrievalResult(list(queries), candidates, ranked, passages, timings)

    def _fuse(self, rankings: list[list[ScoredChunk]]) -> list[RankedChunk]:
        with telemetry.span("rrf_fusion", "CHAIN") as current:
            payloads = {hit.chunk_id: hit.payload for ranking in rankings for hit in ranking}
            fused = rrf_fuse([[hit.chunk_id for hit in ranking] for ranking in rankings], k=self._cfg.rrf_k)
            current.set_attribute("rag.rrf.k", self._cfg.rrf_k)
            current.set_attribute("rag.rrf.rankings", len(rankings))
            current.set_attribute("rag.rrf.fused", len(fused))
            return [RankedChunk(cid, score, score, payloads[cid]) for cid, score in fused]

    def _traced_rerank(self, query: str, candidates: list[RankedChunk], k: int) -> list[RankedChunk]:
        assert self._reranker is not None
        pool = candidates[: self._rr.candidates]
        attributes = {
            "reranker.query": query,
            "reranker.top_k": k,
            "reranker.model_name": self._reranker.model_name,
        }
        with telemetry.span("rerank_passages", "RERANKER", attributes) as current:
            telemetry.set_documents(
                current, "reranker.input_documents", [_traced(c, c.fused_score) for c in pool]
            )
            ranked = self._rerank(query, candidates)[:k]
            telemetry.set_documents(
                current, "reranker.output_documents", [_traced(c, c.score) for c in ranked]
            )
            return ranked

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


_TRACED_METADATA = (
    "file_name",
    "file_path",
    "title",
    "section_path",
    "page_start",
    "page_end",
    "doc_date",
    "parent_id",
)


def _traced(chunk: RankedChunk, score: float) -> telemetry.TracedDocument:
    metadata = {key: chunk.payload[key] for key in _TRACED_METADATA if chunk.payload.get(key) is not None}
    return telemetry.TracedDocument(chunk.chunk_id, score, str(chunk.payload.get("text", "")), metadata)


def _filters_json(filters: SearchFilters | None) -> dict[str, object] | None:
    if filters is None or filters.is_empty():
        return None
    return {
        "file_names": list(filters.file_names),
        "folder": str(filters.folder) if filters.folder else None,
        "date_after": filters.date_after.isoformat() if filters.date_after else None,
        "date_before": filters.date_before.isoformat() if filters.date_before else None,
    }


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
