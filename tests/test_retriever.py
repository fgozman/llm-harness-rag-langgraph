from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from qdrant_client import models

from rag.config import RerankerSettings, RetrievalSettings
from rag.ingestion.discovery import DiscoveredFile, doc_id_for
from rag.ingestion.manifest import Manifest
from rag.ingestion.models import ParentSection
from rag.retrieval.filters import SearchFilters, to_qdrant_filter
from rag.retrieval.retriever import Retriever, SearchMode
from rag.retrieval.types import ScoredChunk


def payload(cid: str) -> dict[str, object]:
    return {
        "parent_id": f"P-{cid}",
        "text": f"text of {cid}",
        "context": f"context {cid}",
        "title": "Doc",
        "section_path": ["S"],
        "chunk_index": 0,
        "file_path": f"/d/{cid}.md",
        "file_name": f"{cid}.md",
        "page_start": None,
        "page_end": None,
    }


def hits(*ids: str) -> list[ScoredChunk]:
    return [ScoredChunk(i, 1.0 / (n + 1), payload(i)) for n, i in enumerate(ids)]


class FakeSearcher:
    def __init__(self, rankings: dict[tuple[str, str], list[ScoredChunk]]) -> None:
        self.rankings = rankings
        self.calls: list[tuple[tuple[str, ...], SearchMode, int, SearchFilters | None]] = []

    def search(
        self, queries: Sequence[str], *, mode: SearchMode, limit: int, filters: SearchFilters | None
    ) -> list[list[ScoredChunk]]:
        self.calls.append((tuple(queries), mode, limit, filters))
        kinds = {"hybrid": ("dense", "sparse"), "dense": ("dense",), "sparse": ("sparse",)}[mode]
        return [self.rankings.get((q, k), []) for q in queries for k in kinds]


class FakeReranker:
    """Scores a passage 1.0 if it mentions the favoured chunk id, else by reverse input position."""

    def __init__(self, favour: str) -> None:
        self.favour = favour
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return [1.0 if self.favour in d else 0.5 - i * 0.01 for i, d in enumerate(documents)]


class FakeParents:
    def get_parents(self, parent_ids: Sequence[str]) -> dict[str, ParentSection]:
        return {p: ParentSection(p, f"parent {p}", ("S",), None, None) for p in parent_ids}


def retriever(
    searcher: FakeSearcher, reranker: FakeReranker | None, *, candidates: int = 10, top_k: int = 3
) -> Retriever:
    return Retriever(
        searcher=searcher,
        reranker=reranker,
        parents=FakeParents(),
        count_tokens=lambda s: len(s.split()),
        retrieval=RetrievalSettings(candidates_per_ranking=20),
        reranker_settings=RerankerSettings(candidates=candidates, top_k=top_k),
    )


RANKINGS = {("q", "dense"): hits("a", "b", "c", "d"), ("q", "sparse"): hits("c", "e", "a")}


def test_hybrid_fuses_both_rankings_then_reranks() -> None:
    searcher = FakeSearcher(RANKINGS)
    result = retriever(searcher, FakeReranker(favour="text of e")).retrieve(["q"])
    assert [c.chunk_id for c in result.candidates] == ["a", "c", "b", "e", "d"]
    assert [c.chunk_id for c in result.ranked] == ["e", "a", "c"]
    assert result.ranked[0].rerank_score == 1.0
    assert [p.parent_id for p in result.passages] == ["P-e", "P-a", "P-c"]
    assert searcher.calls == [(("q",), "hybrid", 20, None)]
    assert {"search", "rerank", "context", "total"} <= set(result.timings_ms)


def test_reranker_sees_title_section_context_and_text() -> None:
    reranker = FakeReranker(favour="zzz")
    retriever(FakeSearcher(RANKINGS), reranker).retrieve(["q"])
    _, docs = reranker.calls[0]
    assert docs[0] == "Doc > S\ncontext a\ntext of a"


def test_only_top_candidates_are_reranked() -> None:
    reranker = FakeReranker(favour="text of d")
    result = retriever(FakeSearcher(RANKINGS), reranker, candidates=2, top_k=5).retrieve(["q"])
    assert len(reranker.calls[0][1]) == 2
    assert [c.chunk_id for c in result.ranked] == ["a", "c"]


def test_without_reranker_fused_order_and_scores_are_used() -> None:
    result = retriever(FakeSearcher(RANKINGS), None).retrieve(["q"])
    assert [c.chunk_id for c in result.ranked] == ["a", "c", "b"]
    assert all(c.rerank_score is None and c.score == c.fused_score for c in result.ranked)


def test_rerank_flag_can_disable_reranking_per_call() -> None:
    reranker = FakeReranker(favour="text of e")
    result = retriever(FakeSearcher(RANKINGS), reranker).retrieve(["q"], rerank=False)
    assert reranker.calls == [] and result.ranked[0].chunk_id == "a"


def test_query_variants_are_fused_and_first_query_drives_reranking() -> None:
    rankings = {**RANKINGS, ("q2", "dense"): hits("f", "a"), ("q2", "sparse"): hits("f")}
    reranker = FakeReranker(favour="zzz")
    result = retriever(FakeSearcher(rankings), reranker).retrieve(["q", "q2"])
    assert result.candidates[0].chunk_id in {"a", "f"}
    assert {"f", "a"} <= {c.chunk_id for c in result.candidates[:3]}
    assert reranker.calls[0][0] == "q"


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_single_modes_and_filters_pass_through(mode: SearchMode) -> None:
    searcher = FakeSearcher(RANKINGS)
    filters = SearchFilters(file_names=("a.md",))
    retriever(searcher, None).retrieve(["q"], mode=mode, filters=filters, top_k=1)
    assert searcher.calls[0][1] == mode and searcher.calls[0][3] == filters


def test_no_hits_yields_empty_result() -> None:
    result = retriever(FakeSearcher({}), FakeReranker("x")).retrieve(["q"])
    assert result.candidates == [] and result.ranked == [] and result.passages == []


def test_requires_at_least_one_query() -> None:
    with pytest.raises(ValueError):
        retriever(FakeSearcher({}), None).retrieve([])


# --- filters ---------------------------------------------------------------------------------------


def test_filters_translate_to_qdrant_conditions() -> None:
    after = datetime(2026, 7, 1, tzinfo=UTC)
    flt = to_qdrant_filter(
        SearchFilters(file_names=("a.pdf", "b.pdf"), folder=Path("/docs/sec"), date_after=after),
        folders=["/docs/sec", "/docs/sec/q3"],
    )
    assert flt is not None and flt.must is not None
    conditions = {c.key: c for c in flt.must if isinstance(c, models.FieldCondition)}
    assert conditions["file_name"].match == models.MatchAny(any=["a.pdf", "b.pdf"])
    assert conditions["folder"].match == models.MatchAny(any=["/docs/sec", "/docs/sec/q3"])
    # Filters on the document date (name > content > mtime), not the raw modification time.
    assert conditions["doc_date_ts"].range == models.Range(gte=after.timestamp())


def test_empty_filters_mean_no_filter() -> None:
    assert to_qdrant_filter(None, folders=[]) is None
    assert to_qdrant_filter(SearchFilters(), folders=[]) is None


def test_unknown_folder_matches_nothing_rather_than_everything() -> None:
    flt = to_qdrant_filter(SearchFilters(folder=Path("/nowhere")), folders=[])
    assert flt is not None and flt.must is not None
    (cond,) = flt.must
    assert isinstance(cond, models.FieldCondition) and cond.match == models.MatchAny(any=["/nowhere"])


def test_manifest_lists_indexed_folders_under_a_prefix(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "rag.db")
    for rel in ("docs/a.md", "docs/sec/b.md", "docs/sec/q3/c.md", "other/d.md", "docs2/e.md"):
        path = (tmp_path / rel).resolve()
        manifest.record_indexed(
            DiscoveredFile(path, doc_id_for(path), "h", 1.0, 1), parser="t", parents=[], chunk_ids=[]
        )
    root = (tmp_path / "docs").resolve()
    assert manifest.folders_under(root / "sec") == [str(root / "sec"), str(root / "sec" / "q3")]
    assert manifest.folders_under(root) == [str(root), str(root / "sec"), str(root / "sec" / "q3")]


def test_search_then_rerank_and_expand_equals_retrieve() -> None:
    r = retriever(FakeSearcher(RANKINGS), FakeReranker(favour="text of e"))
    combined = r.retrieve(["q"])
    candidates = r.search(["q"])
    ranked, passages = r.rerank_and_expand("q", candidates)
    assert [c.chunk_id for c in candidates] == [c.chunk_id for c in combined.candidates]
    assert [c.chunk_id for c in ranked] == [c.chunk_id for c in combined.ranked]
    assert passages == combined.passages
