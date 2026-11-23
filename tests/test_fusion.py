import pytest

from rag.retrieval.fusion import rrf_fuse


def test_rrf_scores_are_sums_of_reciprocal_ranks() -> None:
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)


def test_items_in_both_lists_beat_single_list_top_hits() -> None:
    order = [doc for doc, _ in rrf_fuse([["x", "shared"], ["y", "shared"]], k=60)]
    assert order[0] == "shared"


def test_ties_break_by_first_appearance_for_determinism() -> None:
    order = [doc for doc, _ in rrf_fuse([["a", "b"], ["b", "a"]], k=60)]
    assert order == ["a", "b"]
    order = [doc for doc, _ in rrf_fuse([["b", "a"], ["a", "b"]], k=60)]
    assert order == ["b", "a"]


def test_duplicates_within_a_ranking_count_once_at_best_rank() -> None:
    fused = dict(rrf_fuse([["a", "b", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)


def test_empty_inputs() -> None:
    assert rrf_fuse([], k=60) == []
    assert rrf_fuse([[], []], k=60) == []


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], k=0)
