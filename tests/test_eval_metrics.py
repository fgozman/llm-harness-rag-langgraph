import math

import pytest

from rag.evaluation.metrics import abstention_score, hit_at_k, ndcg_at_k, ranked_unique, reciprocal_rank

RETRIEVED = ["x.pdf", "a.pdf", "x.pdf", "b.pdf", "c.pdf"]


def test_ranked_unique_keeps_first_occurrence_order() -> None:
    assert ranked_unique(RETRIEVED) == ["x.pdf", "a.pdf", "b.pdf", "c.pdf"]


@pytest.mark.parametrize(
    ("gold", "k", "expected"),
    [
        ({"a.pdf"}, 2, 1.0),
        ({"b.pdf"}, 2, 0.0),
        ({"b.pdf"}, 3, 1.0),
        ({"z.pdf"}, 10, 0.0),
        ({"z.pdf", "c.pdf"}, 4, 1.0),
    ],
)
def test_hit_at_k_counts_documents_not_chunks(gold: set[str], k: int, expected: float) -> None:
    assert hit_at_k(RETRIEVED, gold, k) == expected


def test_reciprocal_rank_of_first_gold_document() -> None:
    assert reciprocal_rank(RETRIEVED, {"b.pdf", "c.pdf"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RETRIEVED, {"x.pdf"}) == 1.0
    assert reciprocal_rank(RETRIEVED, {"z.pdf"}) == 0.0


def test_ndcg_binary_relevance_with_ideal_ordering() -> None:
    # docs: x a b c ; gold {a, c} -> gains at ranks 2 and 4
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RETRIEVED, {"a.pdf", "c.pdf"}, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a.pdf", "c.pdf"], {"a.pdf", "c.pdf"}, 10) == pytest.approx(1.0)
    assert ndcg_at_k(RETRIEVED, {"z.pdf"}, 10) == 0.0
    # Only the top k count; ideal DCG is limited to k as well.
    assert ndcg_at_k(RETRIEVED, {"c.pdf"}, 3) == 0.0


def test_metrics_need_a_gold_set() -> None:
    with pytest.raises(ValueError):
        hit_at_k(RETRIEVED, set(), 3)


@pytest.mark.parametrize(
    ("answerable", "abstained", "expected"),
    [(False, True, 1.0), (False, False, 0.0), (True, False, 1.0), (True, True, 0.0)],
)
def test_abstention_score(answerable: bool, abstained: bool, expected: float) -> None:
    assert abstention_score(answerable=answerable, abstained=abstained) == expected
