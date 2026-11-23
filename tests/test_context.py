from rag.ingestion.models import ParentSection
from rag.retrieval.context import build_context
from rag.retrieval.types import RankedChunk


def words(text: str) -> int:
    return len(text.split())


def chunk(
    cid: str, parent: str, text: str, score: float, index: int = 0, page: int | None = 1
) -> RankedChunk:
    return RankedChunk(
        chunk_id=cid,
        score=score,
        fused_score=score,
        payload={
            "parent_id": parent,
            "text": text,
            "chunk_index": index,
            "file_path": f"/docs/{parent}.pdf",
            "file_name": f"{parent}.pdf",
            "title": f"Title {parent}",
            "section_path": ["Sec", parent],
            "page_start": page,
            "page_end": page,
        },
    )


def parent(pid: str, text: str, pages: tuple[int, int] = (1, 3)) -> ParentSection:
    return ParentSection(
        parent_id=pid, text=text, section_path=("Sec", pid), page_start=pages[0], page_end=pages[1]
    )


def test_children_expand_to_their_parent_in_rank_order() -> None:
    parents = {"P1": parent("P1", "parent one full text"), "P2": parent("P2", "parent two full text")}
    passages = build_context(
        [chunk("c2", "P2", "two", 0.9), chunk("c1", "P1", "one", 0.8)], parents, words, 100
    )
    assert [p.text for p in passages] == ["parent two full text", "parent one full text"]
    assert [p.index for p in passages] == [1, 2]
    assert passages[0].expanded and passages[0].page_start == 1 and passages[0].page_end == 3
    # The snippet quotes the retrieved child chunk, not the start of the (much larger) parent.
    assert passages[0].snippet == "two"


def test_siblings_dedupe_into_one_passage_keeping_best_position() -> None:
    parents = {"P1": parent("P1", "big parent"), "P2": parent("P2", "other parent")}
    ranked = [chunk("a", "P1", "a", 0.9, 0), chunk("b", "P2", "b", 0.8), chunk("c", "P1", "c", 0.7, 1)]
    passages = build_context(ranked, parents, words, 100)
    assert [p.parent_id for p in passages] == ["P1", "P2"]
    assert passages[0].chunk_ids == ("a", "c")
    assert passages[0].score == 0.9


def test_parent_too_big_falls_back_to_its_children() -> None:
    parents = {"P1": parent("P1", " ".join(["w"] * 50))}
    ranked = [chunk("b", "P1", "second child", 0.9, index=1), chunk("a", "P1", "first child", 0.5, index=0)]
    (passage,) = build_context(ranked, parents, words, 10)
    assert passage.text == "first child\n\nsecond child"  # document order, not rank order
    assert not passage.expanded


def test_oversized_group_falls_back_to_best_child_then_skips() -> None:
    parents = {"P1": parent("P1", " ".join(["w"] * 50)), "P2": parent("P2", "small parent")}
    ranked = [
        chunk("a", "P1", "best child here", 0.9, 0),
        chunk("b", "P1", " ".join(["x"] * 20), 0.8, 1),
        chunk("c", "P2", "c", 0.7),
    ]
    passages = build_context(ranked, parents, words, 6)
    assert [p.text for p in passages] == ["best child here", "small parent"]
    assert passages[0].chunk_ids == ("a",)


def test_budget_is_respected_and_smaller_later_passages_still_fit() -> None:
    parents = {
        "P1": parent("P1", " ".join(["a"] * 8)),
        "P2": parent("P2", " ".join(["b"] * 8)),
        "P3": parent("P3", "tiny"),
    }
    ranked = [
        chunk("1", "P1", "a a", 0.9),
        chunk("2", "P2", " ".join(["b"] * 8), 0.8),
        chunk("3", "P3", "t", 0.7),
    ]
    passages = build_context(ranked, parents, words, 10)
    assert [p.parent_id for p in passages] == ["P1", "P3"]
    assert sum(words(p.text) for p in passages) <= 10


def test_missing_parent_uses_child_text_with_child_pages() -> None:
    (passage,) = build_context([chunk("a", "gone", "orphan text", 0.9, page=7)], {}, words, 100)
    assert passage.text == "orphan text"
    assert (passage.page_start, passage.page_end) == (7, 7)
    assert passage.file_name == "gone.pdf" and passage.section_path == ("Sec", "gone")


def test_empty_input() -> None:
    assert build_context([], {}, words, 100) == []
