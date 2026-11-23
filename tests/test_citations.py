from rag.graph.citations import (
    NOT_FOUND,
    build_citations,
    cautious_answer,
    citation_numbers,
    strip_invalid_citations,
)
from rag.graph.state import Passage, SupportedClaim


def passage(
    index: int, text: str = "Passage text that is long enough to be cut into a snippet.", snippet: str = ""
) -> Passage:
    return Passage(
        index=index,
        parent_id=f"P{index}",
        chunk_ids=[f"c{index}"],
        text=text,
        score=0.9,
        file_path=f"/docs/f{index}.pdf",
        file_name=f"f{index}.pdf",
        title=f"Doc {index}",
        section_path=["Results", "Q3"],
        page_start=index,
        page_end=index + 1,
        snippet=snippet or text,
    )


def test_numbers_in_order_of_first_use_with_grouped_markers() -> None:
    assert citation_numbers("A [2]. B [1][3]. C [2, 4] and [5,6]. D [1].") == [2, 1, 3, 4, 5, 6]


def test_non_citation_brackets_are_ignored() -> None:
    assert citation_numbers("Array a[0] and [note] and [12a] and [ 7 ]") == [7]


def test_citations_map_to_passage_sources() -> None:
    passages = [passage(1), passage(2)]
    citations = build_citations("Revenue rose [2] while costs fell [1].", passages)
    assert [c.index for c in citations] == [2, 1]
    first = citations[0]
    assert (first.file_path, first.file_name, first.title) == ("/docs/f2.pdf", "f2.pdf", "Doc 2")
    assert (first.page_start, first.page_end, first.section) == (2, 3, "Results > Q3")
    assert first.snippet.startswith("Passage text")


def test_snippets_are_short_and_whitespace_normalised() -> None:
    long = passage(1, "word   \n " * 200)
    (citation,) = build_citations("x [1]", [long])
    assert len(citation.snippet) <= 241 and "\n" not in citation.snippet and "  " not in citation.snippet


def test_invalid_markers_are_removed_and_not_cited() -> None:
    passages = [passage(1)]
    text = strip_invalid_citations("Claim [1]. Made up [7]. Mixed [1, 9].", passages)
    assert text == "Claim [1]. Made up. Mixed [1]."
    assert [c.index for c in build_citations(text, passages)] == [1]


def test_cautious_answer_lists_verified_and_unverified_claims() -> None:
    passages = [passage(1), passage(2)]
    text, citations = cautious_answer(
        supported=[SupportedClaim(claim="Revenue was $14.8B", citations=[2])],
        unsupported=["Margins doubled"],
        passages=passages,
    )
    assert "could not fully verify" in text.lower()
    assert "Revenue was $14.8B [2]" in text and "Margins doubled" in text
    assert [c.index for c in citations] == [2]


def test_cautious_answer_without_supported_claims_is_an_abstention() -> None:
    text, citations = cautious_answer(supported=[], unsupported=["x"], passages=[passage(1)])
    assert NOT_FOUND in text and citations == []


def test_citation_snippet_comes_from_the_retrieved_chunk() -> None:
    p = passage(1, text="## Long heading of the parent\n\nIntro...", snippet="Revenue was $14,766 million.")
    (citation,) = build_citations("x [1]", [p])
    assert citation.snippet == "Revenue was $14,766 million."
