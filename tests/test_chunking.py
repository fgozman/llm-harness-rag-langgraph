import itertools
import re

import pytest

from rag.config import ChunkingSettings
from rag.ingestion.chunking import chunk_document
from rag.ingestion.models import Block, BlockKind, Chunk, ParentSection, ParsedDocument

DOC_ID = "00000000-0000-0000-0000-000000000001"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.findall(r"[^.]+\.", text.replace("\n", " "))]


def count_words(text: str) -> int:
    return len(text.split())


def cfg(**overrides: float) -> ChunkingSettings:
    base: dict[str, float] = {
        "child_target_tokens": 60,
        "child_max_tokens": 80,
        "overlap_ratio": 0.15,
        "parent_max_tokens": 200,
    }
    base.update(overrides)
    return ChunkingSettings.model_validate(base)


def heading(text: str, level: int, page: int | None = None) -> Block:
    return Block(BlockKind.HEADING, text, level=level, page=page)


def para(text: str, page: int | None = None) -> Block:
    return Block(BlockKind.TEXT, text, page=page)


def sentences(n: int, prefix: str = "S") -> str:
    return " ".join(f"{prefix}{i} alpha beta gamma delta." for i in range(n))


def table(rows: int) -> Block:
    lines = ["| Code | Name | Qty |", "|------|------|-----|"]
    lines += [f"| C{i} | item number {i} | {i} |" for i in range(rows)]
    return Block(BlockKind.TABLE, "\n".join(lines))


def run(
    blocks: list[Block], settings: ChunkingSettings | None = None
) -> tuple[list[ParentSection], list[Chunk]]:
    doc = ParsedDocument(title="Doc", blocks=blocks, parser="test")
    return chunk_document(doc, DOC_ID, count_words, settings or cfg())


def test_empty_document_has_no_chunks() -> None:
    assert run([]) == ([], [])


def test_section_paths_follow_heading_hierarchy() -> None:
    _, chunks = run(
        [
            heading("Doc", 0),
            heading("Intro", 1),
            para("intro text here"),
            heading("Details", 1),
            heading("Sub", 2),
            para("sub text here"),
            heading("Other", 1),
            para("other text here"),
        ]
    )
    paths = {c.text: c.section_path for c in chunks}
    # The document title is not repeated in section paths; it is added via the document title.
    assert paths == {
        "intro text here": ("Intro",),
        "sub text here": ("Details", "Sub"),
        "other text here": ("Other",),
    }


def test_chunks_never_span_sections() -> None:
    _, chunks = run([heading("A", 1), para("alpha one"), heading("B", 1), para("beta two")])
    assert [c.text for c in chunks] == ["alpha one", "beta two"]


def test_every_chunk_respects_max_budget() -> None:
    settings = cfg()
    _, chunks = run([heading("A", 1), para(sentences(40)), para(sentences(3, "T")), para(sentences(25, "U"))])
    assert len(chunks) > 3
    for c in chunks:
        assert count_words(c.text) <= settings.child_max_tokens
        assert c.token_count == count_words(c.text)


def test_overlong_sentence_is_split_by_words() -> None:
    settings = cfg()
    _, chunks = run([para(" ".join(f"w{i}" for i in range(300)))])
    assert all(count_words(c.text) <= settings.child_max_tokens for c in chunks)
    assert "w0" in chunks[0].text
    assert "w299" in chunks[-1].text


def test_consecutive_chunks_in_a_section_overlap() -> None:
    settings = cfg()
    _, chunks = run([para(sentences(40))])
    assert len(chunks) >= 3
    overlap_budget = int(settings.overlap_ratio * settings.child_target_tokens)
    for prev, nxt in itertools.pairwise(chunks):
        prev_s, next_s = split_sentences(prev.text), split_sentences(nxt.text)
        # The next chunk opens with sentences carried over from the end of the previous chunk.
        shared = max(k for k in range(len(next_s) + 1) if k == 0 or prev_s[-k:] == next_s[:k])
        assert shared >= 1
        assert count_words(" ".join(next_s[:shared])) <= overlap_budget


def test_no_overlap_across_sections() -> None:
    _, chunks = run([heading("A", 1), para(sentences(3)), heading("B", 1), para(sentences(3, "T"))])
    assert chunks[1].text.startswith("T0 ")


def test_small_table_is_one_intact_chunk() -> None:
    _, chunks = run([heading("A", 1), table(3)])
    assert len(chunks) == 1
    assert chunks[0].text == table(3).text


def test_large_table_splits_on_row_boundaries_with_header_repeated() -> None:
    settings = cfg()
    big = table(40)
    _, chunks = run([heading("A", 1), big])
    assert len(chunks) > 1
    source_rows = set(big.text.splitlines())
    seen_rows: list[str] = []
    for c in chunks:
        lines = c.text.splitlines()
        assert lines[:2] == big.text.splitlines()[:2], "each piece repeats the header"
        assert all(line in source_rows for line in lines), "no row is cut"
        assert count_words(c.text) <= settings.child_max_tokens
        seen_rows += lines[2:]
    assert seen_rows == big.text.splitlines()[2:], "every row appears exactly once, in order"


def test_single_oversized_row_is_kept_whole() -> None:
    row = "| " + " ".join(f"x{i}" for i in range(120)) + " |"
    blk = Block(BlockKind.TABLE, "| H |\n|---|\n" + row)
    _, chunks = run([blk])
    assert any(row in c.text.splitlines() for c in chunks)


def test_table_and_text_do_not_share_overlap() -> None:
    _, chunks = run([para(sentences(10)), table(30)])
    table_chunks = [c for c in chunks if c.text.startswith("| Code")]
    assert table_chunks, "table pieces start with their header, not with overlap text"


def test_every_chunk_links_to_a_parent_containing_it() -> None:
    parents, chunks = run(
        [heading("A", 1), para(sentences(30)), heading("B", 1), table(20), heading("C", 1), para("tail")]
    )
    by_id = {p.parent_id: p for p in parents}
    for c in chunks:
        assert c.parent_id in by_id
        assert c.text in by_id[c.parent_id].text


def test_parents_respect_budget_and_pack_small_sibling_sections() -> None:
    settings = cfg()
    parents, _ = run(
        [
            heading("Top", 1),
            heading("a", 2),
            para("one two"),
            heading("b", 2),
            para("three four"),
            heading("Next", 1),
            para(sentences(80)),
        ]
    )
    small = [p for p in parents if "one two" in p.text]
    assert len(small) == 1 and "three four" in small[0].text, "siblings under one top heading share a parent"
    assert all("one two" not in p.text for p in parents if "S0 " in p.text), (
        "different top sections don't mix"
    )
    for p in parents:
        assert count_words(p.text) <= settings.parent_max_tokens + 10  # + heading lines


def test_parent_text_includes_headings() -> None:
    parents, _ = run([heading("Install", 1), para("mount the arm")])
    assert "Install" in parents[0].text
    assert parents[0].section_path == ("Install",)


def test_page_ranges_propagate() -> None:
    parents, chunks = run([heading("A", 1, page=1), para("on page one", page=1), para("on page two", page=2)])
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)
    assert (parents[0].page_start, parents[0].page_end) == (1, 2)


def test_ids_are_deterministic_and_unique() -> None:
    blocks = [heading("A", 1), para(sentences(30)), heading("B", 1), para("x y z")]
    p1, c1 = run(blocks)
    p2, c2 = run(blocks)
    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]
    assert [p.parent_id for p in p1] == [p.parent_id for p in p2]
    assert len({c.chunk_id for c in c1}) == len(c1)
    assert [c.chunk_index for c in c1] == list(range(len(c1)))


@pytest.mark.parametrize("kind", [BlockKind.LIST_ITEM, BlockKind.CODE])
def test_other_block_kinds_are_chunked(kind: BlockKind) -> None:
    _, chunks = run([Block(kind, "some content")])
    assert chunks[0].text == "some content"


def test_heading_equal_to_title_is_not_in_paths_at_any_level() -> None:
    _, chunks = run([heading("Doc", 1), heading("Sub", 2), para("body text")])
    assert chunks[0].section_path == ("Sub",)
