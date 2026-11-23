import json
import random
from collections.abc import Sequence
from pathlib import Path

from rag.evaluation.golden import (
    AnswerabilityVerdict,
    ChunkRecord,
    GeneratedQA,
    GoldenGenerator,
    GoldenItem,
    QAVerdict,
    UnanswerableBatch,
    UnanswerableQuestion,
    best_match,
    canary_items,
    is_table,
    load_golden,
    plan_candidates,
    quote_in_text,
    save_golden,
    sec_info,
)
from test_graph import FakeLLM  # pyright: ignore[reportImplicitRelativeImport]


def rec(cid: str, file_name: str, text: str, tokens: int = 200, section: str = "S") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=cid,
        file_name=file_name,
        file_path=f"/corpus/{file_name}",
        title=file_name,
        section=section,
        page_start=None,
        page_end=None,
        text=text,
        token_count=tokens,
    )


TABLE = (
    "| Metric | Three Months Ended September 30, 2026 |\n| --- | --- |\n"
    "| Revenue | $14,766 |\n| Net income | $512 |"
)


def test_quote_matching_tolerates_whitespace_and_table_markup() -> None:
    assert quote_in_text("Revenue | $14,766", TABLE)
    assert quote_in_text("revenue   $14,766", TABLE)
    assert quote_in_text("The  NRX gripper\nships", "Intro. The NRX gripper ships in October.")
    assert not quote_in_text("Revenue | $15,000", TABLE)
    assert not quote_in_text("", TABLE)


def test_sec_file_names_parse_company_form_and_filing_month() -> None:
    assert sec_info("UBER_10-Q_period-2026-09-30_filed-2026-11-04.htm") == ("UBER", "10-Q", "2026-11")
    assert sec_info("AMD_8-K_filed-2026-08-04_ex99-1.htm") == ("AMD", "8-K", "2026-08")
    assert sec_info("2608.01147.pdf") is None


def test_table_detection_and_best_match() -> None:
    assert is_table(TABLE)
    assert not is_table("Revenue grew 10% year over year.")
    other = TABLE.replace("September 30, 2026", "June 30, 2026").replace("14,766", "12,651")
    candidates = [rec("x", "a", "Unrelated text about risk factors and litigation."), rec("y", "b", other)]
    match = best_match(rec("t", "t", TABLE), candidates, min_similarity=0.3)
    assert match is not None and match.chunk_id == "y"
    assert best_match(rec("t", "t", TABLE), candidates[:1], min_similarity=0.3) is None


def test_plan_pairs_temporal_and_comparison_chunks_within_one_company() -> None:
    q2 = TABLE.replace("September 30", "June 30").replace("14,766", "12,651")
    q1 = TABLE.replace("September 30", "March 31").replace("14,766", "11,533")
    pool = [
        rec("nov", "UBER_10-Q_period-2026-09-30_filed-2026-11-04.htm", TABLE),
        rec("aug", "UBER_10-Q_period-2026-06-30_filed-2026-08-05.htm", q2),
        rec("may", "UBER_10-Q_period-2026-03-31_filed-2026-05-06.htm", q1),
        rec(
            "amd", "AMD_10-Q_period-2026-09-26_filed-2026-11-04.htm", TABLE.replace("Revenue", "Net revenue")
        ),
        rec(
            "rn", "release_notes/x.md", "## 1.2.0 (2026-09-01)\n\nAdded streaming checkpoints and a new API."
        ),
        rec(
            "paper",
            "2608.00001.pdf",
            "| Model | NDCG@10 |\n| --- | --- |\n| Ours | 0.512 |\n| BM25 | 0.401 |\n| Dense | 0.455 |",
        ),
    ]
    plan = plan_candidates(pool, random.Random(0))
    for pair in plan["temporal"] + plan["comparison"]:
        companies = {info[0] for c in pair if (info := sec_info(c.file_name)) is not None}
        assert len(pair) == 2 and len(companies) == 1
    assert ("nov", "aug") in {tuple(c.chunk_id for c in p) for p in plan["comparison"]}
    assert [c[0].chunk_id for c in plan["table_lookup"]] == ["paper"]
    assert [c[0].chunk_id for c in plan["release_notes"]] == ["rn"]
    assert plan == plan_candidates(pool, random.Random(0)), "deterministic for a seed"


# --- generation loop ---------------------------------------------------------------------------------


def qa(question: str, quote: str, usable: bool = True, sources: Sequence[int] = (1,)) -> GeneratedQA:
    return GeneratedQA(
        question=question,
        answer="$14,766 million",
        supporting_quote=quote,
        source_passages=list(sources),
        usable=usable,
    )


def verdict(ok: bool = True) -> QAVerdict:
    return QAVerdict(self_contained=ok, supported=True, unambiguous=True, note="")


def test_items_need_a_verbatim_quote_and_a_passing_judge() -> None:
    llm = FakeLLM(
        [
            qa("Q made up quote?", "Revenue | $99,999"),  # quote not in passage -> rejected without judging
            qa("Q vague?", "Revenue | $14,766"),
            verdict(ok=False),  # judge: not self-contained -> rejected
            qa("What was Uber's revenue for Q3 2026?", "Revenue | $14,766"),
            verdict(),
        ]
    )
    candidates = [
        (rec(f"c{i}", "UBER_10-Q_period-2026-09-30_filed-2026-11-04.htm", TABLE),) for i in range(5)
    ]
    items = GoldenGenerator(llm).generate_category("exact_figure", candidates, quota=1)
    (item,) = items
    assert item.question == "What was Uber's revenue for Q3 2026?"
    assert item.answerable and item.category == "exact_figure" and not item.reviewed
    assert item.sources[0].chunk_id == "c2" and item.sources[0].file_name.startswith("UBER_10-Q")
    assert item.verification["quote_found"] is True
    assert llm.count("QAVerdict") == 2


def test_unusable_candidates_are_skipped_and_quota_respected() -> None:
    llm = FakeLLM(
        [
            qa("x", "", usable=False),
            qa("Q1?", "Revenue | $14,766"),
            verdict(),
            qa("Q2?", "Net income | $512"),
            verdict(),
        ]
    )
    candidates = [(rec(f"c{i}", "f.htm", TABLE),) for i in range(6)]
    items = GoldenGenerator(llm).generate_category("exact_figure", candidates, quota=2)
    assert [i.question for i in items] == ["Q1?", "Q2?"]
    assert [i.id for i in items] == ["exact_figure-001", "exact_figure-002"]


def test_only_listed_source_passages_become_gold_sources() -> None:
    other = rec("q2", "UBER_10-Q_period-2026-06-30_filed-2026-08-05.htm", TABLE.replace("14,766", "12,651"))
    llm = FakeLLM([qa("Q3 revenue?", "Revenue | $14,766", sources=[1]), verdict()])
    target = rec("q3", "UBER_10-Q_period-2026-09-30_filed-2026-11-04.htm", TABLE)
    (item,) = GoldenGenerator(llm).generate_category("temporal", [(target, other)], quota=1)
    assert [s.chunk_id for s in item.sources] == ["q3"]


def test_unanswerable_questions_are_dropped_if_the_corpus_answers_them() -> None:
    batch = UnanswerableBatch(
        questions=[
            UnanswerableQuestion(question="What was Uber's Q4 2026 revenue?", why="not filed yet"),
            UnanswerableQuestion(
                question="What was Uber's Q3 2026 revenue?", why="oops, it is in the corpus"
            ),
        ]
    )
    llm = FakeLLM(
        [
            batch,
            AnswerabilityVerdict(answerable=False, note=""),
            AnswerabilityVerdict(answerable=True, note=""),
        ]
    )

    def retrieve(question: str) -> list[tuple[str, str]]:
        return [("UBER 10-Q", TABLE)]

    items = GoldenGenerator(llm).unanswerable("coverage", count=2, retrieve=retrieve)
    assert [i.question for i in items] == ["What was Uber's Q4 2026 revenue?"]
    assert not items[0].answerable and items[0].sources == [] and items[0].category == "unanswerable"


# --- canary + persistence --------------------------------------------------------------------------


def test_canary_facts_become_items(tmp_path: Path) -> None:
    facts = tmp_path / "canary_facts.jsonl"
    facts.write_text(
        json.dumps(
            {
                "id": "pptx-1",
                "file": "canary/deck.pptx",
                "location": "slide 2",
                "question": "Q?",
                "answer": "A",
                "answerable": True,
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "neg-1",
                "file": "canary/deck.pptx",
                "location": "-",
                "question": "N?",
                "answer": "",
                "answerable": False,
            }
        )
        + "\n"
    )
    items = canary_items(facts, corpus_dir=Path("/corpus"))
    assert [(i.id, i.category, i.answerable) for i in items] == [
        ("canary-pptx-1", "canary", True),
        ("canary-neg-1", "canary_unanswerable", False),
    ]
    assert items[0].sources[0].file_name == "deck.pptx" and items[0].sources[0].location == "slide 2"
    assert items[1].sources == []


def test_golden_roundtrip(tmp_path: Path) -> None:
    item = GoldenItem(
        id="x-001", category="factual", question="Q?", reference_answer="A", answerable=True, sources=[],
        supporting_quote="", generator="test", verification={},
    )  # fmt: skip
    path = tmp_path / "golden.jsonl"
    save_golden([item], path)
    assert load_golden(path) == [item]
