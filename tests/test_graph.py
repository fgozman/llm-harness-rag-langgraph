import logging
from pathlib import Path

import pytest

from graph_fakes import (
    DOCS_DIR,
    SETTINGS,
    THREE,
    FakeLLM,
    FakeRetriever,
    analysis,
    ask,
    bad,
    grades,
    make,
    ok,
    plan,
)
from rag.config import GraphSettings
from rag.graph.build import GraphDeps, build_checkpointer, build_graph, run_turn
from rag.graph.citations import NOT_FOUND
from rag.graph.state import (
    SupportedClaim,
)
from rag.retrieval.filters import SearchFilters

# --- scenarios -------------------------------------------------------------------------------------


def test_answer_with_citations_mapped_to_sources_and_streamed_tokens() -> None:
    llm = FakeLLM([analysis(), plan(), grades(True, False, True), ok()], ["Revenue 10 [1] and costs 4 [2]."])
    retriever = FakeRetriever([THREE])
    state, events = ask(make(llm, retriever), "What were revenue and costs?")
    final = state["final"]
    assert final.status == "answered"
    assert final.text == "Revenue 10 [1] and costs 4 [2]."
    # Only relevant passages reach the generator, renumbered 1..n: [2] is now file c.
    assert [(c.index, c.file_name) for c in final.citations] == [(1, "a.pdf"), (2, "c.pdf")]
    assert "".join(e["text"] for e in events if e["type"] == "token").strip() == final.text
    assert retriever.searches[0][0] == ["standalone q", "variant"]
    assert retriever.rerank_queries == ["standalone q"]
    assert [t.role for t in state["history"]] == ["user", "assistant"]
    assert state["history"][-1].content == final.text


@pytest.mark.parametrize("intent", ["chitchat", "out_of_scope"])
def test_direct_replies_skip_retrieval(intent: str) -> None:
    llm = FakeLLM([analysis(intent, reply="I answer questions about your documents.")])
    retriever = FakeRetriever([THREE])
    state, _ = ask(make(llm, retriever), "hello")
    assert state["final"].status == intent
    assert state["final"].text == "I answer questions about your documents."
    assert retriever.searches == [] and llm.count("text") == 0


def test_too_few_relevant_passages_retries_rewrite_with_feedback_up_to_cap() -> None:
    llm = FakeLLM(
        [
            analysis(),
            plan("q1"),
            grades(True, False, False, missing="Q3 figure"),
            plan("q2"),
            grades(True, False, False, missing="still Q3"),
            plan("q3"),
            grades(True, False, False),
            ok(),
        ],
        ["Revenue 10 [1]."],
    )
    retriever = FakeRetriever([THREE])
    state, _ = ask(make(llm, retriever), "q")
    assert llm.count("QueryPlan") == 3  # 1 initial + max_rewrite_retries (2)
    assert state["retrieval_attempts"] == 3
    second_rewrite = llm.prompts("QueryPlan")[1]
    assert "Q3 figure" in second_rewrite and "q1" in second_rewrite
    assert state["final"].status == "answered"  # proceeds anyway with the one relevant passage


def test_single_sufficient_passage_answers_without_retrying() -> None:
    llm = FakeLLM(
        [analysis(), plan(), grades(True, False, False, sufficient=True), ok()], ["Range 0-45 C [1]."]
    )
    retriever = FakeRetriever([THREE])
    state, _ = ask(make(llm, retriever), "temperature range?")
    assert len(retriever.searches) == 1 and state["retrieval_attempts"] == 1
    assert state["final"].status == "answered"


def test_no_relevant_passages_after_cap_abstains_without_generating() -> None:
    llm = FakeLLM([analysis(), *(x for _ in range(3) for x in (plan(), grades(False, False, False)))])
    state, _ = ask(make(llm, FakeRetriever([THREE])), "q")
    assert state["final"].status == "abstained"
    assert state["final"].text.startswith(NOT_FOUND)
    assert state["final"].citations == []
    assert llm.count("text") == 0 and llm.count("AnswerCheck") == 0


def test_empty_retrieval_abstains() -> None:
    llm = FakeLLM([analysis(), plan(), grades(), plan(), grades(), plan(), grades()])
    state, _ = ask(make(llm, FakeRetriever([[]])), "q")
    assert state["final"].status == "abstained"


def test_failed_check_regenerates_once_with_critique_then_passes() -> None:
    llm = FakeLLM(
        [analysis(), plan(), grades(True, True, True), bad("Wrong number"), ok()], ["Bad [1].", "Good [1]."]
    )
    state, events = ask(make(llm, FakeRetriever([THREE])), "q")
    assert llm.count("text") == 2
    assert "Wrong number" in llm.prompts("text")[1]
    assert state["final"].status == "answered" and state["final"].text == "Good [1]."
    assert any(e["type"] == "status" and "revis" in e["text"] for e in events)


def test_failed_check_twice_returns_cautious_answer() -> None:
    claims = [SupportedClaim(claim="Revenue was 10", citations=[1])]
    llm = FakeLLM(
        [analysis(), plan(), grades(True, True, True), bad(), bad(supported=claims)],
        ["Bad [1].", "Still bad [1]."],
    )
    state, _ = ask(make(llm, FakeRetriever([THREE])), "q")
    assert llm.count("text") == 2  # 1 + max_regenerations
    final = state["final"]
    assert final.status == "cautious"
    assert "Revenue was 10 [1]" in final.text and "Margins doubled" in final.text
    assert [c.file_name for c in final.citations] == ["a.pdf"]


def test_model_abstention_is_reported_as_abstained() -> None:
    llm = FakeLLM(
        [analysis(), plan(), grades(True, True, True), ok(abstention=True)], [f"{NOT_FOUND} Only costs [3]."]
    )
    state, _ = ask(make(llm, FakeRetriever([THREE])), "q")
    assert state["final"].status == "abstained"
    assert [c.index for c in state["final"].citations] == [3]


def test_invalid_citations_are_removed() -> None:
    llm = FakeLLM([analysis(), plan(), grades(True, True, False), ok()], ["A [1]. B [9]."])
    state, _ = ask(make(llm, FakeRetriever([THREE])), "q")
    assert state["final"].text == "A [1]. B."
    assert [c.index for c in state["final"].citations] == [1]


def test_follow_up_sees_history_and_turn_state_resets() -> None:
    llm = FakeLLM(
        [
            analysis(),
            plan("first"),
            grades(True, True, True),
            ok(),
            analysis(),
            plan("second"),
            grades(True, True, True),
            ok(),
        ],
        ["One [1].", "Two [2]."],
    )
    graph = make(llm, FakeRetriever([THREE]))
    ask(graph, "What are the products?")
    state, _ = ask(graph, "What about the second one?")
    rewrite_prompt = llm.prompts("QueryPlan")[1]
    assert "What are the products?" in rewrite_prompt and "One [1]." in rewrite_prompt
    assert "What about the second one?" in rewrite_prompt
    assert state["retrieval_attempts"] == 1 and state["generation_count"] == 1
    assert [t.content for t in state["history"]] == [
        "What are the products?",
        "One [1].",
        "What about the second one?",
        "Two [2].",
    ]


def test_threads_are_isolated() -> None:
    llm = FakeLLM([analysis(), plan(), grades(True, True, True), ok()] * 2, ["One [1].", "Two [1]."])
    graph = make(llm, FakeRetriever([THREE]))
    ask(graph, "q1", thread="a")
    state, _ = ask(graph, "q2", thread="b")
    assert [t.content for t in state["history"]] == ["q2", "Two [1]."]


def test_explicit_filters_are_applied_and_dropped_when_they_match_nothing() -> None:
    llm = FakeLLM([analysis(), plan(file_names=["missing.pdf"]), grades(True, True, True), ok()], ["A [1]."])
    retriever = FakeRetriever([THREE])
    state, _ = ask(make(llm, retriever), "q in missing.pdf")
    assert retriever.searches[0][1] == SearchFilters(file_names=("missing.pdf",))
    assert retriever.searches[1][1] is None
    assert state["filters_dropped"] is True


@pytest.mark.parametrize(
    ("folder", "expected"),
    [("contracts", DOCS_DIR / "contracts"), ("/elsewhere/reports", Path("/elsewhere/reports"))],
)
def test_folder_filter_is_relative_to_docs_dir(folder: str, expected: Path) -> None:
    llm = FakeLLM([analysis(), plan(folder=folder), grades(True, True, True), ok()], ["A [1]."])
    retriever = FakeRetriever([THREE])
    ask(make(llm, retriever), "q in that folder")
    filters = retriever.searches[0][1]
    assert filters is not None and filters.folder == expected.resolve()


def test_query_variants_are_capped() -> None:
    many = plan(variants=[f"v{i}" for i in range(10)])
    llm = FakeLLM([analysis(), many, grades(True, True, True), ok()], ["A [1]."])
    retriever = FakeRetriever([THREE])
    ask(make(llm, retriever, GraphSettings(max_query_variants=2)), "q")
    assert retriever.searches[0][0] == ["standalone q", "v0", "v1"]


def test_node_tasks_use_configured_llm_roles() -> None:
    llm = FakeLLM([analysis(), plan(), grades(True, True, True), ok()], ["A [1]."])
    ask(make(llm, FakeRetriever([THREE])), "q")
    assert [(task, name) for task, name, _ in llm.calls] == [
        ("analyze", "QueryAnalysis"),
        ("rewrite", "QueryPlan"),
        ("grade", "GradeReport"),
        ("generate", "text"),
        ("check", "AnswerCheck"),
    ]


def test_documents_are_delimited_and_marked_untrusted_for_generation() -> None:
    llm = FakeLLM([analysis(), plan(), grades(True, True, True), ok()], ["A [1]."])
    ask(make(llm, FakeRetriever([THREE])), "q")
    prompt = llm.prompts("text")[0]
    assert '<passage index="1" source="a.pdf, p. 3, S">' in prompt and "</documents>" in prompt


def test_sqlite_checkpointer_persists_threads_across_graph_instances(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    scripted = [analysis(), plan(), grades(True, True, True), ok()]
    db = tmp_path / "checkpoints.db"
    first = build_graph(
        GraphDeps(
            llm=FakeLLM(scripted, ["One [1]."]),
            retriever=FakeRetriever([THREE]),
            settings=SETTINGS,
            docs_dir=DOCS_DIR,
        ),
        checkpointer=build_checkpointer(db),
    )
    run_turn(first, "q1", thread_id="t")
    # A new process/graph sees the same conversation (e.g. `rag chat --thread-id t`).
    second = build_graph(
        GraphDeps(
            llm=FakeLLM(scripted, ["Two [1]."]),
            retriever=FakeRetriever([THREE]),
            settings=SETTINGS,
            docs_dir=DOCS_DIR,
        ),
        checkpointer=build_checkpointer(db),
    )
    with caplog.at_level(logging.WARNING):
        state = run_turn(second, "q2", thread_id="t")
    assert [t.content for t in state["history"]] == ["q1", "One [1].", "q2", "Two [1]."]
    assert "unregistered type" not in caplog.text
