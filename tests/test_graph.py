import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from rag.config import GraphSettings, LLMTask
from rag.graph.build import GraphDeps, build_checkpointer, build_graph, run_turn, state_serializer
from rag.graph.citations import NOT_FOUND
from rag.graph.state import (
    AnswerCheck,
    ExtractedFilters,
    GradeReport,
    PassageGrade,
    QueryAnalysis,
    QueryPlan,
    SupportedClaim,
)
from rag.llm.base import Message, TokenCallback
from rag.retrieval.filters import SearchFilters
from rag.retrieval.types import ContextPassage, RankedChunk

# --- fakes -----------------------------------------------------------------------------------------


class FakeLLM:
    def __init__(self, structured: Sequence[BaseModel] = (), answers: Sequence[str] = ()) -> None:
        self.queue: dict[str, list[BaseModel]] = {}
        for item in structured:
            self.queue.setdefault(type(item).__name__, []).append(item)
        self.answers = list(answers)
        self.calls: list[tuple[LLMTask, str, str]] = []  # (task, schema or "text", last user message)

    def structured[T: BaseModel](
        self, task: LLMTask, *, system: str, messages: Sequence[Message], schema: type[T]
    ) -> T:
        self.calls.append((task, schema.__name__, messages[-1].content))
        item = self.queue[schema.__name__].pop(0)
        assert isinstance(item, schema)
        return item

    def generate(
        self,
        task: LLMTask,
        *,
        system: str,
        messages: Sequence[Message],
        on_token: TokenCallback | None = None,
    ) -> str:
        self.calls.append((task, "text", messages[-1].content))
        answer = self.answers.pop(0)
        for word in answer.split(" "):
            if on_token:
                on_token(word + " ")
        return answer

    def count(self, schema_or_text: str) -> int:
        return sum(1 for _, name, _ in self.calls if name == schema_or_text)

    def prompts(self, schema_or_text: str) -> list[str]:
        return [msg for _, name, msg in self.calls if name == schema_or_text]


def chunk(cid: str, text: str) -> RankedChunk:
    payload: dict[str, object] = {
        "parent_id": f"P-{cid}",
        "text": text,
        "file_path": f"/docs/{cid}.pdf",
        "file_name": f"{cid}.pdf",
        "title": f"Title {cid}",
        "section_path": ["S"],
        "page_start": 3,
        "page_end": 3,
        "chunk_index": 0,
    }
    return RankedChunk(cid, 0.9, 0.03, payload, rerank_score=0.9)


class FakeRetriever:
    def __init__(self, results: Sequence[Sequence[RankedChunk]]) -> None:
        self.results = [list(r) for r in results]
        self.searches: list[tuple[list[str], SearchFilters | None]] = []
        self.rerank_queries: list[str] = []

    def search(self, queries: Sequence[str], *, filters: SearchFilters | None = None) -> list[RankedChunk]:
        self.searches.append((list(queries), filters))
        index = min(len(self.searches) - 1, len(self.results) - 1)
        return self.results[index] if not (filters and filters.file_names == ("missing.pdf",)) else []

    def rerank_and_expand(
        self, query: str, candidates: Sequence[RankedChunk]
    ) -> tuple[list[RankedChunk], list[ContextPassage]]:
        self.rerank_queries.append(query)
        passages = [
            ContextPassage(
                index=i + 1,
                parent_id=str(c.payload["parent_id"]),
                chunk_ids=(c.chunk_id,),
                text=str(c.payload["text"]),
                expanded=True,
                score=c.score,
                file_path=str(c.payload["file_path"]),
                file_name=str(c.payload["file_name"]),
                title=str(c.payload["title"]),
                section_path=("S",),
                page_start=3,
                page_end=3,
                snippet=str(c.payload["text"]),
            )
            for i, c in enumerate(candidates)
        ]
        return list(candidates), passages


# --- scripted outputs ------------------------------------------------------------------------------


def analysis(intent: str = "retrieval", reply: str | None = None) -> QueryAnalysis:
    return QueryAnalysis.model_validate({"intent": intent, "is_complex": False, "direct_reply": reply})


def plan(
    query: str = "standalone q", variants: Sequence[str] = ("variant",), file_names: Sequence[str] = ()
) -> QueryPlan:
    return QueryPlan(
        standalone_query=query,
        variants=list(variants),
        sub_questions=[],
        filters=ExtractedFilters(file_names=list(file_names), folder=None, date_after=None, date_before=None),
    )


def grades(*relevant: bool, missing: str = "", sufficient: bool = False) -> GradeReport:
    return GradeReport(
        grades=[PassageGrade(index=i + 1, relevant=r, reason="r") for i, r in enumerate(relevant)],
        sufficient=sufficient,
        missing_information=missing,
    )


def ok(abstention: bool = False) -> AnswerCheck:
    return AnswerCheck(
        grounded=not abstention,
        addresses_question=True,
        is_abstention=abstention,
        supported_claims=[],
        unsupported_claims=[],
        critique="",
    )


def bad(critique: str = "Fix the revenue number", supported: Sequence[SupportedClaim] = ()) -> AnswerCheck:
    return AnswerCheck(
        grounded=False,
        addresses_question=True,
        is_abstention=False,
        supported_claims=list(supported),
        unsupported_claims=["Margins doubled"],
        critique=critique,
    )


THREE = [chunk("a", "Revenue was 10."), chunk("b", "Unrelated."), chunk("c", "Costs were 4.")]
SETTINGS = GraphSettings(max_rewrite_retries=2, max_regenerations=1, min_relevant_passages=2)


def make(llm: FakeLLM, retriever: FakeRetriever, settings: GraphSettings = SETTINGS) -> Any:
    return build_graph(
        GraphDeps(llm=llm, retriever=retriever, settings=settings),
        checkpointer=InMemorySaver(serde=state_serializer()),
    )


def ask(graph: Any, question: str, thread: str = "t1") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    state = run_turn(graph, question, thread_id=thread, on_event=events.append)
    return state, events


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
        GraphDeps(llm=FakeLLM(scripted, ["One [1]."]), retriever=FakeRetriever([THREE]), settings=SETTINGS),
        checkpointer=build_checkpointer(db),
    )
    run_turn(first, "q1", thread_id="t")
    # A new process/graph sees the same conversation (e.g. `rag chat --thread-id t`).
    second = build_graph(
        GraphDeps(llm=FakeLLM(scripted, ["Two [1]."]), retriever=FakeRetriever([THREE]), settings=SETTINGS),
        checkpointer=build_checkpointer(db),
    )
    with caplog.at_level(logging.WARNING):
        state = run_turn(second, "q2", thread_id="t")
    assert [t.content for t in state["history"]] == ["q1", "One [1].", "q2", "Two [1]."]
    assert "unregistered type" not in caplog.text
