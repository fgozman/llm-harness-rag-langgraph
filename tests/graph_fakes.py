"""Scripted LLM and retriever fakes for graph and API tests (no network, deterministic)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from rag.config import GraphSettings, LLMTask
from rag.graph.build import GraphDeps, build_graph, run_turn, state_serializer
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
    query: str = "standalone q",
    variants: Sequence[str] = ("variant",),
    file_names: Sequence[str] = (),
    folder: str | None = None,
) -> QueryPlan:
    return QueryPlan(
        standalone_query=query,
        variants=list(variants),
        sub_questions=[],
        filters=ExtractedFilters(
            file_names=list(file_names), folder=folder, date_after=None, date_before=None
        ),
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
DOCS_DIR = Path("/srv/docs")


def make(llm: FakeLLM, retriever: FakeRetriever, settings: GraphSettings = SETTINGS) -> Any:
    return build_graph(
        GraphDeps(llm=llm, retriever=retriever, settings=settings, docs_dir=DOCS_DIR),
        checkpointer=InMemorySaver(serde=state_serializer()),
    )


def ask(graph: Any, question: str, thread: str = "t1") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    state = run_turn(graph, question, thread_id=thread, on_event=events.append)
    return state, events
