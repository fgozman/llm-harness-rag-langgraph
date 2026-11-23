"""Graph nodes. Each node reads the state, does one job, and returns only the keys it changes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

from langgraph.config import get_stream_writer

from rag.config import GraphSettings
from rag.graph.citations import NOT_FOUND, build_citations, cautious_answer, strip_invalid_citations
from rag.graph.edges import relevant_indexes
from rag.graph.state import (
    AnswerCheck,
    Candidate,
    ExtractedFilters,
    FinalAnswer,
    GradeReport,
    GraphState,
    Passage,
    QueryAnalysis,
    QueryPlan,
    Turn,
)
from rag.llm.base import LLM, Message
from rag.prompts import analyze, check, generate, grade, rewrite
from rag.retrieval.filters import SearchFilters
from rag.retrieval.types import ContextPassage, RankedChunk

_DEFAULT_REPLY = "I answer questions about the documents in your collection. What would you like to know?"


class RetrieverPort(Protocol):
    def search(
        self, queries: Sequence[str], *, filters: SearchFilters | None = None
    ) -> list[RankedChunk]: ...

    def rerank_and_expand(
        self, query: str, candidates: Sequence[RankedChunk]
    ) -> tuple[list[RankedChunk], list[ContextPassage]]: ...


@dataclass(frozen=True, slots=True)
class GraphDeps:
    llm: LLM
    retriever: RetrieverPort
    settings: GraphSettings
    # Folder names the user mentions ("the contracts folder") are relative to the document collection.
    docs_dir: Path


def _emit(event: dict[str, Any]) -> None:
    # No-op unless the caller streams with stream_mode="custom".
    get_stream_writer()(event)


def _user(text: str) -> list[Message]:
    return [Message("user", text)]


def _parse_day(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        day = date.fromisoformat(value[:10])
    except ValueError:
        return None  # an unparseable date must not silently filter everything out
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def to_search_filters(extracted: ExtractedFilters, *, docs_dir: Path) -> SearchFilters | None:
    # Joining keeps absolute paths as they are and puts relative ones under docs_dir (not the process cwd).
    folder = (docs_dir / Path(extracted.folder).expanduser()).resolve() if extracted.folder else None
    filters = SearchFilters(
        file_names=tuple(extracted.file_names),
        folder=folder,
        date_after=_parse_day(extracted.date_after),
        date_before=_parse_day(extracted.date_before),
    )
    return None if filters.is_empty() else filters


def _to_candidate(c: RankedChunk) -> Candidate:
    return Candidate(
        chunk_id=c.chunk_id,
        score=c.score,
        fused_score=c.fused_score,
        rerank_score=c.rerank_score,
        payload=c.payload,
    )


def _to_ranked(c: Candidate) -> RankedChunk:
    return RankedChunk(c.chunk_id, c.score, c.fused_score, c.payload, rerank_score=c.rerank_score)


def _to_passage(p: ContextPassage, index: int | None = None) -> Passage:
    return Passage(
        index=index if index is not None else p.index,
        parent_id=p.parent_id,
        chunk_ids=list(p.chunk_ids),
        text=p.text,
        score=p.score,
        file_path=p.file_path,
        file_name=p.file_name,
        title=p.title,
        section_path=list(p.section_path),
        page_start=p.page_start,
        page_end=p.page_end,
        snippet=p.snippet,
    )


class Nodes:
    def __init__(self, deps: GraphDeps) -> None:
        self._llm = deps.llm
        self._retriever = deps.retriever
        self._cfg = deps.settings
        self._docs_dir = deps.docs_dir

    def _recent(self, state: GraphState) -> list[Turn]:
        history = state.get("history", [])
        return history[-(2 * self._cfg.history_turns + 1) :]

    def _question(self, state: GraphState) -> str:
        plan = state.get("plan")
        return plan.standalone_query if plan else state["question"]

    def analyze_query(self, state: GraphState) -> dict[str, Any]:
        _emit({"type": "status", "node": "analyze_query", "text": "analyzing question"})
        result = self._llm.structured(
            "analyze",
            system=analyze.SYSTEM,
            messages=_user(analyze.user_message(self._recent(state))),
            schema=QueryAnalysis,
        )
        # Reset every per-turn field: the checkpointer carries the previous turn's values forward.
        return {
            "analysis": result,
            "plan": None,
            "queries": [],
            "filters_dropped": False,
            "candidates": [],
            "reranked": [],
            "passages": [],
            "grades": None,
            "retrieval_attempts": 0,
            "generation_count": 0,
            "answer": "",
            "answer_passages": [],
            "abstained": False,
            "check": None,
            "final": None,
        }

    def rewrite_query(self, state: GraphState) -> dict[str, Any]:
        attempts = state.get("retrieval_attempts", 0)
        report = state.get("grades")
        message = rewrite.user_message(
            self._recent(state),
            previous_queries=state.get("queries", []) if attempts else [],
            missing=report.missing_information if report and attempts else "",
        )
        plan = self._llm.structured(
            "rewrite",
            system=rewrite.system(self._cfg.max_query_variants),
            messages=_user(message),
            schema=QueryPlan,
        )
        extra = [q for q in (*plan.variants, *plan.sub_questions) if q.strip()]
        queries = list(dict.fromkeys([plan.standalone_query, *extra]))[: 1 + self._cfg.max_query_variants]
        _emit({"type": "status", "node": "rewrite_query", "text": f"searching: {plan.standalone_query}"})
        return {"plan": plan, "queries": queries, "retrieval_attempts": attempts + 1}

    def retrieve(self, state: GraphState) -> dict[str, Any]:
        plan = state.get("plan")
        filters = to_search_filters(plan.filters, docs_dir=self._docs_dir) if plan else None
        found = self._retriever.search(state.get("queries", []), filters=filters)
        dropped = False
        if not found and filters is not None:
            # Filters extracted from the question matched nothing: better to search everything than to fail.
            found = self._retriever.search(state.get("queries", []), filters=None)
            dropped = True
        return {"candidates": [_to_candidate(c) for c in found], "filters_dropped": dropped}

    def rerank(self, state: GraphState) -> dict[str, Any]:
        candidates = [_to_ranked(c) for c in state.get("candidates", [])]
        ranked, passages = self._retriever.rerank_and_expand(self._question(state), candidates)
        return {
            "reranked": [_to_candidate(c) for c in ranked],
            "passages": [_to_passage(p) for p in passages],
        }

    def grade_documents(self, state: GraphState) -> dict[str, Any]:
        passages = state.get("passages", [])
        if not passages:
            return {
                "grades": GradeReport(
                    grades=[], sufficient=False, missing_information="No passages were retrieved."
                )
            }
        report = self._llm.structured(
            "grade",
            system=grade.SYSTEM,
            messages=_user(grade.user_message(self._question(state), passages)),
            schema=GradeReport,
        )
        valid = {p.index for p in passages}
        # Ignore grades for passage numbers that don't exist.
        report = report.model_copy(update={"grades": [g for g in report.grades if g.index in valid]})
        return {"grades": report}

    def generate(self, state: GraphState) -> dict[str, Any]:
        count = state.get("generation_count", 0)
        relevant = set(relevant_indexes(state))
        # Renumber the relevant passages 1..n so citations match what the model sees.
        selected = [p for p in state.get("passages", []) if p.index in relevant]
        answer_passages = [p.model_copy(update={"index": i}) for i, p in enumerate(selected, start=1)]
        if not answer_passages:
            return {
                "answer": NOT_FOUND,
                "answer_passages": [],
                "abstained": True,
                "generation_count": count + 1,
            }
        previous = state.get("check")
        critique = previous.critique if count and previous else None
        if critique:
            _emit({"type": "status", "node": "generate", "text": "revising answer after verification"})
        message = generate.user_message(self._question(state), answer_passages, critique=critique)
        _emit({"type": "answer_start", "attempt": count + 1})
        text = self._llm.generate(
            "generate",
            system=generate.SYSTEM,
            messages=_user(message),
            on_token=lambda token: _emit({"type": "token", "text": token}),
        )
        return {
            "answer": strip_invalid_citations(text.strip(), answer_passages),
            "answer_passages": answer_passages,
            "abstained": False,
            "generation_count": count + 1,
        }

    def check_answer(self, state: GraphState) -> dict[str, Any]:
        _emit({"type": "status", "node": "check_answer", "text": "verifying answer"})
        result = self._llm.structured(
            "check",
            system=check.SYSTEM,
            messages=_user(
                check.user_message(
                    self._question(state), state.get("answer_passages", []), state.get("answer", "")
                )
            ),
            schema=AnswerCheck,
        )
        return {"check": result}

    def respond(self, state: GraphState) -> dict[str, Any]:
        final = self._final(state)
        _emit({"type": "final", "status": final.status})
        return {"final": final, "history": [Turn(role="assistant", content=final.text)]}

    def _final(self, state: GraphState) -> FinalAnswer:
        analysis = state.get("analysis")
        if analysis is not None and analysis.intent != "retrieval":
            return FinalAnswer(
                text=analysis.direct_reply or _DEFAULT_REPLY, status=analysis.intent, citations=[]
            )
        if state.get("abstained", False):
            return FinalAnswer(text=NOT_FOUND, status="abstained", citations=[])
        answer, passages, result = (
            state.get("answer", ""),
            state.get("answer_passages", []),
            state.get("check"),
        )
        if result is None or result.is_abstention or (result.grounded and result.addresses_question):
            status = "abstained" if result is not None and result.is_abstention else "answered"
            return FinalAnswer(text=answer, status=status, citations=build_citations(answer, passages))
        # Verification still failed after the regeneration cap: say what could and couldn't be verified.
        text, citations = cautious_answer(
            supported=result.supported_claims, unsupported=result.unsupported_claims, passages=passages
        )
        status = "abstained" if text.startswith(NOT_FOUND) else "cautious"
        return FinalAnswer(text=text, status=status, citations=citations)
