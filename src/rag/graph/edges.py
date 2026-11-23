"""Conditional edges: decisions read structured fields only (never free text); every loop is capped."""

from __future__ import annotations

from typing import Literal

from rag.graph.state import GraphState


def route_after_analyze(state: GraphState) -> Literal["rewrite_query", "respond"]:
    analysis = state.get("analysis")
    return "rewrite_query" if analysis is not None and analysis.intent == "retrieval" else "respond"


def relevant_indexes(state: GraphState) -> list[int]:
    report = state.get("grades")
    return [g.index for g in report.grades if g.relevant] if report else []


def route_after_grade(
    state: GraphState, *, min_relevant: int, max_attempts: int
) -> Literal["generate", "rewrite_query"]:
    relevant = len(relevant_indexes(state))
    report = state.get("grades")
    # Enough evidence: several relevant passages, or at least one the grader judged sufficient on its own
    # (single-fact questions often have exactly one relevant passage).
    if relevant >= min_relevant or (relevant >= 1 and report is not None and report.sufficient):
        return "generate"
    # Too little evidence: retry retrieval with feedback until the cap, then answer with what we have.
    return "rewrite_query" if state.get("retrieval_attempts", 0) < max_attempts else "generate"


def route_after_generate(state: GraphState) -> Literal["check_answer", "respond"]:
    # A deterministic "not found" (no relevant passages) has nothing to verify.
    return "respond" if state.get("abstained", False) else "check_answer"


def route_after_check(state: GraphState, *, max_generations: int) -> Literal["respond", "generate"]:
    check = state.get("check")
    if check is None or check.is_abstention or (check.grounded and check.addresses_question):
        return "respond"
    # Failed verification: one more attempt with the critique, then respond cautiously.
    return "generate" if state.get("generation_count", 0) < max_generations else "respond"
