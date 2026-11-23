"""Graph state and the structured-output schemas that drive routing.

LLM-facing schemas avoid numeric/length constraints (unsupported by Anthropic structured outputs) and use
nullable fields instead of optional ones (OpenAI strict mode requires every property).
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, Required, TypedDict

from pydantic import BaseModel, Field

# --- conversation ---------------------------------------------------------------------------------


class Turn(BaseModel):
    """Plain text only: thinking blocks never enter history, so they can't leak across models."""

    role: Literal["user", "assistant"]
    content: str


# --- structured LLM outputs -----------------------------------------------------------------------


class QueryAnalysis(BaseModel):
    intent: Literal["retrieval", "chitchat", "out_of_scope"] = Field(
        description="retrieval: needs the document collection; chitchat: greetings/small talk/questions "
        "about the assistant; out_of_scope: clearly unrelated to documents and needing outside knowledge."
    )
    is_complex: bool = Field(
        description="Multi-part or comparative question that benefits from decomposition."
    )
    direct_reply: str | None = Field(
        description="Short reply for chitchat/out_of_scope (e.g. explain you answer from the documents); "
        "null for retrieval."
    )


class ExtractedFilters(BaseModel):
    file_names: list[str] = Field(
        description="Exact file names explicitly mentioned by the user; else empty."
    )
    folder: str | None = Field(description="A folder path explicitly mentioned by the user; else null.")
    date_after: str | None = Field(
        description="ISO date (YYYY-MM-DD) if the user explicitly restricts to documents dated on/after it."
    )
    date_before: str | None = Field(
        description="ISO date (YYYY-MM-DD) if the user explicitly restricts to documents dated on/before it."
    )


class QueryPlan(BaseModel):
    standalone_query: str = Field(description="The latest user question rewritten to be self-contained.")
    variants: list[str] = Field(description="0-3 alternative phrasings (synonyms, expanded acronyms).")
    sub_questions: list[str] = Field(
        description="For complex questions: the separate facts to look up; else empty."
    )
    filters: ExtractedFilters


class PassageGrade(BaseModel):
    index: int = Field(description="Passage number as shown in the <passage index=...> tag.")
    relevant: bool = Field(description="True if the passage helps answer the question.")
    reason: str = Field(description="One short sentence.")


class GradeReport(BaseModel):
    grades: list[PassageGrade]
    sufficient: bool = Field(
        description="True if the relevant passages together fully answer the question (no retrieval needed)."
    )
    missing_information: str = Field(
        description="What information the question needs that the passages lack; empty if nothing is missing."
    )


class SupportedClaim(BaseModel):
    claim: str
    citations: list[int] = Field(description="Passage numbers that support the claim.")


class AnswerCheck(BaseModel):
    grounded: bool = Field(description="Every factual claim is supported by the cited passages.")
    addresses_question: bool = Field(description="The answer responds to what was asked.")
    is_abstention: bool = Field(description="The answer says the documents do not contain the answer.")
    supported_claims: list[SupportedClaim]
    unsupported_claims: list[str] = Field(description="Claims not supported by the passages.")
    critique: str = Field(description="Concrete instructions to fix the answer; empty if it passes.")


# --- retrieval artefacts carried in state ---------------------------------------------------------


class Candidate(BaseModel):
    chunk_id: str
    score: float
    fused_score: float
    rerank_score: float | None = None
    payload: dict[str, object]


class Passage(BaseModel):
    index: int
    parent_id: str
    chunk_ids: list[str]
    text: str
    score: float
    file_path: str
    file_name: str
    title: str
    section_path: list[str]
    page_start: int | None
    page_end: int | None
    snippet: str  # the retrieved child chunk's text, quoted in citations


class Citation(BaseModel):
    index: int
    file_path: str
    file_name: str
    title: str
    page_start: int | None
    page_end: int | None
    section: str
    snippet: str


AnswerStatus = Literal["answered", "abstained", "cautious", "chitchat", "out_of_scope"]


class FinalAnswer(BaseModel):
    text: str
    status: AnswerStatus
    citations: list[Citation]


# --- graph state ----------------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    history: Annotated[list[Turn], operator.add]  # accumulates across turns of a thread
    question: Required[str]  # always supplied as turn input
    # Per-turn fields below are reset by analyze_query at the start of every turn.
    analysis: QueryAnalysis | None
    plan: QueryPlan | None
    queries: list[str]
    filters_dropped: bool
    candidates: list[Candidate]  # retrieved (fused) pool, before reranking
    reranked: list[Candidate]
    passages: list[Passage]
    grades: GradeReport | None
    retrieval_attempts: int
    generation_count: int
    answer: str
    answer_passages: list[Passage]  # the (renumbered) passages the answer cites
    abstained: bool
    check: AnswerCheck | None
    final: FinalAnswer | None


STATE_MODELS: tuple[type[BaseModel], ...] = (
    Turn,
    QueryAnalysis,
    ExtractedFilters,
    QueryPlan,
    PassageGrade,
    GradeReport,
    SupportedClaim,
    AnswerCheck,
    Candidate,
    Passage,
    Citation,
    FinalAnswer,
)
