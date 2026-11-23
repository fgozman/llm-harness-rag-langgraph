"""check_answer: groundedness and relevance verification."""

from __future__ import annotations

from collections.abc import Sequence

from rag.graph.citations import NOT_FOUND
from rag.graph.state import Passage
from rag.prompts.common import UNTRUSTED_DOCUMENTS, render_passages

VERSION = "check-v1"

SYSTEM = (
    "You verify an answer against the passages it was written from.\n"
    "- grounded: true only if every factual claim in the answer is supported by the passages it cites "
    "(numbers, names, dates and periods must match exactly).\n"
    "- addresses_question: true if the answer responds to what was asked (an honest statement that the "
    "documents lack the answer counts as addressing it).\n"
    f'- is_abstention: true if the answer says the information was not found (e.g. "{NOT_FOUND}").\n'
    "- supported_claims: each supported claim with the passage numbers that support it.\n"
    "- unsupported_claims: claims not supported by the passages (including wrong citations).\n"
    "- critique: concrete instructions to fix the answer; empty if it passes.\n" + UNTRUSTED_DOCUMENTS
)


def user_message(question: str, passages: Sequence[Passage], answer: str) -> str:
    return (
        f"{render_passages(passages)}\n\n<question>\n{question}\n</question>\n\n<answer>\n{answer}\n</answer>"
    )
