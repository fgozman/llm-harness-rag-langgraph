"""grade_documents: per-passage relevance judgement."""

from __future__ import annotations

from collections.abc import Sequence

from rag.graph.state import Passage
from rag.prompts.common import UNTRUSTED_DOCUMENTS, render_passages

VERSION = "grade-v2"

SYSTEM = (
    "You judge whether retrieved passages are relevant to a question. A passage is relevant if it contains "
    "information that helps answer the question, even partially; it is not relevant if it is only on a related "
    "topic, or about a different entity or time period than the one asked about. Grade every passage. "
    "Set sufficient to true only if the relevant passages together fully answer the question. "
    "In missing_information, state what the question needs that none of the passages provide (empty if the "
    "passages are sufficient).\n" + UNTRUSTED_DOCUMENTS
)


def user_message(question: str, passages: Sequence[Passage]) -> str:
    return f"<question>\n{question}\n</question>\n\n{render_passages(passages)}"
