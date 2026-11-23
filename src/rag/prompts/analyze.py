"""analyze_query: intent classification and complexity detection."""

from __future__ import annotations

from collections.abc import Sequence

from rag.graph.state import Turn
from rag.prompts.common import render_transcript

VERSION = "analyze-v1"

SYSTEM = (
    "You route questions for an assistant that answers ONLY from the user's private document collection "
    "(company filings, reports, papers, policies, release notes and similar files).\n"
    "Classify the LAST user message of the conversation:\n"
    "- retrieval: anything that could be answered from documents, including follow-ups that refer to earlier "
    "turns. When in doubt, choose retrieval.\n"
    "- chitchat: greetings, thanks, or questions about the assistant itself.\n"
    "- out_of_scope: requests that clearly need no documents and are not about them (e.g. write a poem, general "
    "coding help, current weather).\n"
    "For chitchat and out_of_scope, write direct_reply: one or two friendly sentences; for out_of_scope explain "
    "that you only answer questions about the documents. For retrieval, direct_reply is null.\n"
    "is_complex is true for multi-part or comparative questions."
)


def user_message(history: Sequence[Turn]) -> str:
    return render_transcript(history) + "\n\nClassify the last USER message."
