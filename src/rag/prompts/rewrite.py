"""rewrite_query: standalone query, variants/sub-questions, explicit metadata filters."""

from __future__ import annotations

from collections.abc import Sequence

from rag.graph.state import Turn
from rag.prompts.common import render_transcript

VERSION = "rewrite-v1"

SYSTEM = (
    "You turn the latest user question into search queries over a private document collection.\n"
    "- standalone_query: rewrite the LAST user message so it is fully self-contained, resolving references to "
    'earlier turns ("the second one", "that company", "same quarter") using the conversation. Keep exact '
    "names, codes, numbers and dates.\n"
    "- variants: up to {max_variants} alternative phrasings that could match differently worded passages "
    "(synonyms, expanded acronyms, likely document wording). Empty if not useful.\n"
    "- sub_questions: if the question is complex or comparative, the separate facts to look up "
    "(e.g. one per company or period); otherwise empty.\n"
    "- filters: ONLY constraints the user stated explicitly: exact file names, a folder path, or a date range "
    "for the documents themselves (dates as YYYY-MM-DD). A period the question is ABOUT (e.g. revenue for Q3) "
    "is not a document filter. Leave everything empty/null when not explicitly requested.\n"
    "If feedback from a previous search attempt is given, write different queries that target the missing "
    "information."
)


def system(max_variants: int) -> str:
    return SYSTEM.replace("{max_variants}", str(max_variants))


def user_message(history: Sequence[Turn], *, previous_queries: Sequence[str], missing: str) -> str:
    parts = [render_transcript(history)]
    if previous_queries:
        tried = "\n".join(f"- {q}" for q in previous_queries)
        parts.append(
            f"<previous_attempt>\nQueries tried:\n{tried}\nMissing information: {missing or 'unspecified'}\n"
            "</previous_attempt>"
        )
    parts.append("Write the search plan for the last USER message.")
    return "\n\n".join(parts)
