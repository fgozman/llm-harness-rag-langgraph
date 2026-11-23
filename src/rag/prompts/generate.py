"""generate: grounded answer with inline citations."""

from __future__ import annotations

from collections.abc import Sequence

from rag.graph.citations import NOT_FOUND
from rag.graph.state import Passage
from rag.prompts.common import UNTRUSTED_DOCUMENTS, render_passages

VERSION = "generate-v1"

SYSTEM = (
    "You answer questions using ONLY the passages in <documents>. Rules:\n"
    "1. Every factual statement must be supported by the passages and followed by its citation(s) in square "
    "brackets with the passage index, e.g. [1] or [2][3]. Never cite a passage that does not support the "
    "statement. Do not use outside knowledge.\n"
    f'2. If the passages do not contain the answer, reply exactly "{NOT_FOUND}" and then, in one sentence, '
    "what related information the passages do contain (with citations), if any.\n"
    "3. If the passages only partly answer, answer that part and state clearly what is not covered.\n"
    "4. Pay close attention to entities, periods and dates: do not mix up quarters, years or companies.\n"
    "5. Be concise and direct; use a short list or table only when comparing several items.\n"
    + UNTRUSTED_DOCUMENTS
)


def user_message(question: str, passages: Sequence[Passage], *, critique: str | None) -> str:
    parts = [render_passages(passages), f"<question>\n{question}\n</question>"]
    if critique:
        parts.append(
            "<review_of_previous_answer>\n"
            f"{critique}\n"
            "</review_of_previous_answer>\nWrite a corrected answer that fixes these problems."
        )
    return "\n\n".join(parts)
