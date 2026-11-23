"""Golden-set generation and verification prompts."""

from __future__ import annotations

from collections.abc import Sequence

VERSION = "golden-v1"

CATEGORY_INSTRUCTIONS: dict[str, str] = {
    "exact_figure": (
        "Ask for ONE specific number stated in the passage (e.g. revenue, net income, margin, users, bookings). "
        "The question must name the company, the metric and the exact period (e.g. 'the three months ended "
        "September 30, 2026'). The answer is the number with its unit."
    ),
    "temporal": (
        "The passages come from filings of the SAME company for DIFFERENT periods. Pick one metric reported in both "
        "with different values, and ask for it for ONE explicitly named period, so that a system confusing the "
        "periods would answer wrongly. The answer is the value for the asked period only. In source_passages list "
        "only the passage that answers."
    ),
    "comparison": (
        "The passages come from two filings of the SAME company for different periods. Ask how one metric changed "
        "between the two periods. The answer states both values (with periods) and the direction/size of the change. "
        "List both passages in source_passages."
    ),
    "table_lookup": (
        "Ask for one specific value from the table in the passage, identifying the document (paper title or report "
        "name/number) and what the row and column represent. The answer is the value (with unit if any)."
    ),
    "release_notes": (
        "Ask which release/version of the named project introduced, fixed or changed one specific item described in "
        "the passage. The answer is the version (and date if given)."
    ),
    "factual": (
        "Ask a specific factual question answerable from the passage alone; name the document (title, agency, report "
        "number or paper title) in the question. The answer is short and specific."
    ),
}

SYSTEM = (
    "You write evaluation questions for a retrieval-augmented QA system over a private document collection. "
    "Each question must be answerable ONLY from the given passages and must be fully self-contained: a reader who "
    "has not seen the passages must know which document, entity and period it refers to. Never use phrases like "
    "'the passage', 'this document' or 'the table above'. Copy supporting_quote VERBATIM from a passage (one "
    "sentence or one table row, at most 300 characters). If the passages don't allow a good question of the "
    "requested kind, set usable to false."
)

VERIFY_SYSTEM = (
    "You audit evaluation questions for a document QA benchmark. Judge strictly:\n"
    "- self_contained: the question identifies the document/entity/period so it can be answered without seeing "
    "the passages (it must not say 'the passage', 'this report', etc.).\n"
    "- supported: the reference answer is fully supported by the passages (numbers, units and periods match).\n"
    "- unambiguous: exactly one correct answer exists for the question as worded."
)

UNANSWERABLE_SYSTEM = (
    "You write UNANSWERABLE evaluation questions for a QA system over a document collection whose coverage is "
    "described below. Each question must sound plausible for this collection (same companies, topics, report "
    "types and time frames) but ask for information that is NOT in it, for example: a quarter or year that is not "
    "covered, a company or project that is not in the collection, or a specific detail these document types "
    "would not contain. Make each question self-contained and specific."
)

ANSWERABILITY_SYSTEM = (
    "Decide whether the passages contain the information needed to answer the question. answerable is true only "
    "if the passages state the answer (not just related information)."
)


def describe_passages(passages: Sequence[tuple[str, str]]) -> str:
    """passages: (source description, text) pairs, numbered from 1."""
    return "\n\n".join(
        f'<passage index="{i}" source="{source}">\n{text}\n</passage>'
        for i, (source, text) in enumerate(passages, 1)
    )


def generation_message(category: str, passages: Sequence[tuple[str, str]]) -> str:
    return f"{describe_passages(passages)}\n\nTask: {CATEGORY_INSTRUCTIONS[category]}"


def verification_message(question: str, answer: str, passages: Sequence[tuple[str, str]]) -> str:
    return f"{describe_passages(passages)}\n\n<question>{question}</question>\n<reference_answer>{answer}</reference_answer>"


def unanswerable_message(coverage: str, count: int) -> str:
    return (
        f"<collection_coverage>\n{coverage}\n</collection_coverage>\n\nWrite {count} unanswerable questions."
    )


def answerability_message(question: str, passages: Sequence[tuple[str, str]]) -> str:
    return f"{describe_passages(passages)}\n\n<question>{question}</question>"
