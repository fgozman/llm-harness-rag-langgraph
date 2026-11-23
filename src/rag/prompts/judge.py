"""LLM-as-judge prompts for evaluation (correctness, faithfulness, relevance, citation support) and the
closed-book baseline."""

from __future__ import annotations

from collections.abc import Sequence

from rag.prompts.common import UNTRUSTED_DOCUMENTS

VERSION = "judge-v2"

CORRECTNESS = (
    "You grade an answer against a reference answer for the same question. correct: contains the key facts of "
    "the reference (numbers must match, allowing equivalent units/rounding such as $14,766 million = $14.8 billion) "
    "and nothing contradicting it; extra correct detail is fine. partially_correct: some key facts right but "
    "others missing or wrong. incorrect: wrong, contradicting, or a refusal/'not found' when the reference has "
    "an answer."
)

FAITHFULNESS = (
    "You check whether every factual claim in an answer is supported by the passages it was written from. List "
    "supported and unsupported claims (ignore citation formatting and statements that information is missing). "
    + UNTRUSTED_DOCUMENTS
)

RELEVANCE = (
    "Decide whether the answer responds to what the question asks (even if it says the information is not "
    "available). An answer about a different entity, period or topic is not relevant."
)

CITATIONS = (
    "For each citation marker [n] in the answer, decide whether passage n supports the sentence(s) that cite it. "
    "Return one entry per distinct cited passage number. " + UNTRUSTED_DOCUMENTS
)

CLOSED_BOOK = (
    "Answer the question from your own knowledge only (no documents are available). Always give your best "
    "guess, even if unsure. Set confident to true only if you are confident the answer is correct; set it to "
    "false if you are guessing or the question concerns events after your training data."
)


def correctness_message(question: str, reference: str, answer: str) -> str:
    return f"<question>{question}</question>\n<reference_answer>{reference}</reference_answer>\n<answer>{answer}</answer>"


def passages_block(passages: Sequence[tuple[int, str]]) -> str:
    return (
        "<documents>\n"
        + "\n".join(f'<passage index="{i}">\n{t}\n</passage>' for i, t in passages)
        + "\n</documents>"
    )


def faithfulness_message(passages: Sequence[tuple[int, str]], answer: str) -> str:
    return f"{passages_block(passages)}\n\n<answer>\n{answer}\n</answer>"


def relevance_message(question: str, answer: str) -> str:
    return f"<question>{question}</question>\n<answer>\n{answer}\n</answer>"


def citations_message(passages: Sequence[tuple[int, str]], answer: str) -> str:
    return faithfulness_message(passages, answer)
