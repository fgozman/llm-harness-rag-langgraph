"""Exceptions that mean "the LLM provider failed" (after SDK retries), as opposed to a bug in our code."""

from __future__ import annotations

import anthropic
import openai

from rag.llm.base import LLMError

PROVIDER_ERRORS: tuple[type[Exception], ...] = (LLMError, anthropic.APIError, openai.APIError)
