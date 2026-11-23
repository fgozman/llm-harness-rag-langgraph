"""LLM protocol used by graph nodes, plus the JSON-schema helper shared by all providers."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel

from rag.config import LLMTask

TokenCallback = Callable[[str], None]

# JSON-schema keywords Anthropic structured outputs reject; descriptions carry the intent instead.
_UNSUPPORTED = frozenset(
    {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength",
     "maxItems", "uniqueItems", "default"}
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


class LLMError(RuntimeError):
    """The model refused, ran out of tokens, or returned output that doesn't match the schema."""


class LLM(Protocol):
    def structured[T: BaseModel](
        self, task: LLMTask, *, system: str, messages: Sequence[Message], schema: type[T]
    ) -> T: ...

    def generate(
        self,
        task: LLMTask,
        *,
        system: str,
        messages: Sequence[Message],
        on_token: TokenCallback | None = None,
    ) -> str: ...


def _strictify(node: object) -> None:
    if isinstance(node, dict):
        d = cast(dict[str, Any], node)
        for key in _UNSUPPORTED & d.keys():
            del d[key]
        if d.get("type") == "object" and "properties" in d:
            # OpenAI strict mode requires every property; optional values are expressed as nullable.
            d["required"] = list(d["properties"])
            d["additionalProperties"] = False
        for value in d.values():
            _strictify(value)
    elif isinstance(node, list):
        for item in cast(list[object], node):
            _strictify(item)


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Schema accepted by Anthropic, OpenAI (strict) and Ollama: all properties required, no extra keys."""
    schema = copy.deepcopy(model.model_json_schema())
    _strictify(schema)
    return schema


def parse_structured[T: BaseModel](text: str, schema: type[T]) -> T:
    try:
        return schema.model_validate_json(text)
    except ValueError as exc:
        raise LLMError(f"structured output did not match {schema.__name__}: {exc}") from exc
