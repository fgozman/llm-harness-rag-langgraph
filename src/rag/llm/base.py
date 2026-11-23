"""LLM protocol used by graph nodes, plus the JSON-schema helper shared by all providers."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel

from rag.config import LLMTask

TokenCallback = Callable[[str], None]

# JSON-schema keywords some providers' structured outputs reject; descriptions carry the intent instead.
_UNSUPPORTED = frozenset(
    {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength",
     "maxItems", "uniqueItems", "default"}
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


class LLMError(RuntimeError):
    """The provider call failed (after the integration's retries), the model refused or ran out of tokens, or
    its output doesn't match the schema.

    `status_code` is the provider's HTTP status when the failure came from the API, else None. `transient`
    marks failures a later attempt may not hit (rate limits, server errors, timeouts), as opposed to
    configuration/request problems every further call would hit too.
    """

    def __init__(self, message: str, *, status_code: int | None = None, transient: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.transient = transient


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
            # Strict modes require every property; optional values are expressed as nullable.
            d["required"] = list(d["properties"])
            d["additionalProperties"] = False
        for value in d.values():
            _strictify(value)
    elif isinstance(node, list):
        for item in cast(list[object], node):
            _strictify(item)


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Lowest-common-denominator schema for native structured outputs: all properties required, no extra
    keys, no numeric/length constraints."""
    schema = copy.deepcopy(model.model_json_schema())
    _strictify(schema)
    return schema
