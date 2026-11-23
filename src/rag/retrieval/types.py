"""Types shared between indexing (write path) and retrieval (read path)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast


@dataclass(frozen=True, slots=True)
class SparseVec:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True, slots=True)
class IndexPoint:
    id: str
    dense: list[float]
    sparse: SparseVec
    payload: dict[str, object]


class DenseEmbedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class SparseEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[SparseVec]: ...


class ChunkIndex(Protocol):
    def exists(self) -> bool: ...

    def ensure_collection(self, dense_dim: int) -> None: ...

    def upsert(self, points: Sequence[IndexPoint]) -> None: ...

    def delete_doc(self, doc_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """One hit from a single ranking (one vector type for one query variant)."""

    chunk_id: str
    score: float
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class RankedChunk:
    """A candidate after fusion (and reranking, when enabled). `score` is the final ranking score."""

    chunk_id: str
    score: float
    fused_score: float
    payload: dict[str, object]
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class ContextPassage:
    """A numbered passage handed to the LLM; `index` is the citation number ([1], [2], ...)."""

    index: int
    parent_id: str
    chunk_ids: tuple[str, ...]
    text: str
    expanded: bool
    score: float
    file_path: str
    file_name: str
    title: str
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    snippet: str  # text of the best-ranked child chunk: what actually matched the query


def payload_section(payload: dict[str, object]) -> tuple[str, ...]:
    """Qdrant returns JSON lists as untyped values; normalize section_path to a tuple of strings."""
    value = payload.get("section_path")
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in cast(Sequence[object], value) if item)
