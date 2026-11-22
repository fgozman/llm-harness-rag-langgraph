"""Types shared between indexing (write path) and retrieval (read path)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


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
