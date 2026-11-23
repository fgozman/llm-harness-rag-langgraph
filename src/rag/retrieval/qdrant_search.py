"""Hybrid search against the chunk collection: one batched Qdrant round trip per call."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from qdrant_client import QdrantClient, models

from rag.retrieval.filters import SearchFilters, to_qdrant_filter
from rag.retrieval.qdrant_index import DENSE_VECTOR, SPARSE_VECTOR
from rag.retrieval.retriever import SearchMode
from rag.retrieval.types import ScoredChunk, SparseVec


class DenseQueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class SparseQueryEmbedder(Protocol):
    def embed_query(self, text: str) -> SparseVec: ...


class QdrantHybridSearcher:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        *,
        dense: DenseQueryEmbedder,
        sparse: SparseQueryEmbedder,
        folders_under: Callable[[Path], list[str]],
    ) -> None:
        self._client = client
        self._collection = collection
        self._dense = dense
        self._sparse = sparse
        self._folders_under = folders_under

    def search(
        self, queries: Sequence[str], *, mode: SearchMode, limit: int, filters: SearchFilters | None
    ) -> list[list[ScoredChunk]]:
        folders = self._folders_under(filters.folder) if filters and filters.folder else []
        qfilter = to_qdrant_filter(filters, folders=folders)
        requests: list[models.QueryRequest] = []
        for query in queries:
            if mode in ("hybrid", "dense"):
                requests.append(self._request(self._dense.embed_query(query), DENSE_VECTOR, limit, qfilter))
            if mode in ("hybrid", "sparse"):
                vec = self._sparse.embed_query(query)
                sparse = models.SparseVector(indices=vec.indices, values=vec.values)
                requests.append(self._request(sparse, SPARSE_VECTOR, limit, qfilter))
        responses = self._client.query_batch_points(self._collection, requests=requests)
        return [
            [ScoredChunk(str(p.id), p.score, dict(p.payload or {})) for p in response.points]
            for response in responses
        ]

    @staticmethod
    def _request(
        vector: list[float] | models.SparseVector, using: str, limit: int, qfilter: models.Filter | None
    ) -> models.QueryRequest:
        return models.QueryRequest(query=vector, using=using, limit=limit, filter=qfilter, with_payload=True)
