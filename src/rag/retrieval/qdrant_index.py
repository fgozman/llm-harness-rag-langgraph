"""Qdrant collection holding child chunks with a dense vector and a BM25 sparse vector per point."""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from rag.retrieval.types import IndexPoint

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
_UPSERT_BATCH = 256

# Payload fields used in filters get indexes so filtering stays fast at ~50k pages.
_PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "file_name": models.PayloadSchemaType.KEYWORD,
    "folder": models.PayloadSchemaType.KEYWORD,
    "file_type": models.PayloadSchemaType.KEYWORD,
    "file_mtime": models.PayloadSchemaType.FLOAT,
}


def _doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])


class QdrantChunkIndex:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def collection(self) -> str:
        return self._collection

    def exists(self) -> bool:
        return self._client.collection_exists(self._collection)

    def ensure_collection(self, dense_dim: int) -> None:
        if self.exists():
            self._check_dimension(dense_dim)
            return
        self._client.create_collection(
            self._collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
            },
            # BM25 term frequencies come from FastEmbed; Qdrant applies IDF over the live collection.
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field, schema in _PAYLOAD_INDEXES.items():
            self._client.create_payload_index(self._collection, field, schema)

    def _check_dimension(self, dense_dim: int) -> None:
        vectors = self._client.get_collection(self._collection).config.params.vectors
        size = vectors[DENSE_VECTOR].size if isinstance(vectors, dict) else None
        if size != dense_dim:
            raise RuntimeError(
                f"collection {self._collection!r} has dense size {size}, embedder produces {dense_dim}. "
                "The embedding model changed: use a new collection name, or delete the collection and the "
                "manifest (state_dir/rag.db) and re-ingest."
            )

    def upsert(self, points: Sequence[IndexPoint]) -> None:
        for start in range(0, len(points), _UPSERT_BATCH):
            batch = points[start : start + _UPSERT_BATCH]
            self._client.upsert(
                self._collection,
                points=[
                    models.PointStruct(
                        id=p.id,
                        vector={
                            DENSE_VECTOR: p.dense,
                            SPARSE_VECTOR: models.SparseVector(
                                indices=p.sparse.indices, values=p.sparse.values
                            ),
                        },
                        payload=p.payload,
                    )
                    for p in batch
                ],
                wait=True,
            )

    def delete_doc(self, doc_id: str) -> None:
        if not self.exists():
            return
        self._client.delete(
            self._collection, points_selector=models.FilterSelector(filter=_doc_filter(doc_id)), wait=True
        )

    def count(self, doc_id: str | None = None) -> int:
        if not self.exists():
            return 0
        flt = _doc_filter(doc_id) if doc_id else None
        return self._client.count(self._collection, count_filter=flt, exact=True).count
