# sentence-transformers' overloads contain Unknown parts; relax only the "unknown" checks in this adapter.
# pyright: reportUnknownMemberType=false
"""Dense (Qwen3-Embedding via sentence-transformers) and sparse (BM25 via FastEmbed) embedders.

Models are loaded once per process (see `rag.factory`) and run natively on cuda/mps/cpu.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

from rag import devices  # sets PYTORCH_ENABLE_MPS_FALLBACK before torch initializes
from rag.config import Device, DType
from rag.retrieval.types import SparseVec

log = logging.getLogger(__name__)


def qwen3_query_prompt(instruction: str) -> str:
    # Exact Qwen3-Embedding model-card format (note: no space after "Query:"). Documents get no prompt.
    return f"Instruct: {instruction}\nQuery:"


class QwenDenseEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        device: Device,
        dtype: DType,
        batch_size: int,
        max_length: int,
        query_instruction: str,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = devices.resolve_device(device)
        torch_dtype = devices.resolve_dtype(self.device, dtype)
        log.info("loading %s on %s (%s)", model_name, self.device, torch_dtype)
        self._model = SentenceTransformer(
            model_name,
            device=self.device,
            model_kwargs={"dtype": torch_dtype, "attn_implementation": devices.ATTN_IMPLEMENTATION},
            # Left padding matches the model's last-token pooling (per model card).
            processor_kwargs={"padding_side": "left"},
        )
        self._model.max_seq_length = max_length
        self._batch_size = batch_size
        self._query_prompt = qwen3_query_prompt(query_instruction)
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError("embedding model did not report a dimension")
        self._dimension = dim
        # One model instance per process is shared by API requests and ingest jobs: serialise device access.
        self._lock = threading.Lock()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            vectors = self._model.encode(
                list(texts), batch_size=self._batch_size, normalize_embeddings=True, convert_to_numpy=True
            )
        return vectors.astype("float32").tolist()

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            vector = self._model.encode(
                [text], prompt=self._query_prompt, normalize_embeddings=True, convert_to_numpy=True
            )
        return vector[0].astype("float32").tolist()


class Bm25SparseEmbedder:
    """BM25 term weights; IDF is applied server-side by Qdrant (sparse vector modifier=IDF)."""

    def __init__(self, model_name: str) -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVec]:
        return [
            SparseVec(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self._model.embed(list(texts))
        ]

    def embed_query(self, text: str) -> SparseVec:
        (e,) = list(self._model.query_embed(text))
        return SparseVec(indices=e.indices.tolist(), values=e.values.tolist())


class HFTokenCounter:
    """Counts tokens with the embedding model's tokenizer so chunk budgets match what the embedder sees."""

    def __init__(self, model_name: str) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_pretrained(model_name)

    def __call__(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
