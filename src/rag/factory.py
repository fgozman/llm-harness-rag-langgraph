"""Builds runtime components from settings. Heavy models are created once per process and reused."""

from __future__ import annotations

from functools import cache

import anthropic
from qdrant_client import QdrantClient

from rag.config import Device, DType, Settings
from rag.embeddings import Bm25SparseEmbedder, HFTokenCounter, QwenDenseEmbedder
from rag.ingestion.contextualize import AnthropicBackend, LLMContextualizer
from rag.ingestion.manifest import Manifest
from rag.ingestion.parsing import DocumentParser
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.qdrant_index import QdrantChunkIndex


class ConfigurationError(RuntimeError):
    """Raised when settings are inconsistent with the requested operation."""


@cache
def dense_embedder(
    model: str, device: Device, dtype: DType, batch_size: int, max_length: int, query_instruction: str
) -> QwenDenseEmbedder:
    return QwenDenseEmbedder(
        model,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        max_length=max_length,
        query_instruction=query_instruction,
    )


@cache
def sparse_embedder(model: str) -> Bm25SparseEmbedder:
    return Bm25SparseEmbedder(model)


@cache
def token_counter(model: str) -> HFTokenCounter:
    return HFTokenCounter(model)


def build_dense(settings: Settings) -> QwenDenseEmbedder:
    e = settings.embedding
    # Filled from the hardware profile by Settings validation.
    assert e.model is not None and e.batch_size is not None and settings.dtype is not None
    return dense_embedder(
        e.model, settings.device, settings.dtype, e.batch_size, e.max_length, e.query_instruction
    )


def build_index(settings: Settings) -> QdrantChunkIndex:
    q = settings.qdrant
    api_key = q.api_key.get_secret_value() if q.api_key else None
    return QdrantChunkIndex(QdrantClient(url=q.url, api_key=api_key, timeout=60), q.collection)


def build_manifest(settings: Settings) -> Manifest:
    return Manifest(settings.state_dir / "rag.db")


def build_contextualizer(settings: Settings) -> LLMContextualizer:
    if settings.llm.provider != "anthropic":
        raise ConfigurationError(
            "Contextual enrichment uses Anthropic prompt caching + Message Batches and currently requires "
            "LLM__PROVIDER=anthropic. Disable it with INGESTION__CONTEXTUALIZE=false or `--no-contextualize`."
        )
    if settings.anthropic_api_key is None:
        raise ConfigurationError("ANTHROPIC_API_KEY is not set (needed for contextual enrichment).")
    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key.get_secret_value(),
        max_retries=settings.llm.max_retries,  # SDK retries 429/5xx with exponential backoff
        timeout=settings.llm.request_timeout_s,
    )
    assert settings.embedding.model is not None
    return LLMContextualizer(
        AnthropicBackend(client, poll_interval_s=settings.ingestion.batch_poll_interval_s),
        model=settings.llm.cheap_model,
        settings=settings.ingestion,
        count_tokens=token_counter(settings.embedding.model),
        pricing=settings.llm.pricing.get(settings.llm.cheap_model),
    )


def build_ingestion_pipeline(settings: Settings) -> IngestionPipeline:
    assert settings.embedding.model is not None
    contextualizer = build_contextualizer(settings) if settings.ingestion.contextualize else None
    return IngestionPipeline(
        manifest=build_manifest(settings),
        parser=DocumentParser(pdf_ocr=settings.ingestion.pdf_ocr),
        dense=build_dense(settings),
        sparse=sparse_embedder(settings.qdrant.sparse_model),
        index=build_index(settings),
        count_tokens=token_counter(settings.embedding.model),
        chunking=settings.chunking,
        ingestion=settings.ingestion,
        contextualizer=contextualizer,
    )
