"""Builds runtime components from settings. Heavy models are created once per process and reused."""

from __future__ import annotations

from functools import cache

import anthropic
import openai
from qdrant_client import QdrantClient

from rag.config import Device, DType, Settings
from rag.embeddings import Bm25SparseEmbedder, HFTokenCounter, QwenDenseEmbedder
from rag.ingestion.context_backends import AnthropicBackend, OpenAIResponsesBackend
from rag.ingestion.contextualize import ContextBackend, LLMContextualizer
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


def _anthropic_context_backend(settings: Settings, model: str) -> AnthropicBackend:
    if settings.anthropic_api_key is None:
        raise ConfigurationError("ANTHROPIC_API_KEY is not set (needed for contextual enrichment).")
    effort = settings.ingestion.contextualize_effort or "low"
    if effort in ("none", "minimal"):
        raise ConfigurationError(f"Anthropic does not accept effort {effort!r}; use low, medium or high.")
    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key.get_secret_value(),
        max_retries=settings.llm.max_retries,  # SDK retries 429/5xx with exponential backoff
        timeout=settings.llm.request_timeout_s,
    )
    return AnthropicBackend(
        client,
        model=model,
        effort=effort,
        max_tokens=settings.ingestion.contextualize_max_output_tokens,
        poll_interval_s=settings.ingestion.batch_poll_interval_s,
    )


def _openai_context_backend(settings: Settings, model: str) -> OpenAIResponsesBackend:
    if settings.openai_api_key is None:
        raise ConfigurationError("OPENAI_API_KEY is not set (needed for contextual enrichment with OpenAI).")
    client = openai.OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.llm.openai_base_url,
        max_retries=settings.llm.max_retries,
        timeout=settings.llm.request_timeout_s,
    )
    return OpenAIResponsesBackend(
        client,
        model=model,
        max_tokens=settings.ingestion.contextualize_max_output_tokens,
        reasoning_effort=settings.ingestion.contextualize_effort or "none",
        openai_platform=True,
    )


def _ollama_context_backend(settings: Settings, model: str) -> OpenAIResponsesBackend:
    client = openai.OpenAI(
        base_url=f"{settings.llm.ollama_base_url.rstrip('/')}/v1/",
        api_key="ollama",  # required by the SDK, ignored by Ollama
        max_retries=settings.llm.max_retries,
        timeout=settings.llm.request_timeout_s,
    )
    effort = settings.ingestion.contextualize_effort
    return OpenAIResponsesBackend(
        client,
        model=model,
        max_tokens=settings.ingestion.contextualize_max_output_tokens,
        reasoning_effort=effort,
        # Thinking models (e.g. qwen3) would spend the small output budget on reasoning; Ollama's `think`
        # extension turns it off unless the user explicitly asked for an effort level.
        extra_body=None if effort else {"think": False},
    )


def build_contextualizer(settings: Settings) -> LLMContextualizer:
    llm = settings.llm
    assert llm.cheap_model is not None and settings.embedding.model is not None  # filled by validators
    backend: ContextBackend
    if llm.provider == "anthropic":
        backend = _anthropic_context_backend(settings, llm.cheap_model)
    elif llm.provider == "openai":
        backend = _openai_context_backend(settings, llm.cheap_model)
    else:
        backend = _ollama_context_backend(settings, llm.cheap_model)
    if settings.ingestion.contextualize_mode == "batch" and not backend.supports_batch:
        raise ConfigurationError(
            f"contextualize_mode=batch is only supported with Anthropic; {llm.provider} runs synchronously. "
            "Use contextualize_mode=auto or sync."
        )
    return LLMContextualizer(
        backend,
        settings=settings.ingestion,
        count_tokens=token_counter(settings.embedding.model),
        pricing=llm.pricing.get(llm.cheap_model),
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
