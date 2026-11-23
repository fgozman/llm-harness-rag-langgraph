"""Builds runtime components from settings. Heavy models are created once per process and reused."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import cache
from typing import Any, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import AliasChoices
from qdrant_client import QdrantClient

from rag.config import Device, DType, LLMSettings, Settings
from rag.embeddings import Bm25SparseEmbedder, HFTokenCounter, QwenDenseEmbedder
from rag.graph.build import GraphDeps, build_checkpointer, build_graph
from rag.ingestion.context_backends import ChatModelContextBackend
from rag.ingestion.contextualize import LLMContextualizer
from rag.ingestion.manifest import Manifest
from rag.ingestion.parsing import DocumentParser
from rag.ingestion.pipeline import IngestionPipeline
from rag.llm.base import LLM
from rag.llm.chat_model import ChatModelLLM, TaskModel
from rag.retrieval.qdrant_index import QdrantChunkIndex
from rag.retrieval.qdrant_search import QdrantHybridSearcher
from rag.retrieval.rerank import CrossEncoderReranker, Qwen3Reranker, is_qwen3_reranker
from rag.retrieval.retriever import Reranker, Retriever

log = logging.getLogger(__name__)


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


@cache
def reranker_model(
    model: str, device: Device, dtype: DType, batch_size: int, max_length: int, instruction: str
) -> Reranker:
    if is_qwen3_reranker(model):
        return Qwen3Reranker(
            model,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            max_length=max_length,
            instruction=instruction,
        )
    return CrossEncoderReranker(
        model, device=device, dtype=dtype, batch_size=batch_size, max_length=max_length
    )


def build_reranker(settings: Settings) -> Reranker | None:
    r = settings.reranker
    if not r.enabled:
        return None
    assert r.model is not None and settings.dtype is not None  # filled from the hardware profile
    try:
        return reranker_model(
            r.model, settings.device, settings.dtype, r.batch_size, r.max_length, r.instruction
        )
    except (OSError, RuntimeError, ValueError) as exc:
        # Spec: bge-reranker-v2-m3 is the fallback when the Qwen3 reranker can't be loaded on this machine.
        log.warning("reranker %s failed to load (%s); falling back to %s", r.model, exc, r.fallback_model)
        return reranker_model(
            r.fallback_model, settings.device, settings.dtype, r.batch_size, r.max_length, r.instruction
        )


def build_retriever(settings: Settings) -> Retriever:
    assert settings.embedding.model is not None
    manifest = build_manifest(settings)
    index = build_index(settings)
    searcher = QdrantHybridSearcher(
        index.client,
        index.collection,
        dense=build_dense(settings),
        sparse=sparse_embedder(settings.qdrant.sparse_model),
        folders_under=manifest.folders_under,
    )
    return Retriever(
        searcher=searcher,
        reranker=build_reranker(settings),
        parents=manifest,
        count_tokens=token_counter(settings.embedding.model),
        retrieval=settings.retrieval,
        reranker_settings=settings.reranker,
    )


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


def _accepted_params(chat_class: type[BaseChatModel]) -> set[str]:
    names: set[str] = set()
    for name, field in chat_class.model_fields.items():
        names.add(name)
        if field.alias:
            names.add(field.alias)
        aliases = field.validation_alias
        if isinstance(aliases, str):
            names.add(aliases)
        elif isinstance(aliases, AliasChoices):
            names.update(a for a in aliases.choices if isinstance(a, str))
    return names


def build_chat_model(
    llm: LLMSettings, model: str, *, max_tokens: int, params: Mapping[str, Any]
) -> BaseChatModel:
    """A LangChain chat model for `llm.provider` (any provider `init_chat_model` knows)."""
    assert llm.max_tokens_param is not None  # filled from the provider preset
    kwargs: dict[str, Any] = {**llm.params, **params, llm.max_tokens_param: max_tokens}
    if llm.base_url:
        kwargs["base_url"] = llm.base_url
    try:
        # With a fixed model (no configurable fields) init_chat_model returns the provider's chat model.
        chat = cast(BaseChatModel, init_chat_model(model, model_provider=llm.provider, **kwargs))
    except ImportError as exc:
        raise ConfigurationError(
            f"LLM provider {llm.provider!r} needs its LangChain integration package ({exc}). "
            "Install it, e.g. `uv sync --extra anthropic|openai|ollama` or `uv add langchain-<provider>`."
        ) from exc
    except ValueError as exc:  # unknown provider, invalid option values
        raise ConfigurationError(f"cannot configure {llm.provider} model {model!r}: {exc}") from exc
    # Integrations ignore (or forward to the API) options they don't define: a typo or another provider's
    # option name must fail here, not silently change behaviour.
    unknown = sorted(set(kwargs) - _accepted_params(type(chat)))
    if unknown:
        raise ConfigurationError(
            f"{type(chat).__name__} does not accept {', '.join(unknown)} (check llm.params, "
            "llm.tasks.<task>.params, ingestion.contextualize_params, llm.max_tokens_param)"
        )
    return chat


def build_contextualizer(settings: Settings) -> LLMContextualizer:
    llm, ingestion = settings.llm, settings.ingestion
    # Filled by validators.
    assert llm.cheap_model is not None and settings.embedding.model is not None
    assert ingestion.contextualize_params is not None and llm.prompt_cache_marker is not None
    chat = build_chat_model(
        llm,
        llm.cheap_model,
        max_tokens=ingestion.contextualize_max_output_tokens,
        params=ingestion.contextualize_params,
    )
    backend = ChatModelContextBackend(
        chat, max_tokens=ingestion.contextualize_max_output_tokens, cache_marker=llm.prompt_cache_marker
    )
    return LLMContextualizer(
        backend,
        settings=ingestion,
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


def build_llm(settings: Settings) -> LLM:
    llm = settings.llm
    assert llm.cheap_model and llm.generation_model and llm.upgrade_model  # filled from the provider preset
    names = {"cheap": llm.cheap_model, "generation": llm.generation_model, "upgrade": llm.upgrade_model}
    tasks: dict[str, TaskModel] = {}
    for task, cfg in llm.tasks.items():
        assert cfg.params is not None  # filled from the provider preset
        name = names[cfg.model]
        chat = build_chat_model(llm, name, max_tokens=cfg.max_tokens, params=cfg.params)
        tasks[task] = TaskModel(chat=chat, name=name, max_tokens=cfg.max_tokens)
    return ChatModelLLM(
        tasks,
        structured_method=llm.structured_output_method,
        structured_params=llm.structured_output_params,
    )


def build_qa_graph(settings: Settings) -> Any:
    """The compiled LangGraph pipeline with a SQLite checkpointer (conversation memory per thread_id)."""
    deps = GraphDeps(
        llm=build_llm(settings),
        retriever=build_retriever(settings),
        settings=settings.graph,
        docs_dir=settings.docs_dir,
    )
    return build_graph(deps, checkpointer=build_checkpointer(settings.state_dir / "checkpoints.db"))
