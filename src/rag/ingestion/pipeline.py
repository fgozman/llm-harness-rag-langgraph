"""Incremental ingestion: discover -> plan -> parse -> chunk -> contextualize -> embed -> index -> record."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from rag import telemetry
from rag.config import ChunkingSettings, IngestionSettings
from rag.ingestion.chunking import TokenCounter, chunk_document
from rag.ingestion.contextualize import Contextualizer
from rag.ingestion.dates import CONTENT_WINDOW_CHARS, DocumentDate, infer_document_date
from rag.ingestion.discovery import DiscoveredFile, discover
from rag.ingestion.manifest import Manifest
from rag.ingestion.models import Chunk, ParentSection, ParsedDocument
from rag.retrieval.types import ChunkIndex, DenseEmbedder, IndexPoint, SparseEmbedder

log = logging.getLogger(__name__)


class Parser(Protocol):
    def parse(self, path: Path) -> ParsedDocument: ...


@dataclass(slots=True)
class IngestStats:
    discovered: int = 0
    indexed: int = 0
    unchanged: int = 0
    skipped_failed: int = 0
    failed: int = 0
    removed: int = 0
    chunks_written: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cache_read_tokens: int = 0
    llm_cache_write_tokens: int = 0
    contextualize_failures: int = 0
    llm_cost_usd: float = 0.0
    seconds: float = 0.0
    failures: dict[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True, slots=True)
class _Prepared:
    file: DiscoveredFile
    doc: ParsedDocument
    parents: list[ParentSection]
    chunks: list[Chunk]
    doc_date: DocumentDate


def _opening_text(doc: ParsedDocument) -> str:
    parts: list[str] = []
    size = 0
    for block in doc.blocks:
        if size >= CONTENT_WINDOW_CHARS:
            break
        parts.append(block.text)
        size += len(block.text) + 1
    return "\n".join(parts)


ProgressCallback = Callable[[str, int], None]


class IngestionPipeline:
    def __init__(
        self,
        *,
        manifest: Manifest,
        parser: Parser,
        dense: DenseEmbedder,
        sparse: SparseEmbedder,
        index: ChunkIndex,
        count_tokens: TokenCounter,
        chunking: ChunkingSettings,
        ingestion: IngestionSettings,
        contextualizer: Contextualizer | None = None,
    ) -> None:
        self._manifest = manifest
        self._parser = parser
        self._dense = dense
        self._sparse = sparse
        self._index = index
        self._count_tokens = count_tokens
        self._chunking = chunking
        self._ingestion = ingestion
        self._contextualizer = contextualizer if ingestion.contextualize else None

    def run(
        self,
        root: Path,
        *,
        retry_failed: bool = False,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IngestStats:
        with telemetry.span("ingest", "CHAIN", {"input.value": str(root)}) as current:
            stats = self._run(root, retry_failed=retry_failed, force=force, progress=progress)
            for key in ("discovered", "indexed", "unchanged", "failed", "removed", "chunks_written"):
                current.set_attribute(f"rag.ingest.{key}", getattr(stats, key))
            current.set_attribute("rag.ingest.llm_cost_usd", stats.llm_cost_usd)
            return stats

    def _run(
        self, root: Path, *, retry_failed: bool, force: bool, progress: ProgressCallback | None
    ) -> IngestStats:
        started = time.monotonic()
        stats = IngestStats()
        files = discover(root)
        if not self._index.exists():
            # The manifest says what's indexed, but the collection is gone (dropped, volume wiped): trust
            # the index, not the manifest, and rebuild.
            if not force:
                log.info("collection missing; indexing all files")
            force = True
        plan = self._manifest.plan(files, root=root, retry_failed=retry_failed, force=force)
        stats.discovered = len(files)
        stats.unchanged = len(plan.unchanged)
        stats.skipped_failed = len(plan.skipped_failed)

        for record in plan.removed:
            self._index.delete_doc(record.doc_id)
            self._manifest.remove(record.path)
            stats.removed += 1

        if plan.to_index:
            self._index.ensure_collection(self._dense.dimension)
        window = self._ingestion.window_files
        for start in range(0, len(plan.to_index), window):
            self._process_window(plan.to_index[start : start + window], stats, progress)

        stats.seconds = time.monotonic() - started
        return stats

    def _process_window(
        self, files: Sequence[DiscoveredFile], stats: IngestStats, progress: ProgressCallback | None
    ) -> None:
        prepared: list[_Prepared] = []
        for file in files:
            item = self._prepare(file, stats)
            if item is not None:
                prepared.append(item)
            if progress:
                progress("parsed", 1)
        if not prepared:
            return
        if self._contextualizer is not None:
            prepared = self._contextualize(prepared, stats)
        for item in prepared:
            self._write(item)
            stats.indexed += 1
            stats.chunks_written += len(item.chunks)
            if progress:
                progress("indexed", 1)

    def _prepare(self, file: DiscoveredFile, stats: IngestStats) -> _Prepared | None:
        # Parse/chunk errors are per-file data problems: record them and keep going. Infrastructure errors
        # (embedder, Qdrant) are raised from _write and abort the run instead.
        try:
            with telemetry.span("parse", "CHAIN", {"rag.file.path": str(file.path)}) as current:
                doc = self._parser.parse(file.path)
                current.set_attribute("rag.parser", doc.parser)
                current.set_attribute("rag.blocks", len(doc.blocks))
            with telemetry.span("chunk", "CHAIN", {"rag.file.path": str(file.path)}) as current:
                parents, chunks = chunk_document(doc, file.doc_id, self._count_tokens, self._chunking)
                current.set_attribute("rag.chunks", len(chunks))
                current.set_attribute("rag.parents", len(parents))
        except Exception as exc:
            log.warning("failed to parse %s: %s", file.path, exc)
            error = f"{type(exc).__name__}: {exc}"
            self._index.delete_doc(file.doc_id)
            self._manifest.record_failed(file, error=error)
            stats.failed += 1
            stats.failures[str(file.path)] = error
            return None
        doc_date = infer_document_date(file.path.name, _opening_text(doc), file.mtime)
        return _Prepared(file, doc, parents, chunks, doc_date)

    def _contextualize(self, prepared: list[_Prepared], stats: IngestStats) -> list[_Prepared]:
        assert self._contextualizer is not None
        with telemetry.span("contextualize", "CHAIN", {"rag.documents": len(prepared)}) as current:
            result = self._contextualizer.contextualize([(p.doc, p.chunks) for p in prepared])
            current.set_attribute("rag.contextualize.failed_chunks", result.failed_chunks)
            current.set_attribute("rag.contextualize.cost_usd", result.cost_usd)
            current.set_attribute("llm.token_count.prompt", result.input_tokens)
            current.set_attribute("llm.token_count.completion", result.output_tokens)
        stats.llm_input_tokens += result.input_tokens
        stats.llm_output_tokens += result.output_tokens
        stats.llm_cache_read_tokens += result.cache_read_tokens
        stats.llm_cache_write_tokens += result.cache_write_tokens
        stats.contextualize_failures += result.failed_chunks
        stats.llm_cost_usd += result.cost_usd
        return [
            _Prepared(p.file, p.doc, p.parents, chunks, p.doc_date)
            for p, chunks in zip(prepared, result.chunks, strict=True)
        ]

    def _write(self, item: _Prepared) -> None:
        path = {"rag.file.path": str(item.file.path)}
        with telemetry.span("embed", "EMBEDDING", path) as current:
            points = self._embed(item)
            current.set_attribute("rag.chunks", len(points))
        with telemetry.span("index", "CHAIN", path) as current:
            # Delete-then-upsert keyed by doc_id makes re-runs idempotent even after a crash mid-file.
            self._index.delete_doc(item.file.doc_id)
            if points:
                self._index.upsert(points)
            current.set_attribute("rag.points", len(points))
        self._manifest.record_indexed(
            item.file,
            parser=item.doc.parser,
            parents=item.parents,
            chunk_ids=[c.chunk_id for c in item.chunks],
        )

    def _embed(self, item: _Prepared) -> list[IndexPoint]:
        if not item.chunks:
            return []
        texts = [c.embedding_text(item.doc.title) for c in item.chunks]
        dense = self._dense.embed_documents(texts)
        sparse = self._sparse.embed_documents(texts)
        ingested_at = datetime.now(UTC).isoformat()
        return [
            IndexPoint(id=c.chunk_id, dense=d, sparse=s, payload=self._payload(item, c, ingested_at))
            for c, d, s in zip(item.chunks, dense, sparse, strict=True)
        ]

    def _payload(self, item: _Prepared, chunk: Chunk, ingested_at: str) -> dict[str, object]:
        path = item.file.path
        return {
            "doc_id": item.file.doc_id,
            "file_path": str(path),
            "file_name": path.name,
            "folder": str(path.parent),
            "file_type": path.suffix.lower().lstrip("."),
            "title": item.doc.title,
            "section_path": list(chunk.section_path),
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "chunk_index": chunk.chunk_index,
            "parent_id": chunk.parent_id,
            "content_hash": item.file.content_hash,
            "file_mtime": item.file.mtime,
            # Issue date for date filters: from the file name, else the opening text, else mtime.
            "doc_date": item.doc_date.value.isoformat(),
            "doc_date_source": item.doc_date.source,
            "doc_date_ts": item.doc_date.timestamp,
            "ingested_at": ingested_at,
            "language": self._ingestion.language,
            "parser": item.doc.parser,
            "token_count": chunk.token_count,
            "text": chunk.text,
            "context": chunk.context,
        }
