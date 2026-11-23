"""Background ingestion jobs: one at a time (they share the manifest, the index and the embedder)."""

from __future__ import annotations

import contextvars
import dataclasses
import logging
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from rag.api.schemas import IngestJob
from rag.config import Settings
from rag.ingestion.pipeline import IngestStats, ProgressCallback

log = logging.getLogger(__name__)


class Pipeline(Protocol):
    def run(
        self,
        root: Path,
        *,
        retry_failed: bool = False,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IngestStats: ...


PipelineFactory = Callable[[Settings], Pipeline]


class JobConflictError(RuntimeError):
    """Another ingestion job is still running."""


class IngestJobs:
    def __init__(self, settings: Settings, build_pipeline: PipelineFactory) -> None:
        self._settings = settings
        self._build = build_pipeline
        self._lock = threading.Lock()
        self._jobs: dict[str, IngestJob] = {}
        self._running: str | None = None

    @property
    def running(self) -> bool:
        return self._running is not None

    def get(self, job_id: str) -> IngestJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def submit(self, root: Path, *, force: bool, retry_failed: bool, contextualize: bool | None) -> IngestJob:
        settings = self._settings.model_copy(deep=True)  # per-job overrides never leak into the server
        if contextualize is not None:
            settings.ingestion.contextualize = contextualize
        with self._lock:
            if self._running is not None:
                raise JobConflictError(self._running)
            job = IngestJob(
                id=uuid.uuid4().hex,
                status="running",
                path=str(root),
                force=force,
                retry_failed=retry_failed,
                contextualize=settings.ingestion.contextualize,
                started_at=datetime.now(UTC),
            )
            self._jobs[job.id] = job
            self._running = job.id
        context = contextvars.copy_context()  # keeps the request id on the job's log lines
        threading.Thread(
            target=context.run,
            args=(self._run, job.id, root, settings),
            name=f"ingest-{job.id[:8]}",
            daemon=True,
        ).start()
        return job.model_copy(deep=True)

    def _progress(self, job_id: str) -> ProgressCallback:
        def update(stage: str, n: int) -> None:
            with self._lock:
                progress = self._jobs[job_id].progress
                progress[stage] = progress.get(stage, 0) + n

        return update

    def _run(self, job_id: str, root: Path, settings: Settings) -> None:
        job = self._jobs[job_id]
        log.info("ingest started", extra={"job_id": job_id, "path": str(root)})
        try:
            pipeline = self._build(settings)
            stats = pipeline.run(
                root, force=job.force, retry_failed=job.retry_failed, progress=self._progress(job_id)
            )
        except Exception as exc:
            # Infra errors (Qdrant down, model load, config) end the job; per-file parse errors are in stats.
            log.exception("ingest failed", extra={"job_id": job_id})
            self._finish(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            return
        log.info(
            "ingest finished",
            extra={
                "job_id": job_id,
                "indexed": stats.indexed,
                "failed": stats.failed,
                "seconds": stats.seconds,
            },
        )
        self._finish(job_id, status="succeeded", stats=dataclasses.asdict(stats))

    def _finish(
        self, job_id: str, *, status: str, stats: dict[str, object] | None = None, error: str | None = None
    ) -> None:
        with self._lock:
            self._jobs[job_id] = self._jobs[job_id].model_copy(
                update={"status": status, "stats": stats, "error": error, "finished_at": datetime.now(UTC)}
            )
            self._running = None
