"""Request/response models of the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.graph.state import AnswerStatus, Citation

# Thread ids become checkpoint keys and log fields: keep them to a safe, bounded alphabet.
THREAD_ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,128}$"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="The user's question (or follow-up).")
    thread_id: str | None = Field(
        default=None,
        pattern=THREAD_ID_PATTERN,
        description="Continue a conversation; omitted = a new thread (its id is returned).",
    )


class FinalEvent(BaseModel):
    """Payload of the last SSE event (`event: final`) of a successful turn."""

    thread_id: str
    status: AnswerStatus
    text: str
    citations: list[Citation]


class ErrorEvent(BaseModel):
    """Payload of `event: error`; ends the stream."""

    type: Literal["timeout", "llm_error", "internal_error"]
    message: str
    thread_id: str


class IngestRequest(BaseModel):
    path: str | None = Field(
        default=None,
        description="Folder or file to index; relative paths are under docs_dir; omitted = docs_dir.",
    )
    force: bool = False
    retry_failed: bool = False
    contextualize: bool | None = Field(default=None, description="Override ingestion.contextualize.")


JobStatus = Literal["running", "succeeded", "failed"]


class IngestJob(BaseModel):
    id: str
    status: JobStatus
    path: str
    force: bool
    retry_failed: bool
    contextualize: bool
    started_at: datetime
    finished_at: datetime | None = None
    progress: dict[str, int] = Field(default_factory=dict[str, int])
    stats: dict[str, Any] | None = None
    error: str | None = None


class ServiceStatus(BaseModel):
    url: str
    status: str
    required: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    services: dict[str, ServiceStatus]
    active_turns: int
    ingest_running: bool
