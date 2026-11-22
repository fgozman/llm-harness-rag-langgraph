"""SQLite manifest: which files are indexed (by content hash), their chunk IDs, and parent sections.

Parent sections live here too (a small local docstore) because they are only ever fetched by ID after
retrieval and don't need vectors.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from rag.ingestion.discovery import DiscoveredFile
from rag.ingestion.models import ParentSection

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    parser TEXT,
    chunk_ids TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS parents (
    parent_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    text TEXT NOT NULL,
    section_path TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER
);
CREATE INDEX IF NOT EXISTS parents_doc ON parents(doc_id);
"""


class FileStatus(StrEnum):
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FileRecord:
    path: Path
    doc_id: str
    content_hash: str
    status: FileStatus
    error: str | None
    parser: str | None
    chunk_ids: list[str]


@dataclass(frozen=True, slots=True)
class IngestPlan:
    to_index: list[DiscoveredFile]
    unchanged: list[DiscoveredFile]
    skipped_failed: list[DiscoveredFile]
    removed: list[FileRecord]


class Manifest:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection]:
        # Short-lived connections keep the manifest safe to use from worker threads.
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def get(self, path: Path) -> FileRecord | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM files WHERE path = ?", (str(path),)).fetchone()
        return _record(row) if row else None

    def _records_under(self, root: Path) -> list[FileRecord]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM files").fetchall()
        records = [_record(r) for r in rows]
        return [r for r in records if r.path == root or r.path.is_relative_to(root)]

    def plan(
        self,
        discovered: Sequence[DiscoveredFile],
        root: Path,
        *,
        retry_failed: bool = False,
        force: bool = False,
    ) -> IngestPlan:
        known = {r.path: r for r in self._records_under(root.resolve())}
        to_index: list[DiscoveredFile] = []
        unchanged: list[DiscoveredFile] = []
        skipped_failed: list[DiscoveredFile] = []
        for f in discovered:
            rec = known.get(f.path)
            if force or rec is None or rec.content_hash != f.content_hash:
                to_index.append(f)
            elif rec.status is FileStatus.FAILED:
                (to_index if retry_failed else skipped_failed).append(f)
            else:
                unchanged.append(f)
        seen = {f.path for f in discovered}
        removed = [r for p, r in sorted(known.items()) if p not in seen]
        return IngestPlan(to_index, unchanged, skipped_failed, removed)

    def record_indexed(
        self, file: DiscoveredFile, *, parser: str, parents: Sequence[ParentSection], chunk_ids: Sequence[str]
    ) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM parents WHERE doc_id = ?", (file.doc_id,))
            conn.executemany(
                "INSERT INTO parents VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (p.parent_id, file.doc_id, p.text, json.dumps(p.section_path), p.page_start, p.page_end)
                    for p in parents
                ],
            )
            _upsert_file(conn, file, FileStatus.INDEXED, None, parser, list(chunk_ids))

    def record_failed(self, file: DiscoveredFile, *, error: str) -> None:
        # Old chunks/parents belong to a previous version of the file; keeping them would serve stale content.
        with self._conn() as conn:
            conn.execute("DELETE FROM parents WHERE doc_id = ?", (file.doc_id,))
            _upsert_file(conn, file, FileStatus.FAILED, error, None, [])

    def remove(self, path: Path) -> None:
        with self._conn() as conn:
            row = conn.execute("SELECT doc_id FROM files WHERE path = ?", (str(path),)).fetchone()
            if row:
                conn.execute("DELETE FROM parents WHERE doc_id = ?", (row["doc_id"],))
            conn.execute("DELETE FROM files WHERE path = ?", (str(path),))

    def get_parents(self, parent_ids: Sequence[str]) -> dict[str, ParentSection]:
        if not parent_ids:
            return {}
        marks = ",".join("?" * len(parent_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM parents WHERE parent_id IN ({marks})", list(parent_ids)
            ).fetchall()
        return {
            r["parent_id"]: ParentSection(
                parent_id=r["parent_id"],
                text=r["text"],
                section_path=tuple(json.loads(r["section_path"])),
                page_start=r["page_start"],
                page_end=r["page_end"],
            )
            for r in rows
        }


def _upsert_file(
    conn: sqlite3.Connection,
    file: DiscoveredFile,
    status: FileStatus,
    error: str | None,
    parser: str | None,
    chunk_ids: list[str],
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(file.path),
            file.doc_id,
            file.content_hash,
            file.mtime,
            file.size,
            status.value,
            error,
            parser,
            json.dumps(chunk_ids),
            datetime.now(UTC).isoformat(),
        ),
    )


def _record(row: sqlite3.Row) -> FileRecord:
    return FileRecord(
        path=Path(row["path"]),
        doc_id=row["doc_id"],
        content_hash=row["content_hash"],
        status=FileStatus(row["status"]),
        error=row["error"],
        parser=row["parser"],
        chunk_ids=json.loads(row["chunk_ids"]),
    )
