"""Find supported documents under a folder and fingerprint them."""

from __future__ import annotations

import fnmatch
import hashlib
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = frozenset(
    {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".markdown", ".txt"}
)

_DOC_NAMESPACE = uuid.UUID("0b8f3c52-1d7e-4a3b-9e61-5f2a8c4d7e10")
_HASH_BLOCK = 1 << 20
IGNORE_FILE = ".ragignore"


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    path: Path
    doc_id: str
    content_hash: str
    mtime: float
    size: int


def doc_id_for(path: Path) -> str:
    return str(uuid.uuid5(_DOC_NAMESPACE, str(path)))


def _is_candidate(path: Path, root: Path) -> bool:
    rel_parts = path.relative_to(root).parts if path != root else (path.name,)
    # Hidden files/dirs (.git, .DS_Store) and Office lock files (~$x.docx) are never documents.
    if any(part.startswith((".", "~$")) for part in rel_parts):
        return False
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path) -> DiscoveredFile:
    stat = path.stat()
    return DiscoveredFile(
        path=path,
        doc_id=doc_id_for(path),
        content_hash=_sha256(path),
        mtime=stat.st_mtime,
        size=stat.st_size,
    )


def _ignore_patterns(root: Path) -> list[str]:
    # Glob patterns relative to the ingest root, one per line; for non-content files such as provenance lists.
    ignore = root / IGNORE_FILE
    if not ignore.is_file():
        return []
    lines = (line.strip() for line in ignore.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def iter_candidates(root: Path) -> Iterator[Path]:
    root = root.resolve()
    if root.is_file():
        if _is_candidate(root, root):
            yield root
        return
    patterns = _ignore_patterns(root)
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if _is_candidate(path, root) and not any(fnmatch.fnmatch(rel, p) for p in patterns):
            yield path


def discover(root: Path) -> list[DiscoveredFile]:
    return [_fingerprint(p) for p in iter_candidates(root)]
