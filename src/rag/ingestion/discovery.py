"""Find supported documents under a folder and fingerprint them."""

from __future__ import annotations

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


def iter_candidates(root: Path) -> Iterator[Path]:
    root = root.resolve()
    if root.is_file():
        if _is_candidate(root, root):
            yield root
        return
    for path in sorted(root.rglob("*")):
        if _is_candidate(path, root):
            yield path


def discover(root: Path) -> list[DiscoveredFile]:
    return [_fingerprint(p) for p in iter_candidates(root)]
