"""Metadata filters (file name, folder, modification date) and their Qdrant translation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from qdrant_client import models


@dataclass(frozen=True, slots=True)
class SearchFilters:
    file_names: tuple[str, ...] = ()
    folder: Path | None = None  # matches this folder and everything below it
    modified_after: datetime | None = None
    modified_before: datetime | None = None

    def is_empty(self) -> bool:
        return not (self.file_names or self.folder or self.modified_after or self.modified_before)


def to_qdrant_filter(filters: SearchFilters | None, *, folders: Sequence[str]) -> models.Filter | None:
    """`folders`: the indexed folders under `filters.folder` (from the manifest), because Qdrant keyword
    payloads can't do path-prefix matching."""
    if filters is None or filters.is_empty():
        return None
    must: list[models.Condition] = []
    if filters.file_names:
        must.append(
            models.FieldCondition(key="file_name", match=models.MatchAny(any=list(filters.file_names)))
        )
    if filters.folder is not None:
        # An unknown folder must match nothing; an empty condition list would match everything.
        targets = list(folders) or [str(filters.folder)]
        must.append(models.FieldCondition(key="folder", match=models.MatchAny(any=targets)))
    if filters.modified_after or filters.modified_before:
        must.append(
            models.FieldCondition(
                key="file_mtime",
                range=models.Range(
                    gte=filters.modified_after.timestamp() if filters.modified_after else None,
                    lte=filters.modified_before.timestamp() if filters.modified_before else None,
                ),
            )
        )
    return models.Filter(must=must)
