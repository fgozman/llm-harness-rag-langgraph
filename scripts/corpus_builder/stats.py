"""Per-source outcome of a corpus build run."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SourceStats:
    source: str
    added: int = 0
    skipped: int = 0
    missing: list[str] = field(default_factory=list[str])
    notes: list[str] = field(default_factory=list[str])
