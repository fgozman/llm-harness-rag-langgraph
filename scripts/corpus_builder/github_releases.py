"""GitHub releases since the cutoff, one Markdown file per repository."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from corpus_builder.config import GithubConfig
from corpus_builder.http import NotFoundError, PoliteClient
from corpus_builder.registry import Registry, SourceRecord
from corpus_builder.stats import SourceStats

API = "https://api.github.com"
_MAX_PAGES = 20


@dataclass(frozen=True, slots=True)
class Release:
    tag: str
    name: str
    published: date
    body: str
    url: str


def parse_releases(page: Sequence[dict[str, Any]], *, since: date) -> tuple[list[Release], bool]:
    """Releases on/after `since`, plus whether this page already reached older releases (stop paginating)."""
    out: list[Release] = []
    reached_cutoff = False
    for raw in page:
        if raw.get("draft") or not raw.get("published_at"):
            continue
        published = date.fromisoformat(raw["published_at"][:10])
        if published < since:
            reached_cutoff = True
            continue
        out.append(
            Release(raw["tag_name"], raw.get("name") or "", published, raw.get("body") or "", raw["html_url"])
        )
    return out, reached_cutoff


def render_release_notes(repo: str, releases: Sequence[Release], *, since: date, license_id: str) -> str:
    newest = max((r.published for r in releases), default=since)
    lines = [
        f"# {repo} release notes",
        "",
        # Newest date first: document-date inference reads the first date in the text.
        f"Last updated {newest.isoformat()}; covers releases since {since.isoformat()}.",
        f"Source: https://github.com/{repo}/releases (license: {license_id}).",
        "",
    ]
    for r in sorted(releases, key=lambda r: (r.published, r.tag), reverse=True):
        heading = r.tag if not r.name or r.name == r.tag else f"{r.name} ({r.tag})"
        lines += [
            f"## {heading} ({r.published.isoformat()})",
            "",
            r.body.strip() or "_No release notes._",
            "",
        ]
    return "\n".join(lines)


class GithubReleasesCollector:
    def __init__(
        self, cfg: GithubConfig, http: PoliteClient, out_dir: Path, registry: Registry, cutoff: date
    ) -> None:
        self._cfg = cfg
        self._http = http
        self._out = out_dir
        self._registry = registry
        self._cutoff = cutoff
        if os.environ.get("GITHUB_TOKEN"):
            http.add_header("api.github.com", "Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")

    def _releases(self, repo: str) -> list[Release]:
        releases: list[Release] = []
        for page in range(1, _MAX_PAGES + 1):
            raw = self._http.get_json(
                f"{API}/repos/{repo}/releases", params={"per_page": "100", "page": str(page)}
            )
            batch, reached_cutoff = parse_releases(raw, since=self._cutoff)
            releases += batch
            if reached_cutoff or len(raw) < 100:
                break
        return releases

    def _license(self, repo: str) -> str:
        try:
            info = self._http.get_json(f"{API}/repos/{repo}/license")
        except NotFoundError:
            return "unknown"
        spdx = info["license"]["spdx_id"]
        return spdx if spdx and spdx != "NOASSERTION" else info["license"]["name"]

    def collect(self) -> SourceStats:
        stats = SourceStats("github")
        for repo in self._cfg.repos:
            rel = f"release_notes/{repo.replace('/', '__')}.md"
            if (self._out / rel).exists() and self._registry.has(rel):
                stats.skipped += 1
                continue
            releases = self._releases(repo)
            license_id = self._license(repo)
            (self._out / rel).parent.mkdir(parents=True, exist_ok=True)
            (self._out / rel).write_text(
                render_release_notes(repo, releases, since=self._cutoff, license_id=license_id)
            )
            newest = max((r.published for r in releases), default=self._cutoff)
            self._registry.upsert(
                SourceRecord(
                    rel,
                    "github",
                    f"{repo} release notes",
                    f"https://github.com/{repo}/releases",
                    newest.isoformat(),
                    license_id,
                    notes=f"{len(releases)} releases since {self._cutoff.isoformat()}",
                )
            )
            stats.added += 1
        return stats
