"""GAO: report highlights from GAO's official RSS feed.

GAO offers no API and gao.gov blocks automated downloads of report pages/PDFs, so only the RSS "What GAO
Found" highlights (public domain, plain text) are collected. Full reports must be added manually if needed.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime
from pathlib import Path

from corpus_builder.config import GaoConfig
from corpus_builder.http import PoliteClient
from corpus_builder.registry import Registry, SourceRecord
from corpus_builder.stats import SourceStats

LICENSE = "Public domain (U.S. Government work, 17 U.S.C. 105)"
NOTE = "RSS highlights only: gao.gov blocks automated report/PDF downloads"


@dataclass(frozen=True, slots=True)
class GaoItem:
    title: str
    link: str
    report_number: str
    published: date
    description: str


def parse_feed(xml_text: str) -> list[GaoItem]:
    items: list[GaoItem] = []
    for node in ET.fromstring(xml_text).iterfind("./channel/item"):
        link = node.findtext("link", "").strip()
        items.append(
            GaoItem(
                title=" ".join(node.findtext("title", "").split()),
                link=link,
                report_number=link.rstrip("/").rsplit("/", 1)[-1].upper(),
                published=parsedate_to_datetime(node.findtext("pubDate", "")).date(),
                description=node.findtext("description", ""),
            )
        )
    return items


def _normalize(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    # Keep paragraph breaks (one blank line), drop the feed's runs of blank/indented lines.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def render_report(item: GaoItem) -> str:
    header = (
        f"{item.title}\n\n"
        f"GAO report {item.report_number}, published {item.published.isoformat()}.\n"
        f"Source: {item.link}\n"
        "Content: report highlights from GAO's RSS feed (full report not included).\n"
    )
    return f"{header}\n{_normalize(item.description)}\n"


class GaoCollector:
    def __init__(
        self, cfg: GaoConfig, http: PoliteClient, out_dir: Path, registry: Registry, cutoff: date
    ) -> None:
        self._cfg = cfg
        self._http = http
        self._out = out_dir
        self._registry = registry
        self._cutoff = cutoff
        http.set_min_interval("www.gao.gov", 2.0)

    def collect(self) -> SourceStats:
        stats = SourceStats("gao", notes=[NOTE])
        items = [
            i for i in parse_feed(self._http.get_text(self._cfg.feed_url)) if i.published >= self._cutoff
        ]
        for item in items[: self._cfg.max_reports]:
            rel = f"gao/{item.report_number}_highlights.txt"
            if (self._out / rel).exists() and self._registry.has(rel):
                stats.skipped += 1
                continue
            (self._out / rel).parent.mkdir(parents=True, exist_ok=True)
            (self._out / rel).write_text(render_report(item))
            self._registry.upsert(
                SourceRecord(
                    rel, "gao", item.title, item.link, item.published.isoformat(), LICENSE, notes=NOTE
                )
            )
            stats.added += 1
        if len(items) < self._cfg.max_reports:
            stats.missing.append(f"feed had only {len(items)} reports after cutoff")
        return stats
