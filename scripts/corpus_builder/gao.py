"""GAO reports: manually downloaded PDFs plus, where a PDF is missing, highlights from GAO's RSS feed.

GAO offers no API and gao.gov blocks scripted downloads of report pages/PDFs, so full reports are downloaded
by hand into <output_dir>/gao/ as gao-XX-XXXXXX.pdf. The RSS feed (the only machine-readable surface) supplies
titles and publication dates, and its "What GAO Found" highlights fill in for reports without a PDF.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Protocol, cast

import pymupdf

from corpus_builder.config import GaoConfig
from corpus_builder.registry import Registry, SourceRecord
from corpus_builder.stats import SourceStats

LICENSE = "Public domain (U.S. Government work, 17 U.S.C. 105)"
HIGHLIGHTS_NOTE = "RSS highlights only (no PDF provided; gao.gov blocks scripted downloads)"
PDF_NOTE = "full report PDF, downloaded manually (gao.gov blocks scripted downloads)"
PRODUCT_URL = "https://www.gao.gov/products/{id}"
_PDF_DATE = re.compile(r"D:(\d{4})(\d{2})(\d{2})")


class FeedClient(Protocol):
    def set_min_interval(self, host: str, seconds: float) -> None: ...

    def get_text(self, url: str) -> str: ...


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


def _pdf_metadata(path: Path) -> tuple[str, date | None]:
    with pymupdf.open(path) as pdf:
        # PyMuPDF's metadata is a plain str->str dict but is typed loosely.
        meta = cast(dict[str, str], pdf.metadata or {})  # pyright: ignore[reportUnknownMemberType]
    match = _PDF_DATE.match(meta.get("creationDate") or "")
    created = date(int(match[1]), int(match[2]), int(match[3])) if match else None
    return (meta.get("title") or path.stem).strip(), created


class GaoCollector:
    def __init__(
        self, cfg: GaoConfig, http: FeedClient, out_dir: Path, registry: Registry, cutoff: date
    ) -> None:
        self._cfg = cfg
        self._http = http
        self._out = out_dir
        self._registry = registry
        self._cutoff = cutoff
        http.set_min_interval("www.gao.gov", 2.0)

    def collect(self) -> SourceStats:
        stats = SourceStats("gao")
        feed = {i.report_number: i for i in parse_feed(self._http.get_text(self._cfg.feed_url))}
        with_pdf = self._register_pdfs(feed, stats)
        candidates = [
            i for i in feed.values() if i.published >= self._cutoff and i.report_number not in with_pdf
        ]
        for item in candidates[: max(self._cfg.max_reports - len(with_pdf), 0)]:
            self._write_highlights(item, stats)
        total = len(with_pdf) + min(len(candidates), max(self._cfg.max_reports - len(with_pdf), 0))
        if total < self._cfg.max_reports:
            stats.missing.append(f"only {total} reports after cutoff (wanted {self._cfg.max_reports})")
        return stats

    def _register_pdfs(self, feed: dict[str, GaoItem], stats: SourceStats) -> set[str]:
        numbers: set[str] = set()
        for pdf in sorted((self._out / "gao").glob("gao-*.pdf")):
            number = pdf.stem.upper()
            numbers.add(number)
            rel = f"gao/{pdf.name}"
            if self._registry.has(rel):
                stats.skipped += 1
                continue
            item = feed.get(number)
            meta_title, created = _pdf_metadata(pdf)
            published = item.published if item else created
            if published is not None and published < self._cutoff:
                stats.missing.append(f"{pdf.name} was published {published}, before the cutoff")
            self._registry.upsert(
                SourceRecord(
                    rel,
                    "gao",
                    item.title if item else meta_title,
                    PRODUCT_URL.format(id=pdf.stem.lower()),
                    published.isoformat() if published else "unknown",
                    LICENSE,
                    notes=PDF_NOTE,
                )
            )
            stats.added += 1
        return numbers

    def _write_highlights(self, item: GaoItem, stats: SourceStats) -> None:
        rel = f"gao/{item.report_number}_highlights.txt"
        if (self._out / rel).exists() and self._registry.has(rel):
            stats.skipped += 1
            return
        (self._out / rel).parent.mkdir(parents=True, exist_ok=True)
        (self._out / rel).write_text(render_report(item))
        self._registry.upsert(
            SourceRecord(
                rel, "gao", item.title, item.link, item.published.isoformat(), LICENSE, notes=HIGHLIGHTS_NOTE
            )
        )
        stats.added += 1
