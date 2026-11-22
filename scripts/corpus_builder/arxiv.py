"""arXiv: recent papers in one category under CC BY / CC0 only.

License metadata exists only in OAI-PMH (`arXiv` format), whose datestamps are modification times; the first
submission month is therefore taken from the arXiv ID (YYMM.NNNNN), and the v1 date from the Atom API.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from corpus_builder.config import ArxivConfig
from corpus_builder.http import PoliteClient
from corpus_builder.registry import Registry, SourceRecord
from corpus_builder.stats import SourceStats

log = logging.getLogger(__name__)

OAI_URL = "https://oaipmh.arxiv.org/oai"
ATOM_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://arxiv.org/pdf/{id}"
ABS_URL = "https://arxiv.org/abs/{id}"
_NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "ax": "http://arxiv.org/OAI/arXiv/"}
_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
_NEW_STYLE_ID = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}$")


class OaiError(RuntimeError):
    """The OAI-PMH endpoint answered with an <error> element (bad set, expired token, ...)."""


def oai_set(category: str) -> str:
    # OAI set specs are group:archive:CATEGORY, e.g. cs.IR -> cs:cs:IR.
    archive, sub = category.split(".", 1)
    return f"{archive}:{archive}:{sub}"


@dataclass(frozen=True, slots=True)
class ArxivRecord:
    arxiv_id: str
    title: str
    categories: list[str]
    license: str


def _text(node: ET.Element, path: str) -> str:
    found = node.find(path, _NS)
    return " ".join((found.text or "").split()) if found is not None else ""


def parse_oai_page(xml_text: str) -> tuple[list[ArxivRecord], str | None]:
    root = ET.fromstring(xml_text)
    error = root.find("oai:error", _NS)
    if error is not None:
        code = error.get("code", "")
        if code == "noRecordsMatch":
            return [], None
        raise OaiError(f"{code}: {(error.text or '').strip()}")
    records = [
        ArxivRecord(
            arxiv_id=_text(meta, "ax:id"),
            title=_text(meta, "ax:title"),
            categories=_text(meta, "ax:categories").split(),
            license=_text(meta, "ax:license"),
        )
        for meta in root.iterfind(".//oai:record/oai:metadata/ax:arXiv", _NS)
    ]
    token_node = root.find(".//oai:resumptionToken", _NS)
    token = (token_node.text or "").strip() if token_node is not None else ""
    return records, token or None


def accept(record: ArxivRecord, cfg: ArxivConfig) -> bool:
    match = _NEW_STYLE_ID.match(record.arxiv_id)
    if match is None:
        return False
    yy, mm = int(match.group(1)), int(match.group(2))
    since = cfg.submitted_on_or_after
    if (2000 + yy, mm) < (since.year, since.month):
        return False
    if cfg.require_primary_category and (not record.categories or record.categories[0] != cfg.category):
        return False
    if not cfg.require_primary_category and cfg.category not in record.categories:
        return False
    return any(record.license.startswith(prefix) for prefix in cfg.allowed_licenses)


def parse_atom_published(xml_text: str) -> dict[str, date]:
    out: dict[str, date] = {}
    for entry in ET.fromstring(xml_text).iterfind("a:entry", _ATOM_NS):
        raw_id = (entry.findtext("a:id", "", _ATOM_NS)).rsplit("/", 1)[-1]
        published = entry.findtext("a:published", "", _ATOM_NS)
        if raw_id and published:
            out[re.sub(r"v\d+$", "", raw_id)] = date.fromisoformat(published[:10])
    return out


class ArxivCollector:
    def __init__(self, cfg: ArxivConfig, http: PoliteClient, out_dir: Path, registry: Registry) -> None:
        self._cfg = cfg
        self._http = http
        self._out = out_dir
        self._registry = registry
        # arXiv API terms: no more than one request every 3 seconds.
        for host in ("oaipmh.arxiv.org", "export.arxiv.org", "arxiv.org"):
            http.set_min_interval(host, cfg.delay_s)

    def _harvest(self) -> list[ArxivRecord]:
        params = {
            "verb": "ListRecords",
            "metadataPrefix": "arXiv",
            "set": oai_set(self._cfg.category),
            "from": self._cfg.submitted_on_or_after.isoformat(),
        }
        records: list[ArxivRecord] = []
        while True:
            page, token = parse_oai_page(self._http.get_text(OAI_URL, params=params))
            records.extend(page)
            if token is None:
                return records
            params = {"verb": "ListRecords", "resumptionToken": token}

    def collect(self) -> SourceStats:
        stats = SourceStats("arxiv")
        chosen = sorted((r for r in self._harvest() if accept(r, self._cfg)), key=lambda r: r.arxiv_id)
        chosen = chosen[: self._cfg.max_papers]
        if len(chosen) < self._cfg.max_papers:
            stats.missing.append(f"only {len(chosen)} eligible papers (wanted {self._cfg.max_papers})")
        if not chosen:
            return stats
        published = parse_atom_published(
            self._http.get_text(
                ATOM_URL,
                params={"id_list": ",".join(r.arxiv_id for r in chosen), "max_results": str(len(chosen))},
            )
        )
        for record in chosen:
            rel = f"arxiv/{record.arxiv_id}.pdf"
            if (self._out / rel).exists() and self._registry.has(rel):
                stats.skipped += 1
                continue
            (self._out / rel).parent.mkdir(parents=True, exist_ok=True)
            (self._out / rel).write_bytes(self._http.get_bytes(PDF_URL.format(id=record.arxiv_id)))
            pub = published.get(record.arxiv_id)
            self._registry.upsert(
                SourceRecord(
                    path=rel,
                    source="arxiv",
                    title=record.title,
                    url=ABS_URL.format(id=record.arxiv_id),
                    published=pub.isoformat() if pub else "unknown",
                    license=_license_name(record.license),
                    license_url=record.license,
                    notes=f"categories: {' '.join(record.categories)}",
                )
            )
            stats.added += 1
        return stats


def _license_name(url: str) -> str:
    if "publicdomain/zero" in url:
        return "CC0 1.0"
    version = re.search(r"/by/([\d.]+)/", url)
    return f"CC BY {version.group(1)}" if version else "CC BY"
