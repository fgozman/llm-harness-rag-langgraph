"""Federal Register (federalregister.gov API v1): recent rules as full-text HTML."""

from __future__ import annotations

import html
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from corpus_builder.config import FederalRegisterConfig
from corpus_builder.http import PoliteClient
from corpus_builder.registry import Registry, SourceRecord
from corpus_builder.stats import SourceStats

API_URL = "https://www.federalregister.gov/api/v1/documents.json"
LICENSE = "Public domain (U.S. Government work, 17 U.S.C. 105)"
_FIELDS = [
    "document_number",
    "title",
    "publication_date",
    "page_length",
    "body_html_url",
    "html_url",
    "agencies",
]


def select_documents(
    docs: Sequence[dict[str, Any]], *, min_pages: int, max_pages: int, limit: int
) -> list[dict[str, Any]]:
    # Bounds skip one-page notices (little to retrieve) and 300-page rules (dominate the corpus).
    return [d for d in docs if min_pages <= (d.get("page_length") or 0) <= max_pages][:limit]


class FederalRegisterCollector:
    def __init__(
        self, cfg: FederalRegisterConfig, http: PoliteClient, out_dir: Path, registry: Registry, cutoff: date
    ) -> None:
        self._cfg = cfg
        self._http = http
        self._out = out_dir
        self._registry = registry
        self._cutoff = cutoff
        http.set_min_interval("www.federalregister.gov", 1.0)

    def _query(self) -> list[dict[str, Any]]:
        # Array params must repeat the bare `[]` key: indexed keys are rejected (fields) or ignored (type).
        params: list[tuple[str, str]] = [
            ("per_page", "100"),
            ("order", "newest"),
            ("conditions[publication_date][gte]", self._cutoff.isoformat()),
        ]
        params += [("conditions[type][]", t) for t in self._cfg.document_types]
        if self._cfg.significant_only:
            params.append(("conditions[significant]", "1"))
        params += [("fields[]", f) for f in _FIELDS]
        results: list[dict[str, Any]] = self._http.get_json(API_URL, params=params)["results"]
        return results

    def collect(self) -> SourceStats:
        stats = SourceStats("federal_register")
        chosen = select_documents(
            self._query(),
            min_pages=self._cfg.min_pages,
            max_pages=self._cfg.max_pages,
            limit=self._cfg.max_documents,
        )
        for doc in chosen:
            rel = f"federal_register/{doc['document_number']}.html"
            if (self._out / rel).exists() and self._registry.has(rel):
                stats.skipped += 1
                continue
            body = self._http.get_text(doc["body_html_url"])
            agencies = ", ".join(a["name"] for a in doc.get("agencies", []) if a.get("name"))
            title = html.escape(doc["title"])
            page = (
                "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
                f"<title>{title}</title></head><body>"
                f"<h1>{title}</h1><p>Federal Register document {doc['document_number']}, published "
                f"{doc['publication_date']} by {html.escape(agencies)}.</p>{body}</body></html>"
            )
            (self._out / rel).parent.mkdir(parents=True, exist_ok=True)
            (self._out / rel).write_text(page)
            self._registry.upsert(
                SourceRecord(
                    rel,
                    "federal_register",
                    doc["title"],
                    doc["html_url"],
                    doc["publication_date"],
                    LICENSE,
                    notes=f"{doc.get('page_length')} pages; {agencies}",
                )
            )
            stats.added += 1
        if len(chosen) < self._cfg.max_documents:
            stats.missing.append(f"only {len(chosen)} documents matched")
        return stats
