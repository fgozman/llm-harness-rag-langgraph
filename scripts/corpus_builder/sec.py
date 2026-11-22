"""SEC EDGAR: 10-Qs for configured filing periods, their earnings 8-K Exhibit 99.1, Financial_Report.xlsx
where EDGAR provides one, and one pre-cutoff 10-Q per company as a temporal hard negative."""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from corpus_builder.config import FilingPeriod, SecConfig
from corpus_builder.http import NotFoundError, PoliteClient
from corpus_builder.registry import Registry, SourceRecord
from corpus_builder.stats import SourceStats

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}"
LICENSE = "SEC EDGAR public filing (publicly available; not copyright-cleared, local evaluation use only)"
_DOC_RE = re.compile(r"<TYPE>([^\n<]+)\s.*?<FILENAME>([^\n<]+)", re.S)


@dataclass(frozen=True, slots=True)
class Filing:
    form: str
    filing_date: date
    report_date: date | None
    accession: str
    primary_document: str
    items: str


def filings_from_submissions(data: Mapping[str, Any]) -> list[Filing]:
    recent: dict[str, list[str]] = data["filings"]["recent"]
    columns = zip(
        recent["form"],
        recent["filingDate"],
        recent["reportDate"],
        recent["accessionNumber"],
        recent["primaryDocument"],
        recent["items"],
        strict=True,
    )
    return [
        Filing(form, date.fromisoformat(filed), date.fromisoformat(rep) if rep else None, acc, doc, items)
        for form, filed, rep, acc, doc, items in columns
    ]


def _latest(filings: list[Filing]) -> Filing | None:
    return max(filings, key=lambda f: f.filing_date, default=None)


def select_period_10q(filings: list[Filing], period: FilingPeriod) -> Filing | None:
    return _latest(
        [f for f in filings if f.form == "10-Q" and period.filed_from <= f.filing_date <= period.filed_to]
    )


def select_pre_cutoff_10q(filings: list[Filing], cutoff: date) -> Filing | None:
    return _latest([f for f in filings if f.form == "10-Q" and f.filing_date < cutoff])


def select_earnings_8k(filings: list[Filing], ten_q: Filing, *, window_days: int) -> Filing | None:
    start = ten_q.filing_date - timedelta(days=window_days)
    return _latest(
        [
            f
            for f in filings
            if f.form == "8-K"
            and "2.02" in f.items.split(",")
            and start <= f.filing_date <= ten_q.filing_date
        ]
    )


def exhibit_filename(index_headers_html: str, exhibit_type: str) -> str | None:
    for doc_type, filename in _DOC_RE.findall(html.unescape(index_headers_html)):
        if doc_type.strip() == exhibit_type:
            return filename.strip()
    return None


def has_financial_report(index_listing: Mapping[str, Any]) -> bool:
    return any(item["name"] == "Financial_Report.xlsx" for item in index_listing["directory"]["item"])


class SecCollector:
    def __init__(
        self, cfg: SecConfig, http: PoliteClient, out_dir: Path, registry: Registry, cutoff: date
    ) -> None:
        self._cfg = cfg
        self._http = http
        self._out = out_dir
        self._registry = registry
        self._cutoff = cutoff
        for host in ("www.sec.gov", "data.sec.gov"):
            http.set_min_interval(host, cfg.min_interval_s)

    def collect(self) -> SourceStats:
        stats = SourceStats("sec")
        ciks = {v["ticker"]: int(v["cik_str"]) for v in self._http.get_json(TICKERS_URL).values()}
        for ticker in self._cfg.tickers:
            if ticker not in ciks:
                stats.missing.append(f"{ticker}: unknown ticker")
                continue
            self._collect_company(ticker, ciks[ticker], stats)
        return stats

    def _collect_company(self, ticker: str, cik: int, stats: SourceStats) -> None:
        data = self._http.get_json(SUBMISSIONS_URL.format(cik=cik))
        company: str = data["name"]
        filings = filings_from_submissions(data)
        for period in self._cfg.periods:
            ten_q = select_period_10q(filings, period)
            if ten_q is None:
                stats.missing.append(f"{ticker}: no 10-Q filed in {period.label}")
                continue
            self._save_10q(ticker, cik, company, ten_q, stats, pre_cutoff=False)
            eight_k = select_earnings_8k(filings, ten_q, window_days=self._cfg.earnings_8k_window_days)
            if eight_k is None:
                stats.missing.append(f"{ticker}: no earnings 8-K before 10-Q of {ten_q.filing_date}")
            else:
                self._save_exhibit(ticker, cik, company, eight_k, stats)
        if self._cfg.include_pre_cutoff:
            pre = select_pre_cutoff_10q(filings, self._cutoff)
            if pre is None:
                stats.missing.append(f"{ticker}: no pre-cutoff 10-Q")
            else:
                self._save_10q(ticker, cik, company, pre, stats, pre_cutoff=True)

    def _archive(self, cik: int, filing: Filing) -> str:
        return ARCHIVE_URL.format(cik=cik, acc=filing.accession.replace("-", ""))

    def _save(self, rel_path: str, url: str, record: SourceRecord, stats: SourceStats) -> None:
        target = self._out / rel_path
        if target.exists() and self._registry.has(rel_path):
            stats.skipped += 1
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self._http.get_bytes(url))
        self._registry.upsert(record)
        stats.added += 1

    def _save_10q(
        self, ticker: str, cik: int, company: str, f: Filing, stats: SourceStats, *, pre_cutoff: bool
    ) -> None:
        period = f.report_date.isoformat() if f.report_date else "unknown"
        base = f"sec/{ticker}_10-Q_period-{period}_filed-{f.filing_date.isoformat()}"
        url = f"{self._archive(cik, f)}/{f.primary_document}"
        note = "temporal hard negative" if pre_cutoff else ""
        title = f"{company} Form 10-Q for the period ended {period}"
        self._save(
            f"{base}.htm",
            url,
            SourceRecord(
                f"{base}.htm",
                "sec",
                title,
                url,
                f.filing_date.isoformat(),
                LICENSE,
                None,
                pre_cutoff,
                False,
                note,
            ),
            stats,
        )
        if pre_cutoff:
            return
        listing = self._http.get_json(f"{self._archive(cik, f)}/index.json")
        if not has_financial_report(listing):
            stats.notes.append(f"{ticker} {f.filing_date}: Financial_Report.xlsx not provided by EDGAR")
            return
        xlsx_url = f"{self._archive(cik, f)}/Financial_Report.xlsx"
        try:
            self._save(
                f"{base}_Financial_Report.xlsx",
                xlsx_url,
                SourceRecord(
                    f"{base}_Financial_Report.xlsx",
                    "sec",
                    f"{title} (XBRL financial report)",
                    xlsx_url,
                    f.filing_date.isoformat(),
                    LICENSE,
                ),
                stats,
            )
        except NotFoundError:
            stats.notes.append(f"{ticker} {f.filing_date}: Financial_Report.xlsx listed but not downloadable")

    def _save_exhibit(self, ticker: str, cik: int, company: str, f: Filing, stats: SourceStats) -> None:
        acc = f.accession
        headers = self._http.get_text(f"{self._archive(cik, f)}/{acc}-index-headers.html")
        filename = exhibit_filename(headers, "EX-99.1")
        if filename is None:
            stats.missing.append(f"{ticker}: 8-K {f.filing_date} has no EX-99.1")
            return
        rel = f"sec/{ticker}_8-K_filed-{f.filing_date.isoformat()}_ex99-1{Path(filename).suffix}"
        url = f"{self._archive(cik, f)}/{filename}"
        title = f"{company} earnings release (8-K Exhibit 99.1) filed {f.filing_date.isoformat()}"
        self._save(rel, url, SourceRecord(rel, "sec", title, url, f.filing_date.isoformat(), LICENSE), stats)
