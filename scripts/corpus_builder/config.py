"""Typed view of scripts/corpus_sources.yaml."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class FilingPeriod(BaseModel):
    label: str
    filed_from: date
    filed_to: date


class SecConfig(BaseModel):
    enabled: bool = True
    tickers: list[str]
    periods: list[FilingPeriod]
    earnings_8k_window_days: int = 21
    include_pre_cutoff: bool = True
    # SEC allows 10 req/s; stay well under it.
    min_interval_s: float = Field(default=0.25, ge=0.1)


class ArxivConfig(BaseModel):
    enabled: bool = True
    category: str = "cs.IR"
    submitted_on_or_after: date = date(2026, 8, 1)
    max_papers: int = 30
    require_primary_category: bool = True
    # License URL prefixes accepted: CC BY (any version) and CC0 only.
    allowed_licenses: list[str] = Field(
        default_factory=lambda: [
            "http://creativecommons.org/licenses/by/",
            "http://creativecommons.org/publicdomain/zero/",
        ]
    )
    delay_s: float = Field(default=3.0, ge=3.0)


class GaoConfig(BaseModel):
    enabled: bool = True
    feed_url: str = "https://www.gao.gov/rss/reports.xml"
    max_reports: int = 10


class FederalRegisterConfig(BaseModel):
    enabled: bool = True
    max_documents: int = 10
    document_types: list[str] = Field(default_factory=lambda: ["RULE"])
    significant_only: bool = True
    min_pages: int = 3
    max_pages: int = 80


class GithubConfig(BaseModel):
    enabled: bool = True
    repos: list[str]


class CanaryConfig(BaseModel):
    enabled: bool = True
    facts_path: Path = Path("eval/canary_facts.jsonl")


class CorpusConfig(BaseModel):
    output_dir: Path = Path("data/test_corpus")
    cutoff_date: date = date(2026, 7, 1)
    sec: SecConfig
    arxiv: ArxivConfig = Field(default_factory=ArxivConfig)
    gao: GaoConfig = Field(default_factory=GaoConfig)
    federal_register: FederalRegisterConfig = Field(default_factory=FederalRegisterConfig)
    github: GithubConfig
    canary: CanaryConfig = Field(default_factory=CanaryConfig)


def load_config(path: Path) -> CorpusConfig:
    return CorpusConfig.model_validate(yaml.safe_load(path.read_text()))
