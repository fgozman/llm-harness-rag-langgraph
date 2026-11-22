"""Build data/test_corpus/ from the fixed source list in scripts/corpus_sources.yaml.

Idempotent and cached: files already recorded are skipped, and every HTTP response is cached under
<output_dir>/.cache, so re-runs are offline. Use --refresh to re-fetch listings and pick up new documents.

    uv run python scripts/build_test_corpus.py [--only sec,arxiv,...] [--refresh]
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from corpus_builder.arxiv import ArxivCollector
from corpus_builder.canary import build_canaries
from corpus_builder.config import CorpusConfig, load_config
from corpus_builder.fedreg import FederalRegisterCollector
from corpus_builder.gao import GaoCollector
from corpus_builder.github_releases import GithubReleasesCollector
from corpus_builder.http import PoliteClient
from corpus_builder.registry import RAGIGNORE, Registry
from corpus_builder.sec import SecCollector
from corpus_builder.stats import SourceStats
from rag.config import Settings

ALL_SOURCES = ("sec", "arxiv", "gao", "federal_register", "github", "canary")
DEFAULT_CONFIG = Path(__file__).with_name("corpus_sources.yaml")
console = Console()


def _collectors(
    cfg: CorpusConfig, http: PoliteClient, registry: Registry
) -> dict[str, tuple[bool, Callable[[], SourceStats]]]:
    out = cfg.output_dir
    return {
        "sec": (
            cfg.sec.enabled,
            lambda: SecCollector(cfg.sec, http, out, registry, cfg.cutoff_date).collect(),
        ),
        "arxiv": (cfg.arxiv.enabled, lambda: ArxivCollector(cfg.arxiv, http, out, registry).collect()),
        "gao": (
            cfg.gao.enabled,
            lambda: GaoCollector(cfg.gao, http, out, registry, cfg.cutoff_date).collect(),
        ),
        "federal_register": (
            cfg.federal_register.enabled,
            lambda: FederalRegisterCollector(
                cfg.federal_register, http, out, registry, cfg.cutoff_date
            ).collect(),
        ),
        "github": (
            cfg.github.enabled,
            lambda: GithubReleasesCollector(cfg.github, http, out, registry, cfg.cutoff_date).collect(),
        ),
        "canary": (cfg.canary.enabled, lambda: build_canaries(out, cfg.canary.facts_path, registry)),
    }


def main(
    only: Annotated[
        str | None, typer.Option(help=f"Comma-separated subset of: {', '.join(ALL_SOURCES)}")
    ] = None,
    refresh: Annotated[bool, typer.Option(help="Ignore the HTTP cache and re-fetch listings.")] = False,
    config: Annotated[Path, typer.Option(help="Source list YAML.")] = DEFAULT_CONFIG,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config(config)
    selected = [s.strip() for s in only.split(",")] if only else list(ALL_SOURCES)
    unknown = set(selected) - set(ALL_SOURCES)
    if unknown:
        raise typer.BadParameter(f"unknown sources: {', '.join(sorted(unknown))}")

    email = Settings().corpus.sec_user_agent_email
    if not email:
        console.print(
            "[red]CORPUS__SEC_USER_AGENT_EMAIL is not set[/red] (SEC fair-access rules need a contact)."
        )
        raise typer.Exit(code=2)
    http = PoliteClient(
        cfg.output_dir / ".cache" / "http",
        user_agent=f"llm-harness-rag test-corpus builder (contact: {email})",
        refresh=refresh,
    )
    registry = Registry(cfg.output_dir / "sources.json")
    results: list[SourceStats] = []
    for name, (enabled, run) in _collectors(cfg, http, registry).items():
        if name not in selected or not enabled:
            continue
        console.print(f"[bold]== {name}[/bold]")
        try:
            results.append(run())
        finally:
            for path in registry.prune_missing(cfg.output_dir):
                console.print(f"[dim]dropped record for missing file {path}[/dim]")
            # Persist provenance after every source so an interrupted run loses nothing.
            registry.save()
            (cfg.output_dir / "SOURCES.md").write_text(registry.render_markdown())
            (cfg.output_dir / ".ragignore").write_text(RAGIGNORE)
    _summary(results, len(registry.records))


def _summary(results: list[SourceStats], total: int) -> None:
    table = Table("source", "added", "skipped (cached)", "missing", title=f"Test corpus: {total} documents")
    for s in results:
        table.add_row(s.source, str(s.added), str(s.skipped), str(len(s.missing)))
    console.print(table)
    for s in results:
        for line in s.missing:
            console.print(f"[yellow]{s.source} missing:[/yellow] {line}")
        for line in s.notes:
            console.print(f"[dim]{s.source} note:[/dim] {line}")


if __name__ == "__main__":
    typer.run(main)
