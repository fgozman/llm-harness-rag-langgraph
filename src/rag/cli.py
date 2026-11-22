"""Command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

from rag.config import Settings, get_settings

if TYPE_CHECKING:
    from rag.ingestion.pipeline import IngestStats

app = typer.Typer(help="RAG over a local document folder.", no_args_is_help=True)
console = Console()

_HEALTH_TIMEOUT_S = 3.0


_NOISY_LOGGERS = ("httpx", "docling", "sentence_transformers", "transformers", "huggingface_hub", "RapidOCR")


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False) -> None:
    """RAG over a local document folder."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if not verbose:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


@app.command("config")
def show_config() -> None:
    """Print the fully resolved settings (secrets masked)."""
    settings = get_settings()
    # mode="json" renders SecretStr as "**********", so secrets never reach the terminal.
    console.print_json(json.dumps(settings.model_dump(mode="json")))


def _probe(url: str) -> str:
    try:
        response = httpx.get(url, timeout=_HEALTH_TIMEOUT_S)
    except httpx.HTTPError as exc:
        return f"down ({type(exc).__name__})"
    return "ok" if response.status_code == 200 else f"down (HTTP {response.status_code})"


def _service_checks(settings: Settings) -> list[tuple[str, str, str]]:
    checks = [("qdrant", settings.qdrant.url, _probe(f"{settings.qdrant.url}/readyz"))]
    phoenix_url = settings.phoenix.collector_endpoint
    phoenix_status = _probe(f"{phoenix_url}/healthz") if settings.phoenix.enabled else "disabled"
    checks.append(("phoenix", phoenix_url, phoenix_status))
    return checks


@app.command()
def doctor() -> None:
    """Check service reachability; exit 1 if a required service is down."""
    settings = get_settings()
    table = Table("service", "url", "status")
    checks = _service_checks(settings)
    for name, url, status in checks:
        table.add_row(name, url, status)
    console.print(table)
    # Phoenix is optional (the app must keep working without tracing), so only Qdrant is fatal.
    if any(name == "qdrant" and status.startswith("down") for name, _, status in checks):
        raise typer.Exit(code=1)


@app.command()
def ingest(
    folder: Annotated[
        Path | None, typer.Argument(help="Folder or file to index (default: docs_dir).")
    ] = None,
    retry_failed: Annotated[bool, typer.Option(help="Retry files that failed to parse before.")] = False,
    force: Annotated[bool, typer.Option(help="Re-index every file even if unchanged.")] = False,
    contextualize: Annotated[
        bool | None,
        typer.Option("--contextualize/--no-contextualize", help="Override contextual enrichment."),
    ] = None,
) -> None:
    """Index new/changed files, drop removed ones. Safe to re-run (no-op when nothing changed)."""
    from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

    from rag.factory import ConfigurationError, build_ingestion_pipeline

    settings = get_settings()
    if contextualize is not None:
        settings.ingestion.contextualize = contextualize
    root = (folder or settings.docs_dir).resolve()
    if not root.exists():
        console.print(f"[red]Not found:[/red] {root}")
        raise typer.Exit(code=2)
    try:
        pipeline = build_ingestion_pipeline(settings)
    except ConfigurationError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    with Progress(
        SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), console=console
    ) as bar:
        tasks = {
            "parsed": bar.add_task("parse + chunk", total=None),
            "indexed": bar.add_task("index", total=None),
        }
        stats = pipeline.run(
            root, retry_failed=retry_failed, progress=lambda stage, n: bar.advance(tasks[stage], n)
        )
    _print_ingest_summary(stats, contextualized=settings.ingestion.contextualize)


def _print_ingest_summary(stats: IngestStats, *, contextualized: bool) -> None:
    table = Table("metric", "value", title="Ingestion summary")
    rows: list[tuple[str, object]] = [
        ("files discovered", stats.discovered),
        ("indexed (new/changed)", stats.indexed),
        ("unchanged (skipped)", stats.unchanged),
        ("previously failed (skipped)", stats.skipped_failed),
        ("failed", stats.failed),
        ("removed", stats.removed),
        ("chunks written", stats.chunks_written),
        ("time", f"{stats.seconds:.1f}s"),
    ]
    if contextualized:
        rows += [
            ("LLM tokens in/out", f"{stats.llm_input_tokens}/{stats.llm_output_tokens}"),
            ("LLM cache read/write", f"{stats.llm_cache_read_tokens}/{stats.llm_cache_write_tokens}"),
            ("chunks without context", stats.contextualize_failures),
            ("estimated LLM cost", f"${stats.llm_cost_usd:.4f}"),
        ]
    for name, value in rows:
        table.add_row(name, str(value))
    console.print(table)
    for path, error in stats.failures.items():
        console.print(f"[yellow]failed:[/yellow] {path}: {error}")
