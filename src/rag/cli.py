"""Command-line interface."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import httpx
import typer
from rich.console import Console
from rich.table import Table

from rag.config import Settings, get_settings

if TYPE_CHECKING:
    from rag.graph.state import FinalAnswer
    from rag.ingestion.pipeline import IngestStats
    from rag.retrieval.retriever import RetrievalResult, SearchMode

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


def _start_tracing(settings: Settings) -> None:
    from rag import telemetry

    telemetry.init_tracing(settings.phoenix)


def _trace_context(settings: Settings) -> tuple[str, dict[str, Any]]:
    import getpass

    user = settings.phoenix.user_id or getpass.getuser()
    metadata = {
        "interface": "cli",
        "provider": settings.llm.provider,
        "generation_model": settings.llm.generation_model,
        "cheap_model": settings.llm.cheap_model,
        "reranker": settings.reranker.model,
    }
    return user, metadata


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
    _start_tracing(settings)
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
            root,
            retry_failed=retry_failed,
            force=force,
            progress=lambda stage, n: bar.advance(tasks[stage], n),
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


@app.command()
def search(
    query: Annotated[list[str], typer.Argument(help="Query; extra arguments are fused as query variants.")],
    top_k: Annotated[int | None, typer.Option("--top-k", "-k", help="Results to keep.")] = None,
    mode: Annotated[str | None, typer.Option(help="hybrid | dense | sparse")] = None,
    rerank: Annotated[bool, typer.Option("--rerank/--no-rerank")] = True,
    file: Annotated[list[str] | None, typer.Option(help="Restrict to file name(s).")] = None,
    folder: Annotated[Path | None, typer.Option(help="Restrict to a folder (and subfolders).")] = None,
    after: Annotated[
        datetime | None,
        typer.Option(help="Document date on/after (YYYY-MM-DD; from name, content, else mtime)."),
    ] = None,
    before: Annotated[datetime | None, typer.Option(help="Document date on/before (YYYY-MM-DD).")] = None,
    show_context: Annotated[bool, typer.Option(help="Print the expanded passages sent to the LLM.")] = False,
) -> None:
    """Retrieve and print ranked chunks with scores and sources."""
    from rag.factory import build_retriever
    from rag.retrieval.filters import SearchFilters

    search_mode = _search_mode(mode)
    filters = SearchFilters(
        file_names=tuple(file or ()),
        folder=folder.resolve() if folder else None,
        # CLI dates are calendar days: compare in UTC like the stored document dates.
        date_after=after.replace(tzinfo=UTC) if after else None,
        date_before=before.replace(tzinfo=UTC) if before else None,
    )
    settings = get_settings()
    _start_tracing(settings)
    retriever = build_retriever(settings)
    result = retriever.retrieve(
        query,
        filters=None if filters.is_empty() else filters,
        mode=search_mode,
        rerank=rerank,
        top_k=top_k,
    )
    _print_search(result, show_context=show_context)


_SEARCH_MODES: dict[str, SearchMode] = {"hybrid": "hybrid", "dense": "dense", "sparse": "sparse"}


def _search_mode(mode: str | None) -> SearchMode | None:
    if mode is None:
        return None
    if mode not in _SEARCH_MODES:
        raise typer.BadParameter("mode must be hybrid, dense or sparse")
    return _SEARCH_MODES[mode]


def _print_search(result: RetrievalResult, *, show_context: bool) -> None:
    from rag.retrieval.types import payload_section

    table = Table(
        "#", "score", "rrf", "file", "pages", "section", "snippet", title=" | ".join(result.queries)
    )
    for rank, c in enumerate(result.ranked, start=1):
        p = c.payload
        start, end = p.get("page_start"), p.get("page_end")
        pages = "" if start is None else (f"{start}" if start == end else f"{start}-{end}")
        section = " > ".join(payload_section(p))
        snippet = " ".join(str(p.get("text", "")).split())[:120]
        score = f"{c.rerank_score:.3f}" if c.rerank_score is not None else "-"
        table.add_row(
            str(rank), score, f"{c.fused_score:.4f}", str(p.get("file_name")), pages, section[:40], snippet
        )
    console.print(table)
    timing = ", ".join(f"{k} {v:.0f} ms" for k, v in result.timings_ms.items())
    console.print(f"[dim]{len(result.candidates)} candidates; {timing}[/dim]")
    if show_context:
        for passage in result.passages:
            where = f"{passage.file_name} p.{passage.page_start}" if passage.page_start else passage.file_name
            console.rule(f"[{passage.index}] {where} — {' > '.join(passage.section_path)}")
            console.print(passage.text)


# --- question answering ----------------------------------------------------------------------------


class _TurnPrinter:
    """Renders graph events: dim status lines, streamed answer tokens, then the final answer and sources."""

    def __init__(self) -> None:
        self.streamed = ""
        self._streaming = False

    def __call__(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "status":
            self._end_stream()
            console.print(f"[dim]… {event.get('text', '')}[/dim]")
        elif kind == "answer_start":
            self._end_stream()
            self.streamed = ""
            if int(event.get("attempt", 1)) > 1:
                console.print("[dim]↻ revised answer:[/dim]")
        elif kind == "token":
            text = str(event.get("text", ""))
            self.streamed += text
            self._streaming = True
            console.print(text, end="", markup=False, highlight=False, soft_wrap=True)

    def finish_stream(self) -> None:
        self._end_stream()

    def _end_stream(self) -> None:
        if self._streaming:
            console.print()
            self._streaming = False

    def finish(self, final: FinalAnswer) -> None:
        self._end_stream()
        # The streamed draft is already on screen; re-print only when the final text differs (cautious answer,
        # direct replies, abstentions, or invalid citation markers removed).
        if final.text.strip() != self.streamed.strip():
            console.print(final.text, markup=False, highlight=False)
        if final.status != "answered":
            console.print(f"[dim]({final.status.replace('_', ' ')})[/dim]")
        if final.citations:
            table = Table("#", "source", "pages", "section", "snippet", title="Sources", title_justify="left")
            for c in final.citations:
                pages = (
                    ""
                    if c.page_start is None
                    else (str(c.page_start) if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}")
                )
                table.add_row(f"[{c.index}]", c.file_name, pages, c.section[:40], c.snippet[:100])
            console.print(table)


def _run_question(graph: Any, question: str, thread_id: str, settings: Settings) -> None:
    import anthropic
    import openai

    from rag.graph.build import run_turn
    from rag.llm.base import LLMError

    printer = _TurnPrinter()
    try:
        user, metadata = _trace_context(settings)
        state = run_turn(
            graph, question, thread_id=thread_id, on_event=printer, user_id=user, metadata=metadata
        )
    except (LLMError, anthropic.APIError, openai.APIError) as exc:
        # Provider failures (refusals, truncation, HTTP errors after SDK retries) end the turn cleanly.
        printer.finish_stream()
        console.print(f"[red]LLM error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    final = state.get("final")
    if final is None:
        console.print("[red]No answer was produced.[/red]")
        raise typer.Exit(code=1)
    printer.finish(final)


def _qa_graph(settings: Settings) -> Any:
    from rag.factory import ConfigurationError, build_qa_graph

    _start_tracing(settings)
    try:
        return build_qa_graph(settings)
    except ConfigurationError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question about the documents.")],
    thread_id: Annotated[str | None, typer.Option(help="Continue a conversation thread.")] = None,
) -> None:
    """Answer one question with citations (streams the answer)."""
    import uuid

    thread = thread_id or uuid.uuid4().hex
    settings = get_settings()
    _run_question(_qa_graph(settings), question, thread, settings)
    console.print(f"[dim]thread: {thread}[/dim]")


@app.command()
def chat(
    thread_id: Annotated[str | None, typer.Option(help="Resume a conversation thread.")] = None,
) -> None:
    """Multi-turn conversation; follow-up questions use the conversation history."""
    import uuid

    thread = thread_id or uuid.uuid4().hex
    settings = get_settings()
    graph = _qa_graph(settings)
    console.print(f"[dim]thread {thread} — empty line, 'exit' or Ctrl-D to quit[/dim]")
    while True:
        try:
            question = console.input("[bold]you>[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if question.lower() in ("", "exit", "quit"):
            break
        _run_question(graph, question, thread, settings)
