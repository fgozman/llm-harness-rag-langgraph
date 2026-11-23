"""Command-line interface."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from rag.config import Settings, get_settings

if TYPE_CHECKING:
    from rag.evaluation.experiment import ExperimentReport
    from rag.graph.state import FinalAnswer
    from rag.ingestion.pipeline import IngestStats
    from rag.retrieval.retriever import RetrievalResult, SearchMode

app = typer.Typer(help="RAG over a local document folder.", no_args_is_help=True)
console = Console()

log = logging.getLogger(__name__)


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False) -> None:
    """RAG over a local document folder."""
    from dotenv import load_dotenv

    from rag import logs

    # LangChain provider integrations read their API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...) from the
    # environment; export `.env` so keys kept there work for any provider. Real environment variables win.
    load_dotenv(".env", override=False)
    settings = get_settings()
    logs.configure(level=settings.log.level, fmt=settings.log.format, verbose=verbose)


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


@app.command()
def doctor() -> None:
    """Check service reachability; exit 1 if a required service is down."""
    from rag.services import service_checks

    table = Table("service", "url", "status")
    checks = service_checks(get_settings())
    for c in checks:
        table.add_row(c.name, c.url, c.status)
    console.print(table)
    # Phoenix is optional (the app must keep working without tracing), so only Qdrant is fatal.
    if any(c.failed for c in checks):
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


def _run_question(graph: Any, question: str, thread_id: str, settings: Settings) -> bool:
    """Run one turn and print it; False when it timed out, the LLM provider failed or no answer came back."""
    from rag.graph.build import turn_stream
    from rag.graph.stream import TurnTimeoutError
    from rag.llm.base import LLMError

    printer = _TurnPrinter()
    user, metadata = _trace_context(settings)
    stream = turn_stream(
        graph,
        question,
        thread_id=thread_id,
        timeout_s=settings.graph.turn_timeout_s,
        user_id=user,
        metadata=metadata,
    )
    try:
        for event in stream.events():
            printer(event)
    except TurnTimeoutError as exc:
        printer.finish_stream()
        console.print(f"[red]Timed out:[/red] {exc} (graph.turn_timeout_s)")
        return False
    except LLMError as exc:
        # Provider failures (refusals, truncation, HTTP errors after SDK retries) end the turn cleanly.
        printer.finish_stream()
        console.print(f"[red]LLM error:[/red] {exc}")
        return False
    except KeyboardInterrupt:
        stream.cancel()
        printer.finish_stream()
        raise
    state = stream.result or {}
    final = state.get("final")
    if final is None:
        console.print("[red]No answer was produced.[/red]")
        return False
    printer.finish(final)
    return True


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
    if not _run_question(_qa_graph(settings), question, thread, settings):
        raise typer.Exit(code=1)
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
        # A failed turn is reported and the conversation continues; the user can retry or ask something else.
        _run_question(graph, question, thread, settings)


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Bind address (default api.host, loopback).")] = None,
    port: Annotated[int | None, typer.Option(help="Port (default api.port).")] = None,
) -> None:
    """Run the HTTP API: POST /ask (SSE), POST /ingest, GET /health."""
    import uvicorn

    from rag.api.app import create_app

    settings = get_settings()
    bind_host, bind_port = host or settings.api.host, port or settings.api.port
    if bind_host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("the API has no authentication and is bound to %s", bind_host)
    # Logging is already configured (rag.logs); uvicorn must not install its own handlers. Access lines come
    # from our middleware (with request ids), so uvicorn's access log is off.
    uvicorn.run(create_app(settings), host=bind_host, port=bind_port, log_config=None, access_log=False)


# --- evaluation --------------------------------------------------------------------------------------

golden_app = typer.Typer(help="Golden evaluation set (generate, upload to Phoenix).", no_args_is_help=True)
app.add_typer(golden_app, name="golden")


@golden_app.command("generate")
def golden_generate() -> None:
    """Generate eval/golden.jsonl from the indexed corpus (LLM-written, auto-verified; needs human review)."""
    from collections import Counter

    from rag.evaluation.runner import generate_golden

    settings = get_settings()
    _start_tracing(settings)
    items, cost = generate_golden(settings)
    table = Table(
        "category", "items", title=f"Golden set: {len(items)} items -> {settings.evaluation.golden_path}"
    )
    for category, count in sorted(Counter(i.category for i in items).items()):
        table.add_row(category, str(count))
    console.print(table)
    console.print(
        f"[dim]generation cost ${cost:.2f}; all items have reviewed=false until a human checks them[/dim]"
    )


@golden_app.command("upload")
def golden_upload() -> None:
    """Upload eval/golden.jsonl as a Phoenix dataset (content-addressed; re-uploading is a no-op)."""
    from rag.evaluation.runner import upload_golden

    name, count = upload_golden(get_settings())
    console.print(f"Phoenix dataset [bold]{name}[/bold] ({count} examples)")


_EVAL_VARIANT_HELP = "name[:key=value,...] settings overrides, e.g. ctx:qdrant.collection=x,state_dir=y"


@app.command("eval")
def evaluate(
    no_retrieval: Annotated[
        bool, typer.Option("--no-retrieval", help="Closed-book baseline (no retrieval).")
    ] = False,
    variant: Annotated[str, typer.Option(help=_EVAL_VARIANT_HELP)] = "baseline",
    category: Annotated[list[str] | None, typer.Option(help="Only these golden categories.")] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N examples.")] = None,
) -> None:
    """Run the golden set as a Phoenix experiment and print the report."""
    from rag.evaluation.runner import CLOSED_BOOK, latest_report, leaky_questions, run_evaluation

    settings = get_settings()
    report = run_evaluation(
        settings, variant=variant, closed_book=no_retrieval, categories=category or (), limit=limit
    )
    baseline = None if no_retrieval else latest_report(settings.evaluation.results_dir, CLOSED_BOOK)
    _print_report(report, baseline)
    source = report if no_retrieval else baseline
    leaky = leaky_questions(source) if source is not None else []
    if leaky:
        console.print(
            f"[yellow]Leaky questions (answered correctly without retrieval):[/yellow] {', '.join(leaky)}"
        )


@app.command()
def compare(
    variant: Annotated[list[str], typer.Option(help=_EVAL_VARIANT_HELP + " (repeat; at least two)")],
    category: Annotated[list[str] | None, typer.Option(help="Only these golden categories.")] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N examples.")] = None,
) -> None:
    """Run several configurations as separate Phoenix experiments and compare them side by side."""
    from rag.evaluation.runner import run_evaluation

    if len(variant) < 2:
        raise typer.BadParameter("give at least two --variant values")
    settings = get_settings()
    reports = [run_evaluation(settings, variant=v, categories=category or (), limit=limit) for v in variant]
    table = Table("metric", *(r.metadata["variant"] for r in reports), title="Comparison")
    for evaluator in reports[0].evaluators():
        table.add_row(evaluator, *(_fmt(r.mean(evaluator)) for r in reports))
    # Failed examples have no scores, so a variant's means silently cover fewer questions without this row.
    table.add_row("failed examples", *(f"{len(r.failed())}/{len(r.rows)}" for r in reports))
    console.print(table)
    for r in reports:
        console.print(f"[dim]{r.metadata['variant']}: experiment {r.name}[/dim]")


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _print_report(report: ExperimentReport, closed_book: ExperimentReport | None) -> None:
    table = Table("evaluator", "mean", "n", title=f"Experiment {report.name}")
    for evaluator in report.evaluators():
        table.add_row(evaluator, _fmt(report.mean(evaluator)), str(len(report.scores(evaluator))))
    console.print(table)
    if failed := report.failed():
        console.print(
            f"[yellow]{len(failed)}/{len(report.rows)} examples failed (no scores; excluded from the means):"
            f"[/yellow] {', '.join(failed)}"
        )
    p50, p95 = report.percentile("latency_s", 0.5), report.percentile("latency_s", 0.95)
    cost = _fmt(report.mean("cost_usd"))
    console.print(f"[dim]latency p50 {_fmt(p50)}s, p95 {_fmt(p95)}s; cost/query {cost} USD[/dim]")
    rag = report.by_category("correctness")
    book = closed_book.by_category("correctness") if closed_book else {}
    this = "closed-book" if report.metadata.get("closed_book") else "RAG"
    by_cat = Table(
        "category", f"correctness ({this})", "closed-book baseline", title="Correctness by category"
    )
    for cat in sorted(set(rag) | set(book)):
        by_cat.add_row(cat, _fmt(rag.get(cat)), _fmt(book.get(cat)))
    console.print(by_cat)
