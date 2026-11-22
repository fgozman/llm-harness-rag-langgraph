"""Command-line interface."""

from __future__ import annotations

import json

import httpx
import typer
from rich.console import Console
from rich.table import Table

from rag.config import Settings, get_settings

app = typer.Typer(help="RAG over a local document folder.", no_args_is_help=True)
console = Console()

_HEALTH_TIMEOUT_S = 3.0


@app.callback()
def _main() -> None:
    """Keeps `rag <command>` as a command group even while only a few commands exist."""


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
