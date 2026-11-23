"""Reachability checks for the Docker services (used by `rag doctor` and the API's /health)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx

from rag.config import Settings

_TIMEOUT_S = 3.0

Probe = Callable[[str], str]


def probe(url: str, timeout_s: float = _TIMEOUT_S) -> str:
    """'ok', or 'down (<reason>)'."""
    try:
        response = httpx.get(url, timeout=timeout_s)
    except httpx.HTTPError as exc:
        return f"down ({type(exc).__name__})"
    return "ok" if response.status_code == 200 else f"down (HTTP {response.status_code})"


@dataclass(frozen=True, slots=True)
class ServiceCheck:
    name: str
    url: str
    status: str
    # Phoenix is optional: the app must keep working without tracing.
    required: bool

    @property
    def failed(self) -> bool:
        return self.required and self.status != "ok"


def service_checks(settings: Settings, check: Probe = probe) -> list[ServiceCheck]:
    qdrant = settings.qdrant.url
    phoenix = settings.phoenix.collector_endpoint
    phoenix_status = check(f"{phoenix}/healthz") if settings.phoenix.enabled else "disabled"
    return [
        ServiceCheck("qdrant", qdrant, check(f"{qdrant}/readyz"), required=True),
        ServiceCheck("phoenix", phoenix, phoenix_status, required=False),
    ]
