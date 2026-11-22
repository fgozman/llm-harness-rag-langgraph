"""Polite HTTP client: identifying User-Agent, per-host rate limits, retries, and a permanent disk cache.

The cache makes the corpus build idempotent and re-runs offline; pass refresh=True to re-fetch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx

log = logging.getLogger(__name__)

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
# A list of pairs allows repeated keys (e.g. Federal Register's `fields[]=a&fields[]=b`).
Params = Mapping[str, str] | Sequence[tuple[str, str]]


class NotFoundError(Exception):
    """404/410: the resource does not exist (expected for optional files)."""


class PoliteClient:
    def __init__(
        self,
        cache_dir: Path,
        *,
        user_agent: str,
        refresh: bool = False,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 4,
        timeout_s: float = 60.0,
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir = cache_dir
        self._refresh = refresh
        self._clock = clock
        self._sleep = sleep
        self._max_retries = max_retries
        self._min_interval: dict[str, float] = {}
        self._last_request: dict[str, float] = {}
        self._host_headers: dict[str, dict[str, str]] = {}
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            transport=transport,
            timeout=timeout_s,
            follow_redirects=True,
        )

    def set_min_interval(self, host: str, seconds: float) -> None:
        self._min_interval[host] = seconds

    def add_header(self, host: str, name: str, value: str) -> None:
        self._host_headers.setdefault(host, {})[name] = value

    def get_bytes(self, url: str, *, params: Params | None = None) -> bytes:
        pairs = sorted(params.items()) if isinstance(params, Mapping) else list(params or [])
        full_url = f"{url}?{urlencode(pairs)}" if pairs else url
        cached = self._cache_dir / hashlib.sha256(full_url.encode()).hexdigest()
        if cached.exists() and not self._refresh:
            return cached.read_bytes()
        body = self._fetch(full_url)
        cached.write_bytes(body)
        return body

    def get_text(self, url: str, *, params: Params | None = None) -> str:
        return self.get_bytes(url, params=params).decode("utf-8", errors="replace")

    def get_json(self, url: str, *, params: Params | None = None) -> Any:
        return json.loads(self.get_bytes(url, params=params))

    def _throttle(self, host: str) -> None:
        interval = self._min_interval.get(host, 0.0)
        last = self._last_request.get(host)
        if last is not None and interval > 0:
            wait = interval - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_request[host] = self._clock()

    def _fetch(self, url: str) -> bytes:
        host = urlsplit(url).hostname or ""
        for attempt in range(self._max_retries + 1):
            self._throttle(host)
            try:
                response = self._client.get(url, headers=self._host_headers.get(host))
            except httpx.TransportError as exc:
                if attempt == self._max_retries:
                    raise
                log.warning("transport error on %s (%s); retrying", url, exc)
                self._sleep(2.0**attempt)
                continue
            if response.status_code in (404, 410):
                raise NotFoundError(url)
            if response.status_code in _RETRY_STATUS and attempt < self._max_retries:
                retry_after = response.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else 2.0**attempt
                log.warning("HTTP %s on %s; retrying in %.0fs", response.status_code, url, delay)
                self._sleep(delay)
                continue
            response.raise_for_status()
            return response.content
        raise RuntimeError("unreachable")  # loop either returns or raises
