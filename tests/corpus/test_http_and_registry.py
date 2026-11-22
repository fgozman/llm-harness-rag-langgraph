from pathlib import Path

import httpx
import pytest

from corpus_builder.http import NotFoundError, PoliteClient
from corpus_builder.registry import Registry, SourceRecord


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def client(
    tmp_path: Path, handler: httpx.MockTransport, clock: FakeClock, refresh: bool = False
) -> PoliteClient:
    return PoliteClient(
        tmp_path / "cache",
        user_agent="test-agent a@b.c",
        transport=handler,
        clock=clock.time,
        sleep=clock.sleep,
        refresh=refresh,
    )


def test_cache_makes_reruns_offline(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        assert request.headers["User-Agent"] == "test-agent a@b.c"
        return httpx.Response(200, json={"ok": True})

    clock = FakeClock()
    first = client(tmp_path, httpx.MockTransport(handler), clock)
    assert first.get_json("https://x.test/a", params={"q": "1"}) == {"ok": True}
    second = client(tmp_path, httpx.MockTransport(handler), clock)
    assert second.get_json("https://x.test/a", params={"q": "1"}) == {"ok": True}
    assert len(calls) == 1
    client(tmp_path, httpx.MockTransport(handler), clock, refresh=True).get_json(
        "https://x.test/a", params={"q": "1"}
    )
    assert len(calls) == 2


def test_per_host_rate_limit(tmp_path: Path) -> None:
    clock = FakeClock()
    c = client(tmp_path, httpx.MockTransport(lambda r: httpx.Response(200, text="x")), clock)
    c.set_min_interval("slow.test", 3.0)
    for i in range(3):
        c.get_bytes(f"https://slow.test/{i}")
    assert clock.sleeps == [3.0, 3.0]
    c.get_bytes("https://fast.test/1")
    assert clock.sleeps == [3.0, 3.0]


def test_retries_on_429_honoring_retry_after(tmp_path: Path) -> None:
    responses = [
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(503),
        httpx.Response(200, text="ok"),
    ]
    clock = FakeClock()
    c = client(tmp_path, httpx.MockTransport(lambda r: responses.pop(0)), clock)
    assert c.get_text("https://x.test/r") == "ok"
    assert clock.sleeps[0] == 7.0 and len(clock.sleeps) == 2


def test_404_raises_not_found_and_is_not_cached(tmp_path: Path) -> None:
    clock = FakeClock()
    c = client(tmp_path, httpx.MockTransport(lambda r: httpx.Response(404)), clock)
    with pytest.raises(NotFoundError):
        c.get_bytes("https://x.test/missing")
    assert not any((tmp_path / "cache").glob("*"))


def test_registry_roundtrip_upsert_and_markdown(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    reg = Registry(path)
    reg.upsert(
        SourceRecord(
            path="sec/AMD_10-Q.htm",
            source="sec",
            title="AMD 10-Q",
            url="https://sec.gov/x",
            published="2026-05-06",
            license="SEC public filing",
            pre_cutoff=True,
        )
    )
    reg.upsert(
        SourceRecord(
            path="canary/a.docx",
            source="canary",
            title="Fiction",
            url="local",
            published="2026-11-22",
            license="CC0",
            fiction=True,
        )
    )
    reg.upsert(
        SourceRecord(
            path="canary/a.docx",
            source="canary",
            title="Fiction v2",
            url="local",
            published="x",
            license="CC0",
            fiction=True,
        )
    )
    reg.save()
    loaded = Registry(path)
    assert [r.title for r in loaded.records] == ["Fiction v2", "AMD 10-Q"]
    md = loaded.render_markdown()
    assert "pre_cutoff: true" in md and "FICTION" in md
    assert "| sec/AMD_10-Q.htm |" in md and "https://sec.gov/x" in md


def test_host_headers_only_sent_to_that_host(tmp_path: Path) -> None:
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.host] = request.headers.get("Authorization")
        return httpx.Response(200, text="x")

    c = client(tmp_path, httpx.MockTransport(handler), FakeClock())
    c.add_header("api.github.com", "Authorization", "Bearer t")
    c.get_bytes("https://api.github.com/x")
    c.get_bytes("https://other.test/x")
    assert seen == {"api.github.com": "Bearer t", "other.test": None}


def test_repeated_query_params(tmp_path: Path) -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, text="x")

    c = client(tmp_path, httpx.MockTransport(handler), FakeClock())
    c.get_bytes("https://x.test/a", params=[("fields[]", "title"), ("fields[]", "url"), ("q", "1")])
    assert urls == ["https://x.test/a?fields%5B%5D=title&fields%5B%5D=url&q=1"]
