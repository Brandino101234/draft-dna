from datetime import timedelta
from typing import Any

import pytest
import requests

from draft_dna.config import Settings
from draft_dna.ingest.fetcher import (
    Fetcher,
    NotFoundError,
    RateLimitedError,
    cached_json,
)


class FakeResponse:
    def __init__(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None):
        self.status_code = status
        self.content = body
        self.headers = headers or {}


class FakeSession(requests.Session):
    def __init__(self, responses: list[FakeResponse]):
        super().__init__()
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:  # type: ignore[override]
        self.calls.append(url)
        return self.responses.pop(0)


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def make(settings: Settings, responses: list[FakeResponse], source: str = "sports_reference"):
    clock = FakeClock()
    session = FakeSession(responses)
    f = Fetcher(source, settings=settings, session=session, sleep=clock.sleep, now=clock.now)
    return f, session, clock


def test_second_get_is_served_from_cache(settings: Settings) -> None:
    f, session, _ = make(settings, [FakeResponse(200, b"<html>draft</html>")])
    assert f.get("https://x/draft") == b"<html>draft</html>"
    assert f.get("https://x/draft") == b"<html>draft</html>"
    assert len(session.calls) == 1
    # A fresh fetcher (new process) also hits the disk cache.
    f2, session2, _ = make(settings, [])
    assert f2.get("https://x/draft") == b"<html>draft</html>"
    assert session2.calls == []


def test_requests_are_spaced_by_rate_limit(settings: Settings) -> None:
    f, _, clock = make(settings, [FakeResponse(200, b"a"), FakeResponse(200, b"b")])
    f.get("https://x/a")
    f.get("https://x/b")
    assert clock.slept == [pytest.approx(f.min_interval)]


def test_404_is_cached_and_not_refetched(settings: Settings) -> None:
    f, session, _ = make(settings, [FakeResponse(404)])
    with pytest.raises(NotFoundError):
        f.get("https://x/missing")
    with pytest.raises(NotFoundError):
        f.get("https://x/missing")
    assert len(session.calls) == 1


def test_sports_reference_429_stops_immediately(settings: Settings) -> None:
    f, session, _ = make(settings, [FakeResponse(429), FakeResponse(200, b"x")])
    with pytest.raises(RateLimitedError):
        f.get("https://x/a")
    assert len(session.calls) == 1


def test_other_sources_retry_on_5xx(settings: Settings) -> None:
    f, session, _ = make(settings, [FakeResponse(503), FakeResponse(200, b"ok")], source="espn")
    assert f.get("https://x/a") == b"ok"
    assert len(session.calls) == 2


def test_max_age_forces_refresh_of_stale_pages(settings: Settings) -> None:
    f, session, _ = make(settings, [FakeResponse(200, b"v1"), FakeResponse(200, b"v2")])
    f.get("https://x/season")
    assert f.get("https://x/season", max_age=timedelta(0)) == b"v2"
    assert len(session.calls) == 2


def test_cached_json_calls_library_once(settings: Settings) -> None:
    calls = []

    def call() -> dict[str, int]:
        calls.append(1)
        return {"a": 1}

    noop = lambda s: None  # noqa: E731
    assert cached_json("nba_api", "draft_history", call, settings=settings, sleep=noop) == {"a": 1}
    assert cached_json("nba_api", "draft_history", call, settings=settings, sleep=noop) == {"a": 1}
    assert len(calls) == 1


def test_cached_json_retries_transient_failures(settings: Settings) -> None:
    attempts = []

    def flaky() -> dict[str, int]:
        attempts.append(1)
        if len(attempts) < 3:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")  # throttled HTML
        return {"ok": 1}

    noop = lambda s: None  # noqa: E731
    assert cached_json("nba_api", "flaky", flaky, settings=settings, sleep=noop) == {"ok": 1}
    assert len(attempts) == 3
