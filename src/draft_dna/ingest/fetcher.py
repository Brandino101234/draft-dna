"""Cached, rate-limited HTTP fetching shared by every ingest module.

Guarantees:
- Every successful response is cached on disk under data/raw/http/<source>/ and is
  never fetched again (unless the caller passes `max_age`, used only for pages that
  change during the current season).
- Requests to one source are spaced at least `min_seconds_between_requests` apart,
  across processes (the last-request time lives in a file).
- 404s are cached too, so a missing page is not re-requested on every rebuild.
- A 429 from Sports-Reference stops the run immediately: retrying gets the client
  blocked for hours.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from draft_dna.config import Settings, get_settings
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_USER_AGENT = "draft-dna research project (personal, non-commercial)"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
NO_RETRY_429_SOURCES = {"sports_reference"}


class FetchError(RuntimeError):
    pass


class NotFoundError(FetchError):
    pass


class RateLimitedError(FetchError):
    pass


@dataclass(frozen=True)
class CacheEntry:
    body_path: Path
    meta_path: Path

    def exists(self) -> bool:
        return self.meta_path.exists()

    def meta(self) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(self.meta_path.read_text())
        return result

    def is_fresh(self, max_age: timedelta | None) -> bool:
        if not self.exists():
            return False
        if max_age is None:
            return True
        fetched = datetime.fromisoformat(self.meta()["fetched_at"])
        return datetime.now(UTC) - fetched < max_age


def cache_key(url: str, params: dict[str, Any] | None = None) -> str:
    canonical = url + ("?" + json.dumps(params, sort_keys=True) if params else "")
    return hashlib.sha1(canonical.encode()).hexdigest()


class Fetcher:
    def __init__(
        self,
        source: str,
        settings: Settings | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
    ) -> None:
        load_dotenv()
        self.settings = settings or get_settings()
        if source not in self.settings.rate_limits:
            raise ValueError(f"no rate limit configured for source {source!r}")
        self.source = source
        self.min_interval = self.settings.rate_limits[source].min_seconds_between_requests
        self.root = self.settings.paths.raw / "http" / source
        self.root.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = os.getenv("HTTP_USER_AGENT", DEFAULT_USER_AGENT)
        self._sleep = sleep
        self._now = now
        self.network_requests = 0

    # -- cache -------------------------------------------------------------------
    def entry(
        self, url: str, params: dict[str, Any] | None = None, ext: str = "html"
    ) -> CacheEntry:
        key = cache_key(url, params)
        d = self.root / key[:2]
        return CacheEntry(d / f"{key}.{ext}", d / f"{key}.meta.json")

    def is_cached(self, url: str, params: dict[str, Any] | None = None, ext: str = "html") -> bool:
        return self.entry(url, params, ext).exists()

    def _write(
        self, entry: CacheEntry, url: str, params: dict[str, Any] | None, status: int, body: bytes
    ) -> None:
        entry.body_path.parent.mkdir(parents=True, exist_ok=True)
        entry.body_path.write_bytes(body)
        meta = {
            "source": self.source,
            "url": url,
            "params": params,
            "status": status,
            "fetched_at": datetime.now(UTC).isoformat(),
            "bytes": len(body),
        }
        entry.meta_path.write_text(json.dumps(meta))
        with (self.root.parent / "fetch_log.jsonl").open("a") as f:
            f.write(json.dumps(meta) + "\n")

    # -- rate limiting ------------------------------------------------------------
    def _wait_turn(self) -> None:
        stamp = self.root / ".last_request"
        if stamp.exists():
            elapsed = self._now() - float(stamp.read_text() or 0)
            if elapsed < self.min_interval:
                self._sleep(self.min_interval - elapsed)
        stamp.write_text(str(self._now()))

    # -- public API -----------------------------------------------------------------
    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        ext: str = "html",
        max_age: timedelta | None = None,
    ) -> bytes:
        """Return the response body, from cache when possible.

        Raises NotFoundError for (cached or live) 404s.
        """
        entry = self.entry(url, params, ext)
        if entry.is_fresh(max_age):
            if entry.meta()["status"] == 404:
                raise NotFoundError(url)
            return entry.body_path.read_bytes()

        http = self.settings.http
        for attempt in range(http.max_retries + 1):
            self._wait_turn()
            self.network_requests += 1
            try:
                resp = self.session.get(url, params=params, timeout=http.timeout_seconds)
            except requests.RequestException as exc:
                log.warning("%s: %s (attempt %d)", url, exc, attempt + 1)
                self._sleep(http.backoff_seconds * (attempt + 1))
                continue
            if resp.status_code == 200:
                self._write(entry, url, params, 200, resp.content)
                log.debug("fetched %s", url)
                return resp.content
            if resp.status_code == 404:
                self._write(entry, url, params, 404, b"")
                raise NotFoundError(url)
            if resp.status_code == 429 and self.source in NO_RETRY_429_SOURCES:
                raise RateLimitedError(
                    f"{self.source} returned 429 for {url}; stopping to avoid a ban. "
                    "Wait at least an hour before resuming (cached pages are kept)."
                )
            if resp.status_code in RETRYABLE_STATUS:
                wait = float(resp.headers.get("Retry-After", http.backoff_seconds * (attempt + 1)))
                log.warning("%s: HTTP %d, retrying in %.0fs", url, resp.status_code, wait)
                self._sleep(wait)
                continue
            raise FetchError(f"{url}: HTTP {resp.status_code}")
        raise FetchError(f"{url}: gave up after {http.max_retries + 1} attempts")


def cached_json(
    source: str,
    key: str,
    call: Callable[[], Any],
    settings: Settings | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Cache the JSON result of a client-library call (e.g. nba_api) under `key`.

    The library makes its own HTTP request; we provide the caching and spacing.
    """
    fetcher = Fetcher(source, settings=settings, sleep=sleep)
    entry = fetcher.entry(key, None, ext="json")
    if entry.exists():
        return json.loads(entry.body_path.read_text())
    http = fetcher.settings.http
    for attempt in range(http.max_retries + 1):
        fetcher._wait_turn()
        fetcher.network_requests += 1
        try:
            result = call()
        except Exception as exc:  # throttling shows up as timeouts or non-JSON bodies
            wait = http.backoff_seconds * 2**attempt
            log.warning(
                "%s %s: %s (attempt %d), waiting %.0fs",
                source,
                key,
                type(exc).__name__,
                attempt + 1,
                wait,
            )
            sleep(wait)
            continue
        fetcher._write(entry, key, None, 200, json.dumps(result).encode())
        return result
    raise FetchError(f"{source} {key}: gave up after {http.max_retries + 1} attempts")
