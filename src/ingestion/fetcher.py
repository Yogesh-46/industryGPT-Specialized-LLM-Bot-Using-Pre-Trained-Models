"""HTTP fetch helpers with retries and polite delays."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    url: str
    status_code: int
    content: bytes
    content_type: str | None
    final_url: str
    elapsed_seconds: float


class FetchError(RuntimeError):
    """Raised when an HTTP fetch fails after retries."""


class DocumentFetcher:
    """Polite HTTP client for curated documentation URLs."""

    def __init__(
        self,
        user_agent: str,
        delay_seconds: float = 1.5,
        max_retries: int = 3,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self._last_request_at: float | None = None

    def _throttle(self) -> None:
        if self.delay_seconds <= 0:
            return
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def fetch(self, url: str) -> FetchResult:
        """Fetch a URL with exponential backoff retries."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((requests.RequestException, FetchError)),
        )
        def _once() -> FetchResult:
            self._throttle()
            started = time.monotonic()
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            self._last_request_at = time.monotonic()
            elapsed = self._last_request_at - started

            if response.status_code in {429, 500, 502, 503, 504}:
                raise FetchError(f"Transient HTTP {response.status_code} for {url}")

            return FetchResult(
                url=url,
                status_code=response.status_code,
                content=response.content or b"",
                content_type=response.headers.get("Content-Type"),
                final_url=str(response.url),
                elapsed_seconds=elapsed,
            )

        try:
            return _once()
        except Exception as exc:  # noqa: BLE001
            logger.error("Fetch failed for %s: %s", url, exc)
            raise FetchError(str(exc)) from exc
