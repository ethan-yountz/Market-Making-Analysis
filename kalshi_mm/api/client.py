"""Kalshi REST API client.

Optional RSA-PSS request signing (required for portfolio + /historical
endpoints; public market-data reads work unauthenticated), token-bucket rate
limiting, retry with backoff on 429/5xx, and cursor pagination.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

DEFAULT_BASE_URL = "https://api.elections.kalshi.com"
API_PREFIX = "/trade-api/v2"
WS_PATH = "/trade-api/ws/v2"

log = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe token bucket."""

    def __init__(self, rate_per_s: float = 8.0, burst: int = 8):
        self.rate = rate_per_s
        self.capacity = burst
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            time.sleep(wait)


class KalshiAuth:
    """Signs requests per Kalshi's scheme: RSA-PSS(SHA256) over
    ``{timestamp_ms}{METHOD}{path}`` where path includes /trade-api/v2 but
    excludes the query string."""

    def __init__(self, key_id: str, private_key_pem: bytes):
        self.key_id = key_id
        self._key = serialization.load_pem_private_key(private_key_pem, password=None)

    @classmethod
    def from_files(cls, key_id_file: str | Path, private_key_file: str | Path) -> "KalshiAuth":
        key_id = Path(key_id_file).read_text(encoding="utf-8").strip()
        pem = Path(private_key_file).read_bytes()
        return cls(key_id, pem)

    def headers(self, method: str, path: str) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + path).encode("utf-8")
        sig = self._key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    def ws_headers(self) -> dict[str, str]:
        return self.headers("GET", WS_PATH)


def load_default_auth(secrets_dir: str | Path = "secrets") -> KalshiAuth | None:
    """Load credentials, in priority order:

    1. env ``KALSHI_API_KEY_ID`` + ``KALSHI_PRIVATE_KEY`` (raw PEM inline —
       used on hosts like Railway with no persistent filesystem; literal
       ``\\n`` escapes are accepted),
    2. env ``KALSHI_API_KEY_ID`` + ``KALSHI_PRIVATE_KEY_PATH`` (file path),
    3. ``secrets/kalshi_key_id.txt`` + ``secrets/kalshi_private_key.pem``.

    Returns None if absent — public endpoints still work unauthenticated."""
    key_id = os.environ.get("KALSHI_API_KEY_ID")
    pem_env = os.environ.get("KALSHI_PRIVATE_KEY")
    if key_id and pem_env:
        return KalshiAuth(key_id.strip(), pem_env.replace("\\n", "\n").encode("utf-8"))
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    if key_id and key_path and Path(key_path).exists():
        return KalshiAuth(key_id.strip(), Path(key_path).read_bytes())
    sdir = Path(secrets_dir)
    id_file, pem_file = sdir / "kalshi_key_id.txt", sdir / "kalshi_private_key.pem"
    if id_file.exists() and pem_file.exists():
        return KalshiAuth.from_files(id_file, pem_file)
    return None


class KalshiClient:
    def __init__(
        self,
        auth: KalshiAuth | None = None,
        base_url: str = DEFAULT_BASE_URL,
        rate_per_s: float = 8.0,
        timeout: float = 30.0,
        max_retries: int = 6,
    ):
        self.auth = auth
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._limiter = RateLimiter(rate_per_s=rate_per_s, burst=max(2, int(rate_per_s)))
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "kalshi-mm-research/0.1"

    # ------------------------------------------------------------------ core

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        """GET {base}{/trade-api/v2}{path} with throttling, signing, retries."""
        full_path = path if path.startswith(API_PREFIX) else API_PREFIX + path
        url = self.base_url + full_path
        params = {k: v for k, v in (params or {}).items() if v is not None}
        backoff = 1.0
        for attempt in range(self.max_retries):
            self._limiter.acquire()
            headers = self.auth.headers("GET", full_path) if self.auth else {}
            resp = self._session.get(url, params=params, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else backoff
                log.warning("GET %s -> %s, retrying in %.1fs", full_path, resp.status_code, wait)
                time.sleep(wait)
                backoff = min(backoff * 2, 30.0)
                continue
            raise KalshiAPIError(resp.status_code, resp.text[:500], full_path)
        raise KalshiAPIError(resp.status_code, resp.text[:500], full_path)

    def paginate(
        self,
        path: str,
        items_key: str,
        params: dict[str, Any] | None = None,
        max_pages: int | None = None,
    ) -> Iterator[dict]:
        """Iterate items across cursor-paginated responses."""
        params = dict(params or {})
        pages = 0
        while True:
            data = self.get(path, params)
            for item in data.get(items_key) or []:
                yield item
            cursor = data.get("cursor")
            pages += 1
            if not cursor or (max_pages is not None and pages >= max_pages):
                return
            params["cursor"] = cursor

    # ------------------------------------------------------- market data

    def get_series(self, series_ticker: str) -> dict:
        return self.get(f"/series/{series_ticker}")

    def get_markets(
        self,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        limit: int = 1000,
    ) -> Iterator[dict]:
        return self.paginate(
            "/markets",
            "markets",
            {
                "series_ticker": series_ticker,
                "event_ticker": event_ticker,
                "status": status,
                "min_close_ts": min_close_ts,
                "max_close_ts": max_close_ts,
                "limit": limit,
            },
        )

    def get_market(self, ticker: str) -> dict:
        return self.get(f"/markets/{ticker}")["market"]

    def get_event(self, event_ticker: str, with_nested_markets: bool = True) -> dict:
        return self.get(f"/events/{event_ticker}", {"with_nested_markets": with_nested_markets})

    def get_trades(
        self,
        ticker: str,
        min_ts: int | None = None,
        max_ts: int | None = None,
        limit: int = 1000,
    ) -> Iterator[dict]:
        return self.paginate(
            "/markets/trades",
            "trades",
            {"ticker": ticker, "min_ts": min_ts, "max_ts": max_ts, "limit": limit},
        )

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> list[dict]:
        data = self.get(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
        )
        return data.get("candlesticks", [])

    def get_orderbook(self, ticker: str, depth: int | None = None) -> dict:
        return self.get(f"/markets/{ticker}/orderbook", {"depth": depth})

    # -------------------------------------------------------- historical

    def get_historical_cutoff(self) -> dict:
        return self.get("/historical/cutoff")

    def get_historical_markets(
        self,
        series_ticker: str | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        limit: int = 1000,
    ) -> Iterator[dict]:
        return self.paginate(
            "/historical/markets",
            "markets",
            {
                "series_ticker": series_ticker,
                "min_close_ts": min_close_ts,
                "max_close_ts": max_close_ts,
                "limit": limit,
            },
        )

    def get_historical_candlesticks(
        self, ticker: str, start_ts: int, end_ts: int, period_interval: int = 1
    ) -> list[dict]:
        data = self.get(
            f"/historical/markets/{ticker}/candlesticks",
            {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
        )
        return data.get("candlesticks", [])

    def get_historical_trades(
        self,
        ticker: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        limit: int = 1000,
    ) -> Iterator[dict]:
        return self.paginate(
            "/historical/trades",
            "trades",
            {"ticker": ticker, "min_ts": min_ts, "max_ts": max_ts, "limit": limit},
        )


class KalshiAPIError(RuntimeError):
    def __init__(self, status: int, body: str, path: str):
        super().__init__(f"Kalshi API {status} on {path}: {body}")
        self.status = status
        self.body = body
        self.path = path
