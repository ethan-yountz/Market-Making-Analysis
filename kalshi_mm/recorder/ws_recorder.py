"""Kalshi websocket order-book recorder.

Subscribes to ``orderbook_delta`` (snapshot + deltas), ``trade``, and
``ticker_v2`` for a set of market tickers and appends every raw message —
wrapped with a local receive timestamp — to daily-rotated gzip JSONL files
under ``data/lob/YYYY-MM-DD/``.

Design notes:
- Sequence gaps on orderbook subscriptions invalidate the book; the cheapest
  correct response is a full reconnect, which yields fresh snapshots.
- The market set is rediscovered periodically; a changed set also triggers a
  clean reconnect (new subscription, new snapshots).
- Raw messages are stored verbatim (no parsing beyond what gap detection
  needs) so the on-disk format never lags the exchange's schema.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

from kalshi_mm.api.client import DEFAULT_BASE_URL, WS_PATH, KalshiAuth, KalshiClient
from kalshi_mm.recorder.discover import GAME_SERIES, discover_game_markets

log = logging.getLogger(__name__)

# NB: "ticker_v2" is not a valid Kalshi channel and triggers an "Unknown
# channel name" error that rejects the whole subscription. Valid here are
# "orderbook_delta" (snapshot + deltas) and "trade".
CHANNELS = ["orderbook_delta", "trade"]


class RotatingJsonlGzWriter:
    """Appends JSON lines to data/lob/YYYY-MM-DD/<prefix>_HHMMSS.jsonl.gz,
    starting a new file on UTC day rollover or explicit reopen."""

    def __init__(self, base_dir: str | Path, prefix: str = "kalshi_ws"):
        self.base_dir = Path(base_dir)
        self.prefix = prefix
        self._fh = None
        self._day = None
        self.lines_written = 0

    def _open(self) -> None:
        now = datetime.now(timezone.utc)
        self._day = now.strftime("%Y-%m-%d")
        day_dir = self.base_dir / self._day
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{self.prefix}_{now.strftime('%H%M%S')}.jsonl.gz"
        self._fh = gzip.open(path, "at", encoding="utf-8")
        log.info("writing to %s", path)

    def write(self, line: str) -> None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._fh is None or day != self._day:
            self.close()
            self._open()
        self._fh.write(line + "\n")
        self.lines_written += 1

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class SeqGapError(RuntimeError):
    pass


class WsRecorder:
    def __init__(
        self,
        auth: KalshiAuth | None,
        base_dir: str | Path = "data/lob",
        series: tuple[str, ...] = GAME_SERIES,
        horizon_hours: float = 36.0,
        rediscover_minutes: float = 30.0,
        ws_url: str | None = None,
    ):
        self.auth = auth
        self.writer = RotatingJsonlGzWriter(base_dir)
        self.series = series
        self.horizon_hours = horizon_hours
        self.rediscover_s = rediscover_minutes * 60.0
        self.ws_url = ws_url or DEFAULT_BASE_URL.replace("https://", "wss://") + WS_PATH
        self.rest = KalshiClient(auth=auth, rate_per_s=4)
        self._tickers: list[str] = []
        self._seq: dict[int, int] = {}  # sid -> last seq
        self._counts: dict[str, int] = {}
        self._last_heartbeat = time.monotonic()

    # ------------------------------------------------------------- lifecycle

    async def run_forever(self) -> None:
        backoff = 2.0
        while True:
            try:
                self._tickers = sorted(
                    m["ticker"]
                    for m in discover_game_markets(
                        self.rest, self.series, self.horizon_hours
                    )
                )
                if not self._tickers:
                    log.info("no active markets; sleeping 10 min")
                    await asyncio.sleep(600)
                    continue
                await self._record_session()
                backoff = 2.0
            except asyncio.CancelledError:
                raise
            except SeqGapError as e:
                log.warning("sequence gap (%s); reconnecting for fresh snapshots", e)
                backoff = 2.0
            except websockets.exceptions.InvalidStatus as e:
                if e.response.status_code == 401:
                    log.error(
                        "websocket rejected with 401 - API credentials required. "
                        "Generate an API key at kalshi.com (Settings -> API) and put "
                        "the key id in secrets/kalshi_key_id.txt and the private key "
                        "in secrets/kalshi_private_key.pem. Retrying in 10 min."
                    )
                    await asyncio.sleep(600)
                else:
                    log.exception("handshake rejected; retrying in %.0fs", backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 120.0)
            except Exception:
                log.exception("session error; reconnecting in %.0fs", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 120.0)
            finally:
                self.writer.flush()

    async def _record_session(self) -> None:
        headers = self.auth.ws_headers() if self.auth else {}
        kwargs: dict = {"max_size": 2**24, "ping_interval": 10, "ping_timeout": 30}
        try:
            conn = websockets.connect(self.ws_url, additional_headers=headers, **kwargs)
        except TypeError:  # websockets < 14 names the kwarg differently
            conn = websockets.connect(self.ws_url, extra_headers=headers, **kwargs)
        session_start = time.monotonic()
        async with conn as ws:
            self._seq.clear()
            await ws.send(
                json.dumps(
                    {
                        "id": 1,
                        "cmd": "subscribe",
                        "params": {"channels": CHANNELS, "market_tickers": self._tickers},
                    }
                )
            )
            log.info("subscribed to %d channels x %d markets", len(CHANNELS), len(self._tickers))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
                self._handle(raw)
                now = time.monotonic()
                if now - self._last_heartbeat >= 60.0:
                    self._heartbeat()
                if now - session_start >= self.rediscover_s:
                    new = sorted(
                        m["ticker"]
                        for m in discover_game_markets(
                            self.rest, self.series, self.horizon_hours
                        )
                    )
                    if new != self._tickers:
                        log.info("market set changed (%d -> %d); reconnecting",
                                 len(self._tickers), len(new))
                        return  # clean exit -> run_forever reconnects
                    session_start = now

    # -------------------------------------------------------------- messages

    def _handle(self, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        recv_ts = time.time()
        # Verbatim envelope: never lose data to schema drift.
        self.writer.write(json.dumps({"recv_ts": recv_ts, "raw": raw}, separators=(",", ":")))
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("non-JSON message: %.120s", raw)
            return
        mtype = msg.get("type", "?")
        self._counts[mtype] = self._counts.get(mtype, 0) + 1
        if mtype == "error":
            log.error("exchange error message: %s", msg)
        sid, seq = msg.get("sid"), msg.get("seq")
        if sid is not None and seq is not None:
            last = self._seq.get(sid)
            if last is not None and seq != last + 1:
                raise SeqGapError(f"sid={sid} expected {last + 1} got {seq}")
            self._seq[sid] = seq

    def _heartbeat(self) -> None:
        self._last_heartbeat = time.monotonic()
        self.writer.flush()
        log.info(
            "alive: %d markets, %d lines written, counts=%s",
            len(self._tickers),
            self.writer.lines_written,
            dict(sorted(self._counts.items())),
        )
