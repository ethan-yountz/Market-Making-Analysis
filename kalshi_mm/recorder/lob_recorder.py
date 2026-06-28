"""Postgres-backed order-book logger with live book reconstruction.

Subscribes to ``orderbook_delta`` + ``trade`` for active game markets,
maintains a local :class:`OrderBook` per market by applying snapshots and
deltas, and writes one row per event (full reconstructed book) plus one row
per trade to a :class:`Sink` (Postgres in production, SQLite locally).

Correctness rules (from Kalshi's protocol, verified against the live feed):
- ``seq`` is monotonic *per sid* (subscription/channel), spanning all markets
  on that channel. A gap means we may have missed a delta on some market, so
  the only safe response is a full reconnect for fresh snapshots.
- The active market set is rediscovered periodically; a change triggers a
  clean reconnect so new games get snapshots and finished ones are dropped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone

import websockets

from kalshi_mm.api.client import DEFAULT_BASE_URL, WS_PATH, KalshiAuth, KalshiClient
from kalshi_mm.recorder.book import OrderBook, price_to_cents
from kalshi_mm.recorder.discover import discover_game_markets
from kalshi_mm.recorder.storage import Sink

log = logging.getLogger(__name__)

CHANNELS = ["orderbook_delta", "trade"]
MLB = ("KXMLBGAME",)


def _exch_iso(msg: dict) -> str | None:
    ms = msg.get("ts_ms")
    if ms is not None:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()
    return msg.get("ts") if isinstance(msg.get("ts"), str) else None


class SeqGapError(RuntimeError):
    pass


class LobRecorder:
    def __init__(
        self,
        auth: KalshiAuth | None,
        sink: Sink,
        series: tuple[str, ...] = MLB,
        horizon_hours: float = 36.0,
        rediscover_minutes: float = 15.0,
        ws_url: str | None = None,
        mode: str = "topbook",
        depth_cents: int = 5,
    ):
        self.auth = auth
        self.sink = sink
        self.series = series
        self.mode = mode  # "topbook" (near-touch, light) | "full" (snapshot+deltas)
        self.depth_cents = depth_cents
        self.horizon_hours = horizon_hours
        self.rediscover_s = rediscover_minutes * 60.0
        self.ws_url = ws_url or DEFAULT_BASE_URL.replace("https://", "wss://") + WS_PATH
        self.rest = KalshiClient(auth=auth, rate_per_s=4)
        self._tickers: list[str] = []
        self._books: dict[str, OrderBook] = {}
        self._last_top: dict[str, tuple] = {}  # ticker -> last written window key
        self._seq: dict[int, int] = {}  # sid -> last seq
        self._counts: Counter[str] = Counter()
        self._last_heartbeat = time.monotonic()

    # ------------------------------------------------------------- lifecycle

    async def run_forever(self) -> None:
        backoff = 2.0
        while True:
            try:
                self._tickers = self._discover()
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
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code == 401:
                    log.error("websocket 401 - check API credentials; retry in 10 min")
                    await asyncio.sleep(600)
                else:
                    log.exception("handshake rejected; retry in %.0fs", backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 120.0)
            except Exception:
                log.exception("session error; reconnecting in %.0fs", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 120.0)
            finally:
                self.sink.flush()

    def _discover(self) -> list[str]:
        return sorted(
            m["ticker"]
            for m in discover_game_markets(self.rest, self.series, self.horizon_hours)
        )

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
            self._books.clear()
            self._last_top.clear()
            await ws.send(json.dumps({
                "id": 1, "cmd": "subscribe",
                "params": {"channels": CHANNELS, "market_tickers": self._tickers},
            }))
            log.info("subscribed %s x %d markets", CHANNELS, len(self._tickers))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=90.0)
                self._handle(raw)
                now = time.monotonic()
                if now - self._last_heartbeat >= 60.0:
                    self._heartbeat()
                if now - session_start >= self.rediscover_s:
                    new = self._discover()
                    if new != self._tickers:
                        log.info("market set changed (%d -> %d); reconnecting",
                                 len(self._tickers), len(new))
                        return  # clean exit -> run_forever reconnects
                    session_start = now

    # -------------------------------------------------------------- messages

    def _handle(self, raw: str | bytes) -> None:
        recv_ts = time.time()
        msg = json.loads(raw)
        mtype = msg.get("type", "?")
        self._counts[mtype] += 1

        if mtype == "error":
            log.error("exchange error: %s", msg)
            return
        if mtype in ("subscribed", "ok", "unsubscribed"):
            return

        self._check_seq(msg)  # raises SeqGapError on a gap
        inner = msg.get("msg", {})

        if mtype == "orderbook_snapshot":
            self._on_book(msg, inner, snapshot=True, recv_ts=recv_ts)
        elif mtype == "orderbook_delta":
            self._on_book(msg, inner, snapshot=False, recv_ts=recv_ts)
        elif mtype == "trade":
            self._on_trade(inner, recv_ts)

    def _check_seq(self, msg: dict) -> None:
        sid, seq = msg.get("sid"), msg.get("seq")
        if sid is None or seq is None:
            return
        last = self._seq.get(sid)
        if last is not None and seq != last + 1:
            raise SeqGapError(f"sid={sid} expected {last + 1} got {seq}")
        self._seq[sid] = seq

    def _on_book(self, msg: dict, inner: dict, snapshot: bool, recv_ts: float) -> None:
        ticker = inner["market_ticker"]
        book = self._books.get(ticker)
        if snapshot or book is None:
            book = OrderBook(ticker)
            self._books[ticker] = book
        if snapshot:
            book.apply_snapshot(inner)
        else:
            book.apply_delta(inner)
        if self.mode == "topbook":
            self._write_topbook(book, msg, inner, recv_ts)
        else:
            self._write_full(book, msg, inner, snapshot, recv_ts)

    def _write_topbook(self, book: OrderBook, msg: dict, inner: dict, recv_ts: float) -> None:
        """Lighter mode: write a row only when the **top of book** (best yes
        bid/ask) moves, carrying the near-touch book (±depth_cents around the
        spread) as a self-contained snapshot. Size churn that doesn't move the
        touch, and all deep-book churn, write nothing — the dominant saving.
        Every trade is still recorded separately, so executions are never lost."""
        byb, bya = book.best_yes_bid, book.best_yes_ask
        if byb is None or bya is None:
            return  # need both sides to define a spread
        key = (byb, bya)
        if self._last_top.get(book.ticker) == key:
            return  # touch unchanged
        yw = book.side_window("yes", self.depth_cents)
        nw = book.side_window("no", self.depth_cents)
        self._last_top[book.ticker] = key
        self.sink.write_orderbook_event({
            "recv_ts": recv_ts,
            "exch_ts": _exch_iso(inner),
            "ticker": book.ticker,
            "sid": msg.get("sid"),
            "seq": msg.get("seq"),
            "msg_type": "book",
            "yes_levels": yw,
            "no_levels": nw,
            "best_yes_bid": byb,
            "best_yes_ask": bya,
            "delta": None,
        })

    def _write_full(self, book: OrderBook, msg: dict, inner: dict,
                    snapshot: bool, recv_ts: float) -> None:
        """Heavy mode: full book on snapshots, compact change on deltas
        (reconstruct offline by replaying deltas in seq order)."""
        self.sink.write_orderbook_event({
            "recv_ts": recv_ts,
            "exch_ts": _exch_iso(inner),
            "ticker": book.ticker,
            "sid": msg.get("sid"),
            "seq": msg.get("seq"),
            "msg_type": "snapshot" if snapshot else "delta",
            "yes_levels": book.yes_levels() if snapshot else None,
            "no_levels": book.no_levels() if snapshot else None,
            "best_yes_bid": book.best_yes_bid,
            "best_yes_ask": book.best_yes_ask,
            "delta": None if snapshot else inner,
        })

    def _on_trade(self, inner: dict, recv_ts: float) -> None:
        self.sink.write_trade({
            "recv_ts": recv_ts,
            "exch_ts": _exch_iso(inner),
            "ticker": inner["market_ticker"],
            "trade_id": inner.get("trade_id"),
            "yes_price": price_to_cents(inner["yes_price_dollars"]),
            "no_price": price_to_cents(inner["no_price_dollars"]),
            "count": float(inner["count_fp"]),
            "taker_side": inner.get("taker_side"),
            "raw": inner,
        })

    def _heartbeat(self) -> None:
        self._last_heartbeat = time.monotonic()
        self.sink.flush()
        log.info(
            "alive: %d books, %d rows written, counts=%s",
            len(self._books), self.sink.rows_written, dict(self._counts),
        )
