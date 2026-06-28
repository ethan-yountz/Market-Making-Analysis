"""Storage sinks for the order-book logger.

One row per order-book event (full reconstructed book as JSON) and one row per
trade. The primary sink is Postgres (Railway addon) with JSONB columns; a
SQLite sink is provided so the logger runs locally with zero infrastructure.

Choose a sink from a connection string via :func:`make_sink`:
    postgres://...  / postgresql://...  -> PostgresSink
    sqlite:///path  / bare filesystem path / None -> SqliteSink
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_BATCH = 500  # rows buffered per table before a flush


def _dt(epoch_s: float | None) -> datetime | None:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc) if epoch_s else None


class Sink:
    """Interface: buffer events/trades and persist in batches."""

    def write_orderbook_event(self, ev: dict[str, Any]) -> None: ...
    def write_trade(self, tr: dict[str, Any]) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None:
        self.flush()

    @property
    def rows_written(self) -> int:
        return 0


# --------------------------------------------------------------------- Postgres

_PG_DDL = """
CREATE TABLE IF NOT EXISTS orderbook_events (
    id            BIGSERIAL PRIMARY KEY,
    recv_ts       TIMESTAMPTZ NOT NULL,
    exch_ts       TIMESTAMPTZ,
    ticker        TEXT NOT NULL,
    sid           INTEGER,
    seq           BIGINT,
    msg_type      TEXT NOT NULL,
    yes_levels    JSONB,
    no_levels     JSONB,
    best_yes_bid  INTEGER,
    best_yes_ask  INTEGER,
    delta         JSONB
);
CREATE INDEX IF NOT EXISTS idx_ob_ticker_ts ON orderbook_events (ticker, recv_ts);
CREATE TABLE IF NOT EXISTS trades (
    id            BIGSERIAL PRIMARY KEY,
    recv_ts       TIMESTAMPTZ NOT NULL,
    exch_ts       TIMESTAMPTZ,
    ticker        TEXT NOT NULL,
    trade_id      TEXT UNIQUE,
    yes_price     INTEGER,
    no_price      INTEGER,
    count         NUMERIC,
    taker_side    TEXT,
    raw           JSONB
);
CREATE INDEX IF NOT EXISTS idx_tr_ticker_ts ON trades (ticker, recv_ts);
"""


class PostgresSink(Sink):
    def __init__(self, dsn: str, connect_timeout_s: float = 600.0):
        import psycopg2  # lazy: only needed when actually using Postgres
        from psycopg2.extras import Json, execute_values

        self._psycopg2 = psycopg2
        self._Json = Json
        self._execute_values = execute_values
        self._dsn = dsn
        self._connect_timeout_s = connect_timeout_s
        self._ob: list[tuple] = []
        self._tr: list[tuple] = []
        self._n = 0
        self._connect()
        self.ensure_schema()

    def _connect(self) -> None:
        """Connect, retrying while the server is unavailable (still in recovery,
        restarting, …). Raises only if it stays down past the timeout."""
        deadline = time.monotonic() + self._connect_timeout_s
        backoff = 2.0
        while True:
            try:
                self._conn = self._psycopg2.connect(self._dsn)
                self._conn.autocommit = False
                return
            except self._psycopg2.OperationalError as e:
                if time.monotonic() >= deadline:
                    raise
                log.warning("postgres not ready (%s); retry in %.0fs",
                            str(e).strip().splitlines()[0][:100], backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _reconnect(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
        self._connect()

    def ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(_PG_DDL)
        self._conn.commit()

    def write_orderbook_event(self, ev: dict[str, Any]) -> None:
        J = self._Json
        self._ob.append((
            _dt(ev["recv_ts"]), ev.get("exch_ts"), ev["ticker"], ev.get("sid"),
            ev.get("seq"), ev["msg_type"],
            J(ev["yes_levels"]) if ev.get("yes_levels") is not None else None,
            J(ev["no_levels"]) if ev.get("no_levels") is not None else None,
            ev.get("best_yes_bid"), ev.get("best_yes_ask"),
            J(ev["delta"]) if ev.get("delta") is not None else None,
        ))
        if len(self._ob) >= _BATCH:
            self._flush_ob()

    def write_trade(self, tr: dict[str, Any]) -> None:
        self._tr.append((
            _dt(tr["recv_ts"]), tr.get("exch_ts"), tr["ticker"], tr.get("trade_id"),
            tr.get("yes_price"), tr.get("no_price"), tr.get("count"),
            tr.get("taker_side"), self._Json(tr.get("raw")),
        ))
        if len(self._tr) >= _BATCH:
            self._flush_tr()

    _OB_SQL = ("INSERT INTO orderbook_events (recv_ts,exch_ts,ticker,sid,seq,"
               "msg_type,yes_levels,no_levels,best_yes_bid,best_yes_ask,delta) "
               "VALUES %s")
    _TR_SQL = ("INSERT INTO trades (recv_ts,exch_ts,ticker,trade_id,yes_price,"
               "no_price,count,taker_side,raw) VALUES %s "
               "ON CONFLICT (trade_id) DO NOTHING")

    def _flush_buf(self, buf: list[tuple], sql: str) -> None:
        """Insert + commit one buffer, reconnecting once if the connection
        dropped. The buffer is cleared only after a successful commit, so a
        reconnect-and-retry never loses rows."""
        if not buf:
            return
        for attempt in (1, 2):
            try:
                with self._conn.cursor() as cur:
                    self._execute_values(cur, sql, buf)
                self._conn.commit()
                break
            except (self._psycopg2.OperationalError, self._psycopg2.InterfaceError):
                if attempt == 2:
                    raise
                log.warning("postgres connection lost; reconnecting")
                self._reconnect()
            except Exception:
                self._conn.rollback()
                raise
        self._n += len(buf)
        buf.clear()

    def _flush_ob(self) -> None:
        self._flush_buf(self._ob, self._OB_SQL)

    def _flush_tr(self) -> None:
        self._flush_buf(self._tr, self._TR_SQL)

    def flush(self) -> None:
        self._flush_ob()
        self._flush_tr()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._conn.close()

    @property
    def rows_written(self) -> int:
        return self._n


# ----------------------------------------------------------------------- SQLite

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS orderbook_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recv_ts TEXT NOT NULL, exch_ts TEXT, ticker TEXT NOT NULL,
    sid INTEGER, seq INTEGER, msg_type TEXT NOT NULL,
    yes_levels TEXT, no_levels TEXT,
    best_yes_bid INTEGER, best_yes_ask INTEGER, delta TEXT
);
CREATE INDEX IF NOT EXISTS idx_ob_ticker_ts ON orderbook_events (ticker, recv_ts);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recv_ts TEXT NOT NULL, exch_ts TEXT, ticker TEXT NOT NULL,
    trade_id TEXT UNIQUE, yes_price INTEGER, no_price INTEGER,
    count REAL, taker_side TEXT, raw TEXT
);
CREATE INDEX IF NOT EXISTS idx_tr_ticker_ts ON trades (ticker, recv_ts);
"""


class SqliteSink(Sink):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._ob: list[tuple] = []
        self._tr: list[tuple] = []
        self._n = 0
        self._conn.executescript(_SQLITE_DDL)
        self._conn.commit()

    @staticmethod
    def _iso(epoch_s: float | None) -> str | None:
        d = _dt(epoch_s)
        return d.isoformat() if d else None

    def write_orderbook_event(self, ev: dict[str, Any]) -> None:
        self._ob.append((
            self._iso(ev["recv_ts"]), ev.get("exch_ts"), ev["ticker"], ev.get("sid"),
            ev.get("seq"), ev["msg_type"],
            json.dumps(ev["yes_levels"]) if ev.get("yes_levels") is not None else None,
            json.dumps(ev["no_levels"]) if ev.get("no_levels") is not None else None,
            ev.get("best_yes_bid"), ev.get("best_yes_ask"),
            json.dumps(ev["delta"]) if ev.get("delta") is not None else None,
        ))
        if len(self._ob) >= _BATCH:
            self._flush_ob()

    def write_trade(self, tr: dict[str, Any]) -> None:
        self._tr.append((
            self._iso(tr["recv_ts"]), tr.get("exch_ts"), tr["ticker"], tr.get("trade_id"),
            tr.get("yes_price"), tr.get("no_price"), tr.get("count"),
            tr.get("taker_side"), json.dumps(tr.get("raw")),
        ))
        if len(self._tr) >= _BATCH:
            self._flush_tr()

    def _flush_ob(self) -> None:
        if not self._ob:
            return
        self._conn.executemany(
            "INSERT INTO orderbook_events (recv_ts,exch_ts,ticker,sid,seq,msg_type,"
            "yes_levels,no_levels,best_yes_bid,best_yes_ask,delta) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)", self._ob)
        self._n += len(self._ob)
        self._ob.clear()

    def _flush_tr(self) -> None:
        if not self._tr:
            return
        self._conn.executemany(
            "INSERT OR IGNORE INTO trades (recv_ts,exch_ts,ticker,trade_id,yes_price,"
            "no_price,count,taker_side,raw) VALUES (?,?,?,?,?,?,?,?,?)", self._tr)
        self._n += len(self._tr)
        self._tr.clear()

    def flush(self) -> None:
        self._flush_ob()
        self._flush_tr()
        self._conn.commit()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._conn.close()

    @property
    def rows_written(self) -> int:
        return self._n


def make_sink(url: str | None) -> Sink:
    """Build a sink from a connection string (or env-style URL).

    None / "" -> SqliteSink at data/lob.sqlite (zero-config local default).
    """
    if not url:
        return SqliteSink("data/lob.sqlite")
    low = url.lower()
    if low.startswith(("postgres://", "postgresql://")):
        return PostgresSink(url)
    if low.startswith("sqlite:///"):
        return SqliteSink(url[len("sqlite:///"):])
    return SqliteSink(url)


def is_hosted_without_db(db_url: str | None, env: Mapping[str, str] | None = None) -> bool:
    """True when we look like a hosted deploy (Railway sets ``RAILWAY_*`` vars)
    but no Postgres URL is configured — the case where falling back to ephemeral
    SQLite would silently lose data on every restart. The CLI uses this to refuse
    to start instead of quietly mis-recording."""
    if db_url:
        return False
    env = os.environ if env is None else env
    return any(k.startswith("RAILWAY_") for k in env)
