"""Normalization of Kalshi API payloads to tidy DataFrames.

Prices are normalized to integer cents; contract counts to floats (Kalshi
supports fractional contracts: ``count_fp``). Handles both the live schema
(``*_dollars`` / ``*_fp`` fields) and the historical schema (bare dollar
strings, ``volume`` / ``open_interest``).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def dollars_to_cents(x) -> float:
    """'0.4600' -> 46.0 (cents). NaN-safe."""
    if x is None:
        return math.nan
    try:
        return round(float(x) * 100.0, 4)
    except (TypeError, ValueError):
        return math.nan


def parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def _pick(d: dict, *names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def trades_to_df(trades: list[dict]) -> pd.DataFrame:
    """Tape rows -> DataFrame[ts, yes_price_c, count, taker_side, trade_id],
    sorted ascending by time."""
    rows = []
    for t in trades:
        rows.append(
            {
                "ts": parse_iso(t.get("created_time")),
                "yes_price_c": dollars_to_cents(
                    _pick(t, "yes_price_dollars", "yes_price")
                ),
                "count": float(_pick(t, "count_fp", "count", default="nan")),
                "taker_side": t.get("taker_side"),
                "trade_id": t.get("trade_id"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["ts", "yes_price_c", "count", "taker_side", "trade_id"]
        )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def _ohlc(block: dict | None, prefix: str) -> dict:
    block = block or {}
    out = {}
    for field in ("open", "high", "low", "close"):
        out[f"{prefix}_{field}"] = dollars_to_cents(
            _pick(block, f"{field}_dollars", field)
        )
    return out


def candles_to_df(candles: list[dict]) -> pd.DataFrame:
    """Candles -> DataFrame indexed work-ready: end_ts (UTC), bid/ask/price
    OHLC in cents, volume, open_interest."""
    rows = []
    for c in candles:
        row = {"end_period_ts": c.get("end_period_ts")}
        row.update(_ohlc(c.get("yes_bid"), "bid"))
        row.update(_ohlc(c.get("yes_ask"), "ask"))
        row.update(_ohlc(c.get("price"), "price"))
        pr = c.get("price") or {}
        row["price_mean"] = dollars_to_cents(_pick(pr, "mean_dollars", "mean"))
        row["volume"] = float(_pick(c, "volume_fp", "volume", default=0.0) or 0.0)
        row["open_interest"] = float(
            _pick(c, "open_interest_fp", "open_interest", default=0.0) or 0.0
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["end_ts"] = pd.to_datetime(df["end_period_ts"], unit="s", utc=True)
    df = df.drop(columns=["end_period_ts"])
    df = df.sort_values("end_ts", kind="stable").reset_index(drop=True)
    return df


def market_to_row(m: dict) -> dict:
    """Market metadata -> flat manifest row."""
    return {
        "ticker": m.get("ticker"),
        "event_ticker": m.get("event_ticker"),
        "title": m.get("title"),
        "yes_sub_title": m.get("yes_sub_title"),
        "open_time": parse_iso(m.get("open_time")),
        "close_time": parse_iso(m.get("close_time")),
        "expected_expiration_time": parse_iso(m.get("expected_expiration_time")),
        "settlement_ts": parse_iso(m.get("settlement_ts")),
        "result": m.get("result"),
        "volume": float(_pick(m, "volume_fp", "volume", default=0.0) or 0.0),
        "open_interest": float(
            _pick(m, "open_interest_fp", "open_interest", default=0.0) or 0.0
        ),
        "status": m.get("status"),
    }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_unix(dt: datetime | pd.Timestamp | None) -> int | None:
    if dt is None or (isinstance(dt, float) and np.isnan(dt)):
        return None
    return int(pd.Timestamp(dt).timestamp())
