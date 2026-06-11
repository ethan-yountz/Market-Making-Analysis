"""Build per-game event streams from downloaded candles + tape, and infer the
pregame window end ("tip time").

Kalshi game-market metadata has no scheduled start field —
``expected_expiration_time`` is the projected game END and ``close_time`` is
when a winner was declared. Tip is inferred from the data: pregame, the mid
drifts a few cents over hours; in-game it swings every minute. We detect the
first sustained jump in minute-level price activity before close, falling
back to ``close_time - typical game duration``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Typical real-time duration from tip to winner-declared, used as fallback
# and as a sanity bound on detection.
GAME_DURATION_H = {"nba": 2.4, "nfl": 3.2, "mlb": 2.8, "nhl": 2.7}


@dataclass
class GameData:
    """Everything the backtest engine needs for one market of one game."""

    ticker: str
    event_ticker: str
    sport: str
    tip_ts: pd.Timestamp
    tip_source: str          # "detected" | "fallback"
    result: str              # "yes" | "no"
    stream: pd.DataFrame     # event stream, see build_stream()
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------- tip inference


def infer_tip_time(
    candles: pd.DataFrame,
    close_time: pd.Timestamp,
    sport: str = "nba",
    activity_threshold_c: float = 0.30,
    search_hours: float = 6.0,
) -> tuple[pd.Timestamp, str]:
    """First sustained burst of minute-level mid movement before close.

    activity(t) = 15-min rolling mean of |1-min change in mean price|; tip is
    the first t in [close - search_hours, close] where activity crosses the
    threshold and stays above it for most of the following 30 minutes.
    """
    fallback = close_time - timedelta(hours=GAME_DURATION_H.get(sport, 2.5))
    if candles.empty or "price_mean" not in candles:
        return fallback, "fallback"

    px = (
        candles.set_index("end_ts")["price_mean"]
        .dropna()
        .resample("1min")
        .last()
        .ffill()
    )
    if len(px) < 30:
        return fallback, "fallback"
    activity = px.diff().abs().rolling(15, min_periods=5).mean()

    lo = close_time - timedelta(hours=search_hours)
    window = activity[(activity.index >= lo) & (activity.index <= close_time)]
    hot = window[window >= activity_threshold_c]
    for t in hot.index:
        nxt = activity[(activity.index > t) & (activity.index <= t + timedelta(minutes=30))]
        if len(nxt) >= 10 and (nxt >= activity_threshold_c * 0.66).mean() >= 0.6:
            # Don't trust detections implying absurd game lengths.
            dur_h = (close_time - t).total_seconds() / 3600.0
            max_dur = GAME_DURATION_H.get(sport, 2.5) * 1.9
            if 0.5 <= dur_h <= max_dur:
                return t, "detected"
            break
    return fallback, "fallback"


# ----------------------------------------------------------- event stream


def build_stream(
    candles: pd.DataFrame,
    trades: pd.DataFrame,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> pd.DataFrame:
    """Chronological merge of book updates and tape trades on [t_start, t_end].

    Columns: ts, etype ('book'|'trade'), bid, ask (best, NaN if side empty,
    forward-filled), trade_price, trade_count, taker_side.
    Book rows precede trades at identical timestamps.
    """
    book = candles[["end_ts", "bid_close", "ask_close"]].copy()
    book.columns = ["ts", "bid", "ask"]
    # Kalshi encodes an empty side as bid=0 / ask=100.
    book.loc[book["bid"] <= 0, "bid"] = np.nan
    book.loc[book["ask"] >= 100, "ask"] = np.nan
    book["etype"] = "book"
    book["_prio"] = 0

    tape = trades[["ts", "yes_price_c", "count", "taker_side"]].copy()
    tape.columns = ["ts", "trade_price", "trade_count", "taker_side"]
    tape["etype"] = "trade"
    tape["_prio"] = 1

    ev = pd.concat([book, tape], ignore_index=True)
    ev = ev.sort_values(["ts", "_prio"], kind="stable").drop(columns="_prio")
    ev[["bid", "ask"]] = ev[["bid", "ask"]].ffill()

    ev = ev[(ev["ts"] >= t_start) & (ev["ts"] <= t_end)].reset_index(drop=True)
    return ev


# ----------------------------------------------------------- season loader


def iter_games(
    base_dir: str | Path,
    sport: str,
    season: str,
    pregame_hours: float = 6.0,
    min_volume: float = 0.0,
    min_pregame_trades: int = 20,
    tickers: list[str] | None = None,
) -> Iterator[GameData]:
    """Yield GameData for every market with usable data in a season."""
    base = Path(base_dir) / sport / season
    markets = pd.read_parquet(base / "markets.parquet")
    if tickers is not None:
        markets = markets[markets["ticker"].isin(tickers)]
    if min_volume > 0:
        markets = markets[markets["volume"] >= min_volume]

    for m in markets.itertuples(index=False):
        cf = base / "candles" / f"{m.ticker}.parquet"
        tf = base / "trades" / f"{m.ticker}.parquet"
        if not cf.exists() or not tf.exists():
            continue
        candles, trades = pd.read_parquet(cf), pd.read_parquet(tf)
        if candles.empty:
            continue
        close_t = pd.Timestamp(m.close_time)
        tip, tip_source = infer_tip_time(candles, close_t, sport)
        t_start = tip - timedelta(hours=pregame_hours)
        stream = build_stream(candles, trades, t_start, tip)
        n_trades = int((stream["etype"] == "trade").sum())
        if n_trades < min_pregame_trades:
            continue
        yield GameData(
            ticker=m.ticker,
            event_ticker=m.event_ticker,
            sport=sport,
            tip_ts=tip,
            tip_source=tip_source,
            result=m.result,
            stream=stream,
            meta={
                "close_time": close_t,
                "volume": m.volume,
                "n_pregame_trades": n_trades,
                "title": m.title,
            },
        )
