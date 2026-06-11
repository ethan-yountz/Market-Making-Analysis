"""Bulk historical downloaders: market discovery, 1-min candlesticks, and the
full trade tape per market, stored as parquet with a resumable manifest.

Layout:
    data/raw/{sport}/{season}/markets.parquet     # all market metadata
    data/raw/{sport}/{season}/manifest.parquet    # per-ticker download status
    data/raw/{sport}/{season}/candles/{ticker}.parquet
    data/raw/{sport}/{season}/trades/{ticker}.parquet

Markets settled before the historical cutoff live on /historical endpoints;
recent ones on the live endpoints. Both are tried where ambiguous.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from kalshi_mm.api.client import KalshiAPIError, KalshiClient
from kalshi_mm.data.normalize import (
    candles_to_df,
    market_to_row,
    to_unix,
    trades_to_df,
    utc_now,
)

log = logging.getLogger(__name__)

# Max 1-min candles per request is ~5000; 3 days = 4320 keeps headroom.
CANDLE_CHUNK = timedelta(days=3)


def discover_markets(
    client: KalshiClient,
    series_ticker: str,
    min_close: datetime,
    max_close: datetime,
) -> pd.DataFrame:
    """Union of live + historical market listings for a series in a window."""
    rows: dict[str, dict] = {}
    for source in ("live", "historical"):
        try:
            it = (
                client.get_markets(
                    series_ticker=series_ticker,
                    min_close_ts=to_unix(min_close),
                    max_close_ts=to_unix(max_close),
                )
                if source == "live"
                else client.get_historical_markets(
                    series_ticker=series_ticker,
                    min_close_ts=to_unix(min_close),
                    max_close_ts=to_unix(max_close),
                )
            )
            n = 0
            for m in it:
                rows[m["ticker"]] = market_to_row(m)
                n += 1
            log.info("%s discovery (%s): %d markets", source, series_ticker, n)
        except KalshiAPIError as e:
            log.warning("%s discovery failed for %s: %s", source, series_ticker, e)
    df = pd.DataFrame(list(rows.values()))
    if not df.empty:
        df = df.sort_values("close_time").reset_index(drop=True)
    return df


def _fetch_candles(
    client: KalshiClient,
    series_ticker: str,
    ticker: str,
    start: datetime,
    end: datetime,
    use_historical: bool,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    t0 = start
    while t0 < end:
        t1 = min(t0 + CANDLE_CHUNK, end)
        raw = None
        for attempt_hist in ([use_historical, not use_historical]):
            try:
                if attempt_hist:
                    raw = client.get_historical_candlesticks(
                        ticker, to_unix(t0), to_unix(t1), period_interval=1
                    )
                else:
                    raw = client.get_candlesticks(
                        series_ticker, ticker, to_unix(t0), to_unix(t1), period_interval=1
                    )
                break
            except KalshiAPIError as e:
                if e.status == 404:
                    continue
                raise
        if raw:
            chunks.append(candles_to_df(raw))
        t0 = t1
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    return df.drop_duplicates(subset="end_ts").sort_values("end_ts").reset_index(drop=True)


def _fetch_trades(
    client: KalshiClient, ticker: str, use_historical: bool
) -> pd.DataFrame:
    for attempt_hist in ([use_historical, not use_historical]):
        try:
            it = (
                client.get_historical_trades(ticker=ticker)
                if attempt_hist
                else client.get_trades(ticker)
            )
            return trades_to_df(list(it))
        except KalshiAPIError as e:
            if e.status == 404:
                continue
            raise
    return pd.DataFrame()


def _load_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def download_season(
    client: KalshiClient,
    series_ticker: str,
    sport: str,
    season: str,
    min_close: datetime,
    max_close: datetime,
    base_dir: str | Path = "data/raw",
    candle_lookback_hours: float = 48.0,
    min_volume: float = 0.0,
) -> pd.DataFrame:
    """Download a full (sport, season). Resumable: tickers already marked done
    in the manifest are skipped. Returns the final manifest."""
    out = Path(base_dir) / sport / season
    (out / "candles").mkdir(parents=True, exist_ok=True)
    (out / "trades").mkdir(parents=True, exist_ok=True)

    markets = discover_markets(client, series_ticker, min_close, max_close)
    if markets.empty:
        log.warning("no markets discovered for %s %s", sport, season)
        return pd.DataFrame()
    markets.to_parquet(out / "markets.parquet", index=False)
    log.info("%s %s: %d markets", sport, season, len(markets))

    cutoff_ts = None
    try:
        cutoff = client.get_historical_cutoff()
        cutoff_ts = pd.Timestamp(cutoff.get("market_settled_ts")).tz_convert("UTC")
    except Exception:
        log.warning("could not fetch historical cutoff; defaulting to live-first")

    manifest_path = out / "manifest.parquet"
    manifest = _load_or_empty(manifest_path)
    done: set[str] = (
        set(manifest.loc[manifest["status"] == "done", "ticker"])
        if not manifest.empty
        else set()
    )
    rows = [] if manifest.empty else manifest.to_dict("records")

    todo = markets[~markets["ticker"].isin(done)]
    if min_volume > 0:
        todo = todo[todo["volume"] >= min_volume]
    for i, m in enumerate(todo.itertuples(index=False)):
        ticker = m.ticker
        close_t = pd.Timestamp(m.close_time)
        open_t = pd.Timestamp(m.open_time) if pd.notna(m.open_time) else close_t - timedelta(days=7)
        use_hist = bool(cutoff_ts is not None and close_t < cutoff_ts)
        start = max(open_t, close_t - timedelta(hours=candle_lookback_hours))
        row = {"ticker": ticker, "status": "done", "error": "",
               "candles_rows": 0, "trades_rows": 0, "updated_at": utc_now()}
        try:
            candles = _fetch_candles(client, series_ticker, ticker, start, close_t, use_hist)
            trades = _fetch_trades(client, ticker, use_hist)
            if not candles.empty:
                candles.to_parquet(out / "candles" / f"{ticker}.parquet", index=False)
            if not trades.empty:
                trades.to_parquet(out / "trades" / f"{ticker}.parquet", index=False)
            row["candles_rows"] = len(candles)
            row["trades_rows"] = len(trades)
            if candles.empty and trades.empty:
                row["status"] = "empty"
        except Exception as e:  # keep going; manifest records the failure
            log.exception("failed %s", ticker)
            row["status"] = "error"
            row["error"] = str(e)[:300]
        rows.append(row)
        if (i + 1) % 25 == 0 or (i + 1) == len(todo):
            pd.DataFrame(rows).drop_duplicates(subset="ticker", keep="last").to_parquet(
                manifest_path, index=False
            )
            log.info("[%s %s] %d/%d markets downloaded", sport, season, i + 1, len(todo))

    manifest = pd.DataFrame(rows).drop_duplicates(subset="ticker", keep="last")
    manifest.to_parquet(manifest_path, index=False)
    return manifest
