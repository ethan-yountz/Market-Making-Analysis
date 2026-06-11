"""Calibrate the fill-intensity model from the trade tape, by hours-to-tip.

Model (see sim/fills.IntensityModel): marketable flow walks to depth delta_t
cents from the prevailing mid; every observed trade reached at least the
touch (0.5c in a 1-tick book), so a touch quote is hit at the per-side
arrival rate R. Beyond the touch the decay k_out is the exponential-tail MLE
of the observed depth distribution:

    R     = (trades / observed seconds) / 2          [per side]
    k_out = 1 / mean(delta_t - 0.5 | delta_t > 0.5)  bounded to [0.5, 6]

Kalshi books are 1-tick wide most of the time, so most depth mass sits AT the
touch; k_out is identified only from the through-trades tail, and the
inside-the-touch elasticity is an explicit assumption (k_in in the model),
not extrapolated from this fit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from kalshi_mm.data.build import GameData
from kalshi_mm.sim.fills import BucketedIntensity

log = logging.getLogger(__name__)

DELTA0 = 0.5  # cents: a touch trade in a 1c-wide book sits half a tick from mid
DEFAULT_BUCKETS_H = (0.5, 1.0, 2.0, 4.0, 6.0, 12.0, 24.0, 48.0)


def collect_trade_depths(games: Iterable[GameData]) -> pd.DataFrame:
    """One row per pregame tape trade: hours_to_tip, depth delta_c, count."""
    rows = []
    for g in games:
        s = g.stream
        trades = s[s["etype"] == "trade"]
        mids = (trades["bid"] + trades["ask"]) / 2.0
        delta = (trades["trade_price"] - mids).abs()
        htt = (g.tip_ts - trades["ts"]).dt.total_seconds() / 3600.0
        ok = delta.notna()
        rows.append(pd.DataFrame({
            "hours_to_tip": htt[ok],
            "delta_c": delta[ok],
            "count": trades.loc[ok, "count"] if "count" in trades else trades.loc[ok, "trade_count"],
            "ticker": g.ticker,
        }))
        # observation time contributes to the rate denominator
        rows[-1].attrs = {}
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_intensity(
    games: list[GameData],
    buckets_h: tuple[float, ...] = DEFAULT_BUCKETS_H,
) -> pd.DataFrame:
    """Fit (A_per_min, k_per_cent) per hours-to-tip bucket across games.

    The rate denominator for bucket (h_lo, h_hi] is the summed game-time each
    market actually spent in that bucket within its pregame window.
    """
    depths = collect_trade_depths(games)
    if depths.empty:
        raise ValueError("no trades to calibrate on")

    # Observed seconds per bucket: each game contributes min(window, bucket span).
    windows = []
    for g in games:
        s = g.stream
        if s.empty:
            continue
        h_start = (g.tip_ts - s["ts"].iloc[0]).total_seconds() / 3600.0
        windows.append((g.ticker, h_start))
    windows = pd.DataFrame(windows, columns=["ticker", "h_start"])

    rows = []
    lo = 0.0
    for hi in buckets_h:
        in_b = depths[(depths["hours_to_tip"] > lo) & (depths["hours_to_tip"] <= hi)]
        # seconds observed in this bucket across games
        span = (windows["h_start"].clip(upper=hi) - lo).clip(lower=0.0)
        obs_s = float(span.sum()) * 3600.0
        n = len(in_b)
        if n >= 50 and obs_s > 0:
            d = in_b["delta_c"].to_numpy(dtype=float)
            r_per_min = (n / obs_s) * 60.0 / 2.0     # arrivals/min per side
            tail = d[d > DELTA0 + 1e-9] - DELTA0     # through-trades only
            frac_at_touch = 1.0 - len(tail) / n
            k_out = (
                float(np.clip(1.0 / max(tail.mean(), 1e-3), 0.5, 6.0))
                if len(tail) >= 20
                else np.nan
            )
            rows.append({"max_h": hi, "R_per_min": r_per_min, "k_out": k_out,
                         "frac_at_touch": frac_at_touch,
                         "n_trades": n, "obs_hours": obs_s / 3600.0})
        else:
            rows.append({"max_h": hi, "R_per_min": np.nan, "k_out": np.nan,
                         "frac_at_touch": np.nan,
                         "n_trades": n, "obs_hours": obs_s / 3600.0})
        lo = hi
    df = pd.DataFrame(rows)
    # Fill sparse buckets from nearest fitted neighbor.
    df[["R_per_min", "k_out"]] = df[["R_per_min", "k_out"]].bfill().ffill()
    return df


def save_intensity(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(df.to_dict("records"), indent=2), encoding="utf-8")


def load_intensity(path: str | Path, k_in: float = 1.4) -> BucketedIntensity:
    recs = json.loads(Path(path).read_text(encoding="utf-8"))
    return BucketedIntensity(
        [(r["max_h"], r["R_per_min"], r["k_out"]) for r in recs], k_in=k_in
    )
