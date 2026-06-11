"""Bounded-market volatility: sigma(m, t).

A binary price is a probability martingale: its local vol must vanish at the
bounds. We model per-minute mid changes as

    dM ~ sigma(m, t) dW,   sigma(m, t) = c(t) * m * (100 - m) / 2500

so c(t) is the vol of a 50c market in cents/sqrt(min), and the m(100-m)
factor (normalized to 1 at m=50) enforces the bounds. c(t) is estimated per
hours-to-tip bucket as a robust std of normalized minute changes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from kalshi_mm.data.build import GameData

log = logging.getLogger(__name__)

DEFAULT_BUCKETS_H = (0.5, 1.0, 2.0, 4.0, 6.0, 12.0, 24.0, 48.0)


def bound_factor(mid_c: float | np.ndarray) -> float | np.ndarray:
    return np.clip(mid_c, 1.0, 99.0) * (100.0 - np.clip(mid_c, 1.0, 99.0)) / 2500.0


class BoundedVol:
    """sigma in cents/sqrt(minute) at (mid, time-to-tip)."""

    def __init__(self, buckets: list[tuple[float, float]]):
        # [(max_hours_to_tip, c_cents_per_sqrt_min), ...]
        self.buckets = sorted(buckets)

    def c(self, t_to_tip_s: float | None) -> float:
        h = (t_to_tip_s or 0.0) / 3600.0
        for max_h, c in self.buckets:
            if h <= max_h:
                return c
        return self.buckets[-1][1]

    def sigma_per_sqrt_min(self, mid_c: float, t_to_tip_s: float | None) -> float:
        return self.c(t_to_tip_s) * float(bound_factor(mid_c))

    def sigma2_per_sec(self, mid_c: float, t_to_tip_s: float | None) -> float:
        s = self.sigma_per_sqrt_min(mid_c, t_to_tip_s)
        return (s * s) / 60.0


def fit_vol(
    games: Iterable[GameData],
    buckets_h: tuple[float, ...] = DEFAULT_BUCKETS_H,
) -> pd.DataFrame:
    """Estimate c(t) per bucket from per-minute normalized mid changes."""
    frames = []
    for g in games:
        s = g.stream.dropna(subset=["bid", "ask"])
        if s.empty:
            continue
        mid = ((s["bid"] + s["ask"]) / 2.0).astype(float)
        m = pd.DataFrame({"ts": s["ts"], "mid": mid}).set_index("ts")
        m = m.resample("1min").last().ffill()
        dm = m["mid"].diff()
        f = bound_factor(m["mid"].shift())
        z = (dm / f.replace(0, np.nan)).dropna()
        htt = (g.tip_ts - z.index).total_seconds() / 3600.0
        frames.append(pd.DataFrame({"hours_to_tip": htt, "z": z.to_numpy()}))
    if not frames:
        raise ValueError("no data to fit vol on")
    allz = pd.concat(frames, ignore_index=True)

    rows, lo = [], 0.0
    for hi in buckets_h:
        zb = allz.loc[
            (allz["hours_to_tip"] > lo) & (allz["hours_to_tip"] <= hi), "z"
        ].to_numpy()
        if len(zb) >= 100:
            # Classical std with jump clipping. MAD fails here: most minutes
            # have exactly zero mid change (sparse books), so the median
            # absolute deviation collapses to 0.
            s0 = float(np.std(zb))
            if s0 > 0:
                c = float(np.std(np.clip(zb, -5.0 * s0, 5.0 * s0)))
            else:
                c = np.nan
            c = max(c, 1e-3) if not np.isnan(c) else c
        else:
            c = np.nan
        rows.append({"max_h": hi, "c_per_sqrt_min": c, "n_obs": len(zb)})
        lo = hi
    df = pd.DataFrame(rows)
    df["c_per_sqrt_min"] = df["c_per_sqrt_min"].bfill().ffill()
    return df


def save_vol(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(df.to_dict("records"), indent=2), encoding="utf-8")


def load_vol(path: str | Path) -> BoundedVol:
    recs = json.loads(Path(path).read_text(encoding="utf-8"))
    return BoundedVol([(r["max_h"], r["c_per_sqrt_min"]) for r in recs])
