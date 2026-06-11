"""Two-layer fill model.

Layer 1 (tape-anchored, deterministic): every historical trade is real
marketable flow, routed against the counterfactual book = historical NBBO +
our quotes. Never invents volume.

Layer 2 (latent demand, stochastic, optional): when we quote tighter than the
historical touch, extra flow that never printed is modeled as Poisson with
intensity lambda(delta) = A * exp(-k * delta) (delta = cents from mid). Only
the *incremental* intensity relative to the historical touch is added, so
layers never double count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

BUY, SELL = 1, -1


@dataclass
class Fill:
    ts: object            # pd.Timestamp
    side: int             # BUY: our bid bought YES; SELL: our ask sold YES
    price_c: float
    qty: float
    mid_c: float          # prevailing mid at fill time
    layer: str            # "tape" | "latent" | "terminal"


TOUCH_DEPTH_C = 0.5  # a touch quote in a 1-tick book sits half a tick from mid


class IntensityModel:
    """Piecewise-exponential fill intensity around the touch depth.

        lambda(d) = R * exp(-k_out * (d - 0.5))   for d >= 0.5  (calibrated)
        lambda(d) = R * exp(+k_in  * (0.5 - d))   for d <  0.5  (assumption)

    R is the per-side marketable arrival rate: every observed trade reached at
    least the touch, so a touch quote sees rate R by construction. Beyond the
    touch, decay k_out is identified from the tape's depth distribution.
    Inside the touch (only reachable when the spread is >= 2c) there is no
    historical flow to learn from — k_in is a capped demand-elasticity
    assumption (default: improving by a full tick ~doubles flow), reported as
    a sensitivity, never extrapolated from k_out.
    """

    def __init__(
        self,
        R_per_min: float = 0.6,
        k_out: float = 1.5,
        k_in: float = 1.4,
    ):
        self.R_per_min = R_per_min
        self.k_out = k_out
        self.k_in = k_in

    def params(self, t_to_tip_s: float | None) -> tuple[float, float]:
        """(R_per_min, k_out) at this horizon."""
        return self.R_per_min, self.k_out

    def rate(self, delta_c: float, t_to_tip_s: float | None = None) -> float:
        r_min, k_out = self.params(t_to_tip_s)
        d = max(delta_c, 0.0)
        if d >= TOUCH_DEPTH_C:
            lam = r_min * math.exp(-k_out * (d - TOUCH_DEPTH_C))
        else:
            lam = r_min * math.exp(self.k_in * (TOUCH_DEPTH_C - d))
        return lam / 60.0


class BucketedIntensity(IntensityModel):
    """(R, k_out) by hours-to-tip bucket, as produced by calib/intensity.py."""

    def __init__(self, buckets: list[tuple[float, float, float]], k_in: float = 1.4):
        # buckets: [(max_hours_to_tip, R_per_min, k_out), ...]
        super().__init__(k_in=k_in)
        self.buckets = sorted(buckets)

    def params(self, t_to_tip_s: float | None) -> tuple[float, float]:
        h = (t_to_tip_s or 0.0) / 3600.0
        for max_h, r, k in self.buckets:
            if h <= max_h:
                return r, max(k, 1e-6)
        _, r, k = self.buckets[-1]
        return r, max(k, 1e-6)


@dataclass
class FillConfig:
    rho: float = 0.5                 # at-price queue participation
    latent: bool = True              # enable layer 2
    latent_cap_frac: float = 0.25    # latent volume <= frac of game tape volume
    seed: int = 7


class FillEngine:
    """Stateful per-game fill simulator. The backtest engine feeds it tape
    trades (layer 1) and elapsed-time spans (layer 2)."""

    def __init__(self, config: FillConfig, intensity: IntensityModel | None = None):
        self.cfg = config
        self.intensity = intensity or IntensityModel()
        self.rng = np.random.default_rng(config.seed)
        self.tape_volume = 0.0
        self.latent_volume = 0.0

    # --------------------------------------------------------- layer 1

    def on_trade(
        self,
        ts,
        trade_price_c: float,
        trade_count: float,
        taker_side: str,
        our_bid_c: float | None,
        our_bid_qty: float,
        our_ask_c: float | None,
        our_ask_qty: float,
        hist_bid_c: float | None,
        hist_ask_c: float | None,
        mid_c: float,
    ) -> list[Fill]:
        """Route one tape trade against the counterfactual book.

        taker_side 'no'  = marketable YES sell -> can hit our bid.
        taker_side 'yes' = marketable YES buy  -> can lift our ask.
        """
        self.tape_volume += trade_count
        fills: list[Fill] = []
        if taker_side == "no" and our_bid_c is not None and our_bid_qty > 0:
            if trade_price_c < our_bid_c:
                # Seller went through a worse price: the whole queue at our
                # level (including us) was cleared first.
                qty = min(our_bid_qty, trade_count)
                fills.append(Fill(ts, BUY, our_bid_c, qty, mid_c, "tape"))
            elif trade_price_c == our_bid_c:
                alone = hist_bid_c is None or our_bid_c > hist_bid_c
                frac = 1.0 if alone else self.cfg.rho
                qty = min(our_bid_qty, trade_count * frac)
                if qty > 0:
                    fills.append(Fill(ts, BUY, our_bid_c, qty, mid_c, "tape"))
        elif taker_side == "yes" and our_ask_c is not None and our_ask_qty > 0:
            if trade_price_c > our_ask_c:
                qty = min(our_ask_qty, trade_count)
                fills.append(Fill(ts, SELL, our_ask_c, qty, mid_c, "tape"))
            elif trade_price_c == our_ask_c:
                alone = hist_ask_c is None or our_ask_c < hist_ask_c
                frac = 1.0 if alone else self.cfg.rho
                qty = min(our_ask_qty, trade_count * frac)
                if qty > 0:
                    fills.append(Fill(ts, SELL, our_ask_c, qty, mid_c, "tape"))
        return fills

    # --------------------------------------------------------- layer 2

    def on_elapsed(
        self,
        ts,
        dt_s: float,
        t_to_tip_s: float,
        our_bid_c: float | None,
        our_bid_qty: float,
        our_ask_c: float | None,
        our_ask_qty: float,
        hist_bid_c: float | None,
        hist_ask_c: float | None,
        mid_c: float,
    ) -> list[Fill]:
        """Latent fills over an elapsed span: Poisson arrivals at the
        *incremental* intensity for quotes inside the historical touch."""
        if not self.cfg.latent or dt_s <= 0:
            return []
        if self.latent_volume >= self.cfg.latent_cap_frac * max(self.tape_volume, 1.0):
            return []
        fills: list[Fill] = []
        for side, ours, qty, hist in (
            (BUY, our_bid_c, our_bid_qty, hist_bid_c),
            (SELL, our_ask_c, our_ask_qty, hist_ask_c),
        ):
            if ours is None or qty <= 0:
                continue
            delta_ours = abs(mid_c - ours)
            improving = hist is None or (
                ours > hist if side == BUY else ours < hist
            )
            if not improving:
                continue
            delta_hist = abs(mid_c - hist) if hist is not None else delta_ours + 10.0
            lam = self.intensity.rate(delta_ours, t_to_tip_s) - self.intensity.rate(
                delta_hist, t_to_tip_s
            )
            if lam <= 0:
                continue
            if self.rng.random() < 1.0 - math.exp(-lam * dt_s):
                fills.append(Fill(ts, side, ours, qty, mid_c, "latent"))
                self.latent_volume += qty
        return fills
