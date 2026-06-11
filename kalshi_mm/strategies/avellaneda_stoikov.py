"""Avellaneda-Stoikov market making adapted to a bounded prediction market.

Classic closed form (Avellaneda & Stoikov 2008):

    reservation price r = m - q * gamma * sigma^2 * (T - t)
    half-spread     d/2 = gamma * sigma^2 * (T - t) / 2 + (1/gamma) ln(1 + gamma/k)

Kalshi adaptations:
- T is TIP time, not settlement: pregame horizon is hours, which shrinks the
  inventory term relative to equity-market intuition.
- sigma is local and bounded: sigma(m, t) = c(t) * m(100-m)/2500 from
  calib/vol.py, so quotes naturally tighten near the 1c/99c bounds where a
  probability martingale cannot move much.
- k comes from the same calibrated intensity used by the fill simulator's
  latent layer (calibrate on the train season, evaluate out-of-sample).
- Quotes round to the 1c tick and are suppressed within a band of the bounds
  where the closed form degenerates.
"""

from __future__ import annotations

import math

from kalshi_mm.calib.vol import BoundedVol
from kalshi_mm.sim.engine import MarketState, Quote, QuoteSet, Strategy
from kalshi_mm.sim.fills import IntensityModel


class AvellanedaStoikovMM(Strategy):
    def __init__(
        self,
        gamma: float = 0.002,           # risk aversion, 1/(cent*contract)
        vol: BoundedVol | None = None,
        intensity: IntensityModel | None = None,
        size: float = 100.0,
        no_quote_band_c: float = 3.0,
        max_half_spread_c: float = 5.0,
    ):
        self.gamma = gamma
        self.vol = vol or BoundedVol([(48.0, 0.05)])
        self.intensity = intensity or IntensityModel()
        self.size = size
        self.band = no_quote_band_c
        self.max_half = max_half_spread_c

    def _k(self, t_to_tip_s: float) -> float:
        _, k_out = self.intensity.params(t_to_tip_s)
        return k_out

    def on_event(self, state: MarketState) -> QuoteSet | None:
        m = state.mid_c
        if math.isnan(m):
            return None
        if m < 1 + self.band or m > 99 - self.band:
            # Closed form degenerates at the bounds; also dodge pin risk.
            return QuoteSet(None, None)

        tau_s = max(state.t_to_tip_s, 1.0)
        sigma2_tau = self.vol.sigma2_per_sec(m, tau_s) * tau_s  # cents^2
        g = self.gamma
        k = max(self._k(tau_s), 1e-3)

        reservation = m - state.inventory * g * sigma2_tau
        half = g * sigma2_tau / 2.0 + (1.0 / g) * math.log(1.0 + g / k)
        half = min(half, self.max_half)

        bid = round(reservation - max(half, 0.5))
        ask = round(reservation + max(half, 0.5))
        if ask <= bid:
            ask = bid + 1
        return QuoteSet(Quote(bid, self.size), Quote(ask, self.size))
