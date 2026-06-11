"""Fixed-spread market making: quote mid +/- s/2 with optional linear
inventory skew. The benchmark every fancier strategy must beat."""

from __future__ import annotations

import math

from kalshi_mm.sim.engine import MarketState, Quote, QuoteSet, Strategy


class FixedSpreadMM(Strategy):
    def __init__(
        self,
        half_spread_c: float = 0.5,
        size: float = 100.0,
        skew_c_per_contract: float = 0.0,
        no_quote_band_c: float = 3.0,
    ):
        self.half = half_spread_c
        self.size = size
        self.skew = skew_c_per_contract
        self.band = no_quote_band_c

    def on_event(self, state: MarketState) -> QuoteSet | None:
        m = state.mid_c
        if math.isnan(m):
            return None
        if m < 1 + self.band or m > 99 - self.band:
            return QuoteSet(None, None)  # too close to the price bounds
        center = m - self.skew * state.inventory
        bid = round(center - self.half)
        ask = round(center + self.half)
        if ask <= bid:
            ask = bid + 1
        return QuoteSet(Quote(bid, self.size), Quote(ask, self.size))
