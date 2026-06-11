"""Kalshi trading fee model.

Game series (KXNBAGAME etc.) use fee type ``quadratic_with_maker_fees``:

    fee = ceil_to_cent( rate * C * P * (1 - P) )      [dollars]

with P the price in dollars, C the contract count. Taker rate is 0.07 and —
since July 2025 — the maker rate on NBA/NFL/NHL/MLB game markets is 0.0175.
Maker fees are the dominant cost of market making here: at P=0.50 a maker
pays 0.4375c/contract per leg, ~0.875c per round trip against a typical 1c
spread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TAKER_RATE = 0.07
MAKER_RATE = 0.0175


@dataclass(frozen=True)
class FeeSchedule:
    maker_rate: float = MAKER_RATE
    taker_rate: float = TAKER_RATE
    multiplier: float = 1.0  # per-series fee_multiplier from series metadata

    def _fee_c(self, rate: float, price_c: float, count: float) -> float:
        if count <= 0 or rate <= 0:
            return 0.0
        p = price_c / 100.0
        fee_dollars = rate * self.multiplier * count * p * (1.0 - p)
        return math.ceil(fee_dollars * 100.0 - 1e-9)  # ceil to whole cents

    def maker_fee_c(self, price_c: float, count: float) -> float:
        return self._fee_c(self.maker_rate, price_c, count)

    def taker_fee_c(self, price_c: float, count: float) -> float:
        return self._fee_c(self.taker_rate, price_c, count)


ZERO_FEES = FeeSchedule(maker_rate=0.0, taker_rate=0.0)
