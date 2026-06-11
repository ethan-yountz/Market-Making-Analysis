"""Pluggable reward functions for the DRL agent — the experiment surface for
the bounded-market / short-horizon quirks.

All rewards are scaled per quoted contract (divide by size) so magnitudes are
comparable across configurations. The terminal liquidation cost at tip flows
through dPnL automatically when the engine runs in "liquidate" mode; the
optional terminal kappa adds *extra* aversion to carrying into the game.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kalshi_mm.sim.engine import MarketState


@dataclass
class RewardConfig:
    name: str = "spooner"      # raw | inv_penalty | spooner
    eta: float = 0.5           # spooner dampening
    lambda_inv: float = 0.001  # quadratic inventory penalty (cents/contract^2)
    terminal_kappa: float = 0.0  # extra |inv| penalty at tip (cents/contract)
    size: float = 100.0


class RewardFn:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

    def step(self, prev: MarketState, cur: MarketState) -> float:
        c = self.cfg
        dpnl = cur.equity_c - prev.equity_c
        r = dpnl
        if c.name == "spooner":
            if not (math.isnan(prev.mid_c) or math.isnan(cur.mid_c)):
                dmid = cur.mid_c - prev.mid_c
                r -= max(0.0, c.eta * prev.inventory * dmid)
        elif c.name == "inv_penalty":
            r -= c.lambda_inv * prev.inventory * prev.inventory
        elif c.name != "raw":
            raise ValueError(f"unknown reward {c.name}")
        return r / c.size

    def terminal(self, last: MarketState) -> float:
        return -self.cfg.terminal_kappa * abs(last.inventory) / self.cfg.size
