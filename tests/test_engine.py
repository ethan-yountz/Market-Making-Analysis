import math

import pytest

from kalshi_mm.eval.metrics import aggregate, game_row
from kalshi_mm.sim.engine import (
    EngineConfig, MarketState, Quote, QuoteSet, Strategy, run_game,
)
from kalshi_mm.sim.fees import ZERO_FEES
from kalshi_mm.sim.fills import FillConfig
from kalshi_mm.strategies.fixed_spread import FixedSpreadMM
from tests.conftest import synthetic_game


class NoQuote(Strategy):
    def on_event(self, state):
        return QuoteSet(None, None)


class JoinTouch(Strategy):
    def __init__(self, size=100.0):
        self.size = size

    def on_event(self, state: MarketState):
        if math.isnan(state.bid_c) or math.isnan(state.ask_c):
            return None
        return QuoteSet(Quote(state.bid_c, self.size), Quote(state.ask_c, self.size))


class WideQuote(Strategy):
    def on_event(self, state: MarketState):
        if math.isnan(state.mid_c):
            return None
        return QuoteSet(
            Quote(state.mid_c - 25, 100), Quote(state.mid_c + 25, 100)
        )


def test_no_quote_strategy_has_exactly_zero_pnl(game):
    res = run_game(game, NoQuote(), fill_config=FillConfig(latent=False))
    assert res.summary["pnl_c"] == 0.0
    assert res.summary["n_fills"] == 0
    assert res.summary["fees_c"] == 0.0


def test_join_touch_fills_a_lot(game):
    res = run_game(game, JoinTouch(), fill_config=FillConfig(latent=False))
    assert res.summary["n_fills"] > 20
    assert res.summary["maker_qty"] > 0


def test_wide_quotes_never_fill(game):
    res = run_game(game, WideQuote(), fill_config=FillConfig(latent=False),
                   config=EngineConfig(terminal_mode="carry"))
    assert res.summary["n_fills"] == 0


def test_accounting_identity(game):
    """pnl == edge_vs_mid + inventory_pnl - fees, exactly (float tolerance)."""
    res = run_game(game, FixedSpreadMM(half_spread_c=0.5, size=100),
                   fill_config=FillConfig(latent=False))
    row = game_row(res)
    assert abs(row["decomp_residual_c"]) < 1e-6


def test_accounting_identity_with_latent_and_carry():
    g = synthetic_game(spread_c=4.0, seed=3)
    res = run_game(g, FixedSpreadMM(half_spread_c=1.0, size=50),
                   fill_config=FillConfig(latent=True, seed=2),
                   config=EngineConfig(terminal_mode="carry"))
    row = game_row(res)
    assert abs(row["decomp_residual_c"]) < 1e-6


def test_inventory_cap_respected(game):
    cfg = EngineConfig(max_inventory=200)
    res = run_game(game, JoinTouch(size=500), fill_config=FillConfig(latent=False),
                   config=cfg)
    assert res.equity["inventory"].abs().max() <= 200 + 1e-9


def test_terminal_liquidate_flattens(game):
    res = run_game(game, JoinTouch(), fill_config=FillConfig(latent=False),
                   config=EngineConfig(terminal_mode="liquidate"))
    assert abs(res.equity["inventory"].iloc[-1]) < 1e-9


def test_zero_fees_join_touch_profitable_without_adverse_selection():
    """With zero fees, no drift, and symmetric uninformed flow, joining the
    touch must capture spread on net (the textbook MM case)."""
    total = 0.0
    for seed in range(5):
        g = synthetic_game(vol_c=0.0, drift_c_per_min=0.0, trades_per_min=3.0,
                           seed=seed)
        res = run_game(g, JoinTouch(), fees=ZERO_FEES,
                       fill_config=FillConfig(latent=False),
                       config=EngineConfig(terminal_mode="liquidate"))
        total += res.summary["pnl_c"]
    assert total > 0


def test_aggregate_runs(game):
    res = run_game(game, FixedSpreadMM(), fill_config=FillConfig(latent=False))
    per_game, summary = aggregate([res])
    assert summary["games"] == 1
    assert "total_pnl_usd" in summary
