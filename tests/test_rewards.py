"""Reward-shaping tests for clipping and inventory-level penalties."""

import pandas as pd

from kalshi_mm.sim.engine import MarketState
from kalshi_mm.strategies.drl.rewards import RewardConfig, RewardFn
from kalshi_mm.strategies.spooner_rl import SpoonerConfig, SpoonerMM


def _state(equity_c: float, inv: float, mid: float = 50.0) -> MarketState:
    return MarketState(
        ts=pd.Timestamp("2025-01-01", tz="UTC"), t_to_tip_s=0.0,
        bid_c=mid - 1, ask_c=mid + 1, mid_c=mid, spread_c=2.0,
        inventory=inv, cash_c=0.0, equity_c=equity_c,
        vol_c_per_sqrt_min=0.5, flow_5m=0.0, our_bid=None, our_ask=None,
        tape_volume=0.0,
    )


def test_reward_clip_bounds_outliers():
    fn = RewardFn(RewardConfig(name="raw", clip=2.0, size=1.0))
    prev = _state(0.0, 0.0)
    assert fn.step(prev, _state(1000.0, 0.0)) == 2.0    # huge gain clipped
    assert fn.step(prev, _state(-1000.0, 0.0)) == -2.0  # huge loss clipped
    # within band -> untouched
    assert abs(fn.step(prev, _state(1.0, 0.0)) - 1.0) < 1e-9


def test_inv_level_penalty_penalizes_holding():
    base = RewardFn(RewardConfig(name="raw", inv_level_penalty=0.0, size=1.0))
    pen = RewardFn(RewardConfig(name="raw", inv_level_penalty=0.001, size=1.0))
    prev, cur = _state(0.0, 100.0), _state(0.0, 100.0)  # no PnL change, inv=100
    assert base.step(prev, cur) == 0.0
    assert abs(pen.step(prev, cur) - (-0.001 * 100 * 100)) < 1e-9  # -10


def test_spooner_reward_applies_inv_level_penalty():
    agent = SpoonerMM(SpoonerConfig(inv_level_penalty=0.001, size=1.0))
    agent.prev_equity, agent.prev_inv, agent.prev_mid = 0.0, 100.0, 50.0
    r = agent._reward(_state(0.0, 100.0, 50.0))  # dpnl=0, dmid=0
    assert abs(r - (-0.001 * 100 * 100)) < 1e-9
