import math

import numpy as np

from kalshi_mm.calib.vol import BoundedVol
from kalshi_mm.sim.engine import MarketState, Quote
from kalshi_mm.sim.fills import BucketedIntensity, IntensityModel
from kalshi_mm.strategies.avellaneda_stoikov import AvellanedaStoikovMM
from kalshi_mm.strategies.spooner_rl import (
    A_CLEAR, A_NOQUOTE, N_ACTIONS, SpoonerMM, ladder_quotes,
)


def state(mid=50.0, bid=49.0, ask=51.0, inv=0.0, tau_s=3600.0, vol=0.3, flow=0.0):
    return MarketState(
        ts=None, t_to_tip_s=tau_s, bid_c=bid, ask_c=ask, mid_c=mid,
        spread_c=ask - bid, inventory=inv, cash_c=0.0, equity_c=0.0,
        vol_c_per_sqrt_min=vol, flow_5m=flow, our_bid=None, our_ask=None,
        tape_volume=0.0,
    )


def test_as_symmetric_when_flat():
    s = AvellanedaStoikovMM(gamma=0.002, vol=BoundedVol([(48.0, 0.3)]))
    qs = s.on_event(state(inv=0.0))
    assert qs.bid is not None and qs.ask is not None
    assert (50 - qs.bid.price_c) == (qs.ask.price_c - 50)


def test_as_skews_against_long_inventory():
    s = AvellanedaStoikovMM(gamma=0.005, vol=BoundedVol([(48.0, 0.5)]))
    flat = s.on_event(state(inv=0.0))
    long = s.on_event(state(inv=400.0))
    # Long inventory pushes the reservation price down: lower bid AND ask.
    assert long.bid is None or long.bid.price_c <= flat.bid.price_c
    if long.ask is not None and flat.ask is not None:
        assert long.ask.price_c <= flat.ask.price_c


def test_as_spread_collapses_toward_tip():
    vol = BoundedVol([(48.0, 0.3)])
    s = AvellanedaStoikovMM(gamma=0.002, vol=vol)
    far = s.on_event(state(tau_s=6 * 3600.0))
    near = s.on_event(state(tau_s=60.0))
    far_w = far.ask.price_c - far.bid.price_c
    near_w = near.ask.price_c - near.bid.price_c
    assert near_w <= far_w


def test_as_no_quotes_near_bounds():
    s = AvellanedaStoikovMM(no_quote_band_c=3.0, vol=BoundedVol([(48.0, 0.3)]))
    qs = s.on_event(state(mid=2.5, bid=2.0, ask=3.0))
    assert qs.bid is None and qs.ask is None


def test_bucketed_intensity_monotone_in_depth():
    bi = BucketedIntensity([(1.0, 4.0, 1.0), (48.0, 0.5, 1.5)])
    assert bi.rate(0.5, 1800) > bi.rate(2.5, 1800)
    # near tip is busier than far from tip
    assert bi.rate(0.5, 1800) > bi.rate(0.5, 6 * 3600)


def test_intensity_inside_touch_capped_extrapolation():
    m = IntensityModel(R_per_min=1.0, k_out=3.0, k_in=1.4)
    # Improving by a full tick should not multiply flow by exp(3*1)
    assert m.rate(0.0, None) / m.rate(0.5, None) < math.exp(1.0)


def test_ladder_quotes_actions():
    s = state()
    qs = ladder_quotes(0, s, 100, 3.0)
    assert qs.bid.price_c == 49 and qs.ask.price_c == 51
    qs = ladder_quotes(A_CLEAR, s, 100, 3.0)
    assert qs.clear and qs.bid is None
    qs = ladder_quotes(A_NOQUOTE, s, 100, 3.0)
    assert qs.bid is None and qs.ask is None and not qs.clear


def test_spooner_learns_something_deterministic():
    agent = SpoonerMM()
    x = agent._features(state())
    idx = agent.coder.indices(x)
    assert len(idx) == agent.coder.n_tilings
    assert np.all(idx >= 0) and np.all(idx < agent.coder.table_size)
    # identical features -> identical indices (deterministic hashing)
    assert np.array_equal(idx, agent.coder.indices(x))
    q = agent._q(idx)
    assert q.shape == (N_ACTIONS,)
