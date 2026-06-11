from kalshi_mm.sim.fills import BUY, SELL, FillConfig, FillEngine, IntensityModel


def make_engine(rho=0.5, latent=False):
    return FillEngine(FillConfig(rho=rho, latent=latent, seed=1), IntensityModel())


def test_trade_through_fills_fully():
    eng = make_engine()
    # Seller prints at 44, our bid is 46: queue at 46 cleared first.
    fills = eng.on_trade(0, 44, 50, "no", our_bid_c=46, our_bid_qty=100,
                         our_ask_c=None, our_ask_qty=0,
                         hist_bid_c=46, hist_ask_c=48, mid_c=47)
    assert len(fills) == 1
    assert fills[0].side == BUY and fills[0].qty == 50 and fills[0].price_c == 46


def test_at_price_join_takes_rho_share():
    eng = make_engine(rho=0.5)
    fills = eng.on_trade(0, 46, 60, "no", our_bid_c=46, our_bid_qty=100,
                         our_ask_c=None, our_ask_qty=0,
                         hist_bid_c=46, hist_ask_c=48, mid_c=47)
    assert len(fills) == 1 and fills[0].qty == 30  # 60 * 0.5


def test_at_price_alone_when_improving_fills_fully():
    eng = make_engine(rho=0.5)
    # We bid 47 inside the 46/48 book: alone at that level.
    fills = eng.on_trade(0, 47, 60, "no", our_bid_c=47, our_bid_qty=100,
                         our_ask_c=None, our_ask_qty=0,
                         hist_bid_c=46, hist_ask_c=48, mid_c=47)
    assert len(fills) == 1 and fills[0].qty == 60


def test_no_fill_when_trade_far_from_quote():
    eng = make_engine()
    fills = eng.on_trade(0, 48, 60, "yes", our_bid_c=44, our_bid_qty=100,
                         our_ask_c=52, our_ask_qty=100,
                         hist_bid_c=46, hist_ask_c=48, mid_c=47)
    assert fills == []


def test_buy_flow_hits_ask_not_bid():
    eng = make_engine()
    fills = eng.on_trade(0, 48, 40, "yes", our_bid_c=46, our_bid_qty=100,
                         our_ask_c=48, our_ask_qty=100,
                         hist_bid_c=46, hist_ask_c=48, mid_c=47)
    assert len(fills) == 1 and fills[0].side == SELL


def test_fill_capped_by_quote_size():
    eng = make_engine()
    fills = eng.on_trade(0, 44, 500, "no", our_bid_c=46, our_bid_qty=70,
                         our_ask_c=None, our_ask_qty=0,
                         hist_bid_c=46, hist_ask_c=48, mid_c=47)
    assert fills[0].qty == 70


def test_layer1_conserves_tape_volume():
    eng = make_engine(rho=1.0)
    total = 0.0
    for _ in range(10):
        fills = eng.on_trade(0, 46, 30, "no", our_bid_c=46, our_bid_qty=1000,
                             our_ask_c=None, our_ask_qty=0,
                             hist_bid_c=46, hist_ask_c=48, mid_c=47)
        total += sum(f.qty for f in fills)
    assert total <= eng.tape_volume + 1e-9


def test_latent_only_when_improving():
    eng = make_engine(latent=True)
    eng.tape_volume = 10_000  # leave cap headroom
    # Joining the touch -> no latent fills ever.
    fills = eng.on_elapsed(0, 3600, 1800, our_bid_c=46, our_bid_qty=100,
                           our_ask_c=48, our_ask_qty=100,
                           hist_bid_c=46, hist_ask_c=48, mid_c=47)
    assert fills == []


def test_latent_respects_cap():
    eng = make_engine(latent=True)
    eng.tape_volume = 100.0
    eng.latent_volume = 25.0  # at the 25% cap already
    fills = eng.on_elapsed(0, 3600, 1800, our_bid_c=47, our_bid_qty=100,
                           our_ask_c=None, our_ask_qty=0,
                           hist_bid_c=46, hist_ask_c=49, mid_c=47.5)
    assert fills == []
