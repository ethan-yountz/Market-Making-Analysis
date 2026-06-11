"""Event-driven market-making backtest engine.

Replays a per-game event stream (book updates + tape trades, plus synthetic
clock ticks so strategies can act between sparse events), routes fills through
the two-layer FillEngine, charges Kalshi fees, and produces a GameResult with
the full fill log and mark-to-market equity curve.

The core loop is a *generator* (``episode``) that yields a MarketState at
every decision point and receives a QuoteSet back. ``run_game`` drives it
with a Strategy object; the DRL gymnasium env drives the same generator with
a policy network. One engine, no duplicated mechanics.

Conventions: YES-space, prices in integer cents in [1, 99], counts in
(fractional) contracts, cash/PnL in cents. Inventory is signed YES exposure.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Generator

import numpy as np
import pandas as pd

from kalshi_mm.data.build import GameData
from kalshi_mm.sim.fees import FeeSchedule
from kalshi_mm.sim.fills import BUY, SELL, Fill, FillConfig, FillEngine, IntensityModel


@dataclass
class Quote:
    price_c: float
    size: float


@dataclass
class QuoteSet:
    """Strategy output. None on a side = no quote there. ``clear`` requests an
    immediate taker flatten of the whole inventory (pays taker fees)."""

    bid: Quote | None = None
    ask: Quote | None = None
    clear: bool = False


@dataclass
class MarketState:
    """Everything a strategy may condition on at a decision point."""

    ts: pd.Timestamp
    t_to_tip_s: float
    bid_c: float          # historical best bid (NaN if empty side)
    ask_c: float
    mid_c: float          # last valid mid
    spread_c: float
    inventory: float
    cash_c: float
    equity_c: float
    vol_c_per_sqrt_min: float   # realized vol of 1-min mid changes
    flow_5m: float              # signed taker volume, trailing 5 min
    our_bid: Quote | None
    our_ask: Quote | None
    tape_volume: float
    new_fills: tuple[Fill, ...] = ()   # fills since the previous decision
    done: bool = False                 # terminal state (no quotes accepted)


class Strategy:
    """Base class. Subclasses override on_event; return None to keep quotes."""

    def reset(self, game: GameData) -> None:  # pragma: no cover - trivial
        pass

    def on_event(self, state: MarketState) -> QuoteSet | None:
        raise NotImplementedError

    def on_game_end(self, state: MarketState) -> None:
        pass


@dataclass
class EngineConfig:
    max_inventory: float = 500.0
    terminal_mode: str = "liquidate"     # "liquidate" | "carry"
    tick_interval_s: float = 30.0        # synthetic decision clock
    min_price_c: int = 1
    max_price_c: int = 99
    vol_window_min: int = 30
    flow_window_s: float = 300.0


@dataclass
class GameResult:
    ticker: str
    event_ticker: str
    tip_ts: pd.Timestamp
    fills: pd.DataFrame
    equity: pd.DataFrame
    summary: dict = field(default_factory=dict)


class _Features:
    """Rolling realized vol of mid + signed trailing taker flow."""

    def __init__(self, vol_window_min: int, flow_window_s: float):
        self.mid_minutes: deque = deque(maxlen=vol_window_min + 1)
        self.flows: deque = deque()
        self.flow_window_s = flow_window_s
        self._last_minute = None

    def on_mid(self, ts: pd.Timestamp, mid: float) -> None:
        minute = ts.floor("1min")
        if self._last_minute is None or minute > self._last_minute:
            self.mid_minutes.append(mid)
            self._last_minute = minute
        else:
            self.mid_minutes[-1] = mid

    def on_trade(self, ts: pd.Timestamp, signed_count: float) -> None:
        self.flows.append((ts, signed_count))

    def vol(self) -> float:
        if len(self.mid_minutes) < 5:
            return 0.5  # prior: half a cent per sqrt-minute
        d = np.diff(np.asarray(self.mid_minutes, dtype=float))
        return float(np.std(d)) if len(d) else 0.5

    def flow(self, now: pd.Timestamp) -> float:
        lo = now - pd.Timedelta(seconds=self.flow_window_s)
        while self.flows and self.flows[0][0] < lo:
            self.flows.popleft()
        return float(sum(c for _, c in self.flows))


def _clamp_quotes(
    qs: QuoteSet, hist_bid: float, hist_ask: float, inv: float, cfg: EngineConfig
) -> tuple[Quote | None, Quote | None]:
    """Make quotes passive, integer-cent, in-bounds, inventory-capped."""
    bid, ask = qs.bid, qs.ask
    if bid is not None:
        p = round(bid.price_c)
        if not math.isnan(hist_ask):
            p = min(p, int(hist_ask) - 1)  # stay passive
        size = min(bid.size, max(0.0, cfg.max_inventory - inv))
        bid = Quote(p, size) if cfg.min_price_c <= p <= cfg.max_price_c and size > 0 else None
    if ask is not None:
        p = round(ask.price_c)
        if not math.isnan(hist_bid):
            p = max(p, int(hist_bid) + 1)
        size = min(ask.size, max(0.0, cfg.max_inventory + inv))
        ask = Quote(p, size) if cfg.min_price_c <= p <= cfg.max_price_c and size > 0 else None
    if bid is not None and ask is not None and bid.price_c >= ask.price_c:
        # Self-crossing request: push the ask out one tick or drop it.
        ask = Quote(bid.price_c + 1, ask.size) if bid.price_c + 1 <= cfg.max_price_c else None
    return bid, ask


def episode(
    game: GameData,
    fees: FeeSchedule | None = None,
    fill_config: FillConfig | None = None,
    intensity: IntensityModel | None = None,
    config: EngineConfig | None = None,
) -> Generator[MarketState, QuoteSet | None, GameResult]:
    """Core engine loop. Yields MarketState at each decision point; the
    caller sends a QuoteSet (or None to keep standing quotes). The final
    yielded state has done=True; the generator then returns a GameResult."""
    fees = fees or FeeSchedule()
    cfg = config or EngineConfig()
    fill_eng = FillEngine(fill_config or FillConfig(), intensity)
    feats = _Features(cfg.vol_window_min, cfg.flow_window_s)

    tip = game.tip_ts
    cash = 0.0
    inv = 0.0
    fees_paid = 0.0
    our_bid: Quote | None = None
    our_ask: Quote | None = None
    mid = math.nan
    hist_bid = hist_ask = math.nan
    prev_ts: pd.Timestamp | None = None
    pending_fills: list[Fill] = []

    fill_rows: list[dict] = []
    curve_rows: list[dict] = []

    def apply_fill(f: Fill, is_taker: bool = False) -> None:
        nonlocal cash, inv, fees_paid, our_bid, our_ask
        cash -= f.side * f.price_c * f.qty
        inv += f.side * f.qty
        fee = (
            fees.taker_fee_c(f.price_c, f.qty)
            if is_taker
            else fees.maker_fee_c(f.price_c, f.qty)
        )
        cash -= fee
        fees_paid += fee
        fill_rows.append(
            {
                "ts": f.ts, "side": f.side, "price_c": f.price_c, "qty": f.qty,
                "mid_c": f.mid_c, "edge_c": f.side * (f.mid_c - f.price_c),
                "fee_c": fee, "layer": f.layer, "taker": is_taker,
            }
        )
        pending_fills.append(f)
        if not is_taker:
            if f.side == BUY and our_bid is not None:
                rem = our_bid.size - f.qty
                our_bid = Quote(our_bid.price_c, rem) if rem > 1e-9 else None
            elif f.side == SELL and our_ask is not None:
                rem = our_ask.size - f.qty
                our_ask = Quote(our_ask.price_c, rem) if rem > 1e-9 else None

    def make_state(ts: pd.Timestamp, done: bool = False) -> MarketState:
        eq = cash + (inv * mid if not math.isnan(mid) else 0.0)
        st = MarketState(
            ts=ts,
            t_to_tip_s=max((tip - ts).total_seconds(), 0.0),
            bid_c=hist_bid, ask_c=hist_ask, mid_c=mid,
            spread_c=(hist_ask - hist_bid)
            if not (math.isnan(hist_bid) or math.isnan(hist_ask))
            else math.nan,
            inventory=inv, cash_c=cash, equity_c=eq,
            vol_c_per_sqrt_min=feats.vol(), flow_5m=feats.flow(ts),
            our_bid=our_bid, our_ask=our_ask,
            tape_volume=fill_eng.tape_volume,
            new_fills=tuple(pending_fills),
            done=done,
        )
        pending_fills.clear()
        return st

    def elapsed_fills(ts: pd.Timestamp) -> None:
        nonlocal prev_ts
        if prev_ts is None or math.isnan(mid):
            prev_ts = ts
            return
        dt = (ts - prev_ts).total_seconds()
        prev_ts = ts
        if dt <= 0:
            return
        for f in fill_eng.on_elapsed(
            ts, dt, max((tip - ts).total_seconds(), 0.0),
            our_bid.price_c if our_bid else None, our_bid.size if our_bid else 0.0,
            our_ask.price_c if our_ask else None, our_ask.size if our_ask else 0.0,
            None if math.isnan(hist_bid) else hist_bid,
            None if math.isnan(hist_ask) else hist_ask,
            mid,
        ):
            apply_fill(f)

    def do_clear(ts: pd.Timestamp) -> None:
        if abs(inv) <= 1e-9 or math.isnan(mid):
            return
        px = hist_bid if inv > 0 else hist_ask
        if math.isnan(px):
            px = mid - 1 if inv > 0 else mid + 1
        px = min(max(round(px), cfg.min_price_c), cfg.max_price_c)
        apply_fill(
            Fill(ts, SELL if inv > 0 else BUY, px, abs(inv), mid, "clear"),
            is_taker=True,
        )

    def decision(ts: pd.Timestamp):
        """Yield state, apply returned quotes. (Sub-generator.)"""
        nonlocal our_bid, our_ask
        qs = yield make_state(ts)
        if qs is not None:
            if qs.clear:
                do_clear(ts)
            our_bid, our_ask = _clamp_quotes(qs, hist_bid, hist_ask, inv, cfg)
        eq = cash + inv * mid
        curve_rows.append({"ts": ts, "mid_c": mid, "inventory": inv,
                           "cash_c": cash, "equity_c": eq})

    # ------------------------------------------------------------- main loop

    next_tick: pd.Timestamp | None = None
    for ev in game.stream.itertuples(index=False):
        ts = ev.ts
        while next_tick is not None and next_tick < ts:
            elapsed_fills(next_tick)
            if not math.isnan(mid):
                feats.on_mid(next_tick, mid)
                yield from decision(next_tick)
            next_tick = next_tick + pd.Timedelta(seconds=cfg.tick_interval_s)

        elapsed_fills(ts)

        if ev.etype == "trade":
            signed = ev.trade_count if ev.taker_side == "yes" else -ev.trade_count
            feats.on_trade(ts, signed)
            if not math.isnan(mid):
                for f in fill_eng.on_trade(
                    ts, ev.trade_price, ev.trade_count, ev.taker_side,
                    our_bid.price_c if our_bid else None, our_bid.size if our_bid else 0.0,
                    our_ask.price_c if our_ask else None, our_ask.size if our_ask else 0.0,
                    None if math.isnan(hist_bid) else hist_bid,
                    None if math.isnan(hist_ask) else hist_ask,
                    mid,
                ):
                    apply_fill(f)
            else:
                fill_eng.tape_volume += ev.trade_count
        else:  # book update
            hist_bid = ev.bid if not pd.isna(ev.bid) else math.nan
            hist_ask = ev.ask if not pd.isna(ev.ask) else math.nan
            if not (math.isnan(hist_bid) or math.isnan(hist_ask)):
                mid = (hist_bid + hist_ask) / 2.0

        if not math.isnan(mid):
            feats.on_mid(ts, mid)
            yield from decision(ts)
            if next_tick is None:
                next_tick = ts + pd.Timedelta(seconds=cfg.tick_interval_s)

    # ------------------------------------------------------------- terminal
    end_ts = tip
    terminal_inv = inv
    if cfg.terminal_mode == "liquidate":
        do_clear(end_ts)
    final_eq = cash + (inv * mid if not math.isnan(mid) else 0.0)
    curve_rows.append({"ts": end_ts, "mid_c": mid, "inventory": inv,
                       "cash_c": cash, "equity_c": final_eq})
    yield make_state(end_ts, done=True)

    fills_df = pd.DataFrame(fill_rows)
    curve_df = pd.DataFrame(curve_rows)
    maker = fills_df[~fills_df["taker"]] if not fills_df.empty else fills_df
    summary = {
        "ticker": game.ticker,
        "event_ticker": game.event_ticker,
        "tip_source": game.tip_source,
        "pnl_c": final_eq,
        "fees_c": fees_paid,
        "spread_captured_c": float((maker["edge_c"] * maker["qty"]).sum()) if not maker.empty else 0.0,
        "n_fills": len(fills_df),
        "maker_qty": float(maker["qty"].sum()) if not maker.empty else 0.0,
        "latent_qty": float(fills_df.loc[fills_df["layer"] == "latent", "qty"].sum()) if not fills_df.empty else 0.0,
        "terminal_inv": terminal_inv,
        "max_abs_inv": float(curve_df["inventory"].abs().max()) if not curve_df.empty else 0.0,
        "mean_abs_inv": float(curve_df["inventory"].abs().mean()) if not curve_df.empty else 0.0,
        "tape_volume": fill_eng.tape_volume,
    }
    return GameResult(game.ticker, game.event_ticker, tip, fills_df, curve_df, summary)


def run_game(
    game: GameData,
    strategy: Strategy,
    fees: FeeSchedule | None = None,
    fill_config: FillConfig | None = None,
    intensity: IntensityModel | None = None,
    config: EngineConfig | None = None,
) -> GameResult:
    """Drive an episode with a Strategy object."""
    strategy.reset(game)
    gen = episode(game, fees, fill_config, intensity, config)
    try:
        state = next(gen)
        while True:
            if state.done:
                strategy.on_game_end(state)
                state = gen.send(None)
            else:
                state = gen.send(strategy.on_event(state))
    except StopIteration as stop:
        return stop.value
