import numpy as np
import pandas as pd
import pytest

from kalshi_mm.data.build import GameData


def synthetic_game(
    n_minutes: int = 120,
    mid0: float = 50.0,
    spread_c: float = 2.0,
    trades_per_min: float = 1.0,
    drift_c_per_min: float = 0.0,
    vol_c: float = 0.3,
    seed: int = 0,
    result: str = "yes",
) -> GameData:
    """Random-walk synthetic market with book updates each minute and Poisson
    trades. Trades print at the touch with the taker side chosen so the tape
    is consistent with the book."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2025-01-01 18:00:00", tz="UTC")
    tip = t0 + pd.Timedelta(minutes=n_minutes)
    mid = mid0
    rows = []
    for i in range(n_minutes):
        ts = t0 + pd.Timedelta(minutes=i)
        mid = float(np.clip(mid + drift_c_per_min + rng.normal(0, vol_c), 5, 95))
        bid = round(mid - spread_c / 2)
        ask = round(mid + spread_c / 2)
        if ask <= bid:
            ask = bid + 1
        rows.append({"ts": ts, "etype": "book", "bid": float(bid), "ask": float(ask),
                     "trade_price": np.nan, "trade_count": np.nan, "taker_side": None})
        for j in range(rng.poisson(trades_per_min)):
            tts = ts + pd.Timedelta(seconds=float(rng.uniform(1, 59)))
            if rng.random() < 0.5:
                rows.append({"ts": tts, "etype": "trade", "bid": np.nan, "ask": np.nan,
                             "trade_price": float(bid), "trade_count": float(rng.integers(5, 80)),
                             "taker_side": "no"})
            else:
                rows.append({"ts": tts, "etype": "trade", "bid": np.nan, "ask": np.nan,
                             "trade_price": float(ask), "trade_count": float(rng.integers(5, 80)),
                             "taker_side": "yes"})
    df = pd.DataFrame(rows).sort_values("ts", kind="stable").reset_index(drop=True)
    df[["bid", "ask"]] = df[["bid", "ask"]].ffill()
    return GameData(
        ticker="TEST-GAME-X", event_ticker="TEST-GAME", sport="nba",
        tip_ts=tip, tip_source="synthetic", result=result, stream=df,
    )


@pytest.fixture
def game():
    return synthetic_game()
