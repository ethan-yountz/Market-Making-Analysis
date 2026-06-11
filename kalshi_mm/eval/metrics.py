"""Aggregate metrics across GameResults.

Per-game PnL decomposes exactly (up to rounding) as

    pnl = spread_captured + inventory_pnl - fees

where spread_captured = sum(edge * qty) over ALL fills (edge measured against
the prevailing mid at fill time, negative for taker crossings) and
inventory_pnl = sum over the equity curve of inventory * d(mid).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kalshi_mm.sim.engine import GameResult


def game_row(res: GameResult) -> dict:
    s = dict(res.summary)
    fills, curve = res.fills, res.equity
    if not fills.empty:
        edge_all = float((fills["edge_c"] * fills["qty"]).sum())
    else:
        edge_all = 0.0
    if len(curve) >= 2:
        inv_pnl = float(
            (curve["inventory"].shift().fillna(0.0) * curve["mid_c"].diff().fillna(0.0)).sum()
        )
    else:
        inv_pnl = 0.0
    s["edge_all_c"] = edge_all
    s["inventory_pnl_c"] = inv_pnl
    s["decomp_residual_c"] = s["pnl_c"] - (edge_all + inv_pnl - s["fees_c"])
    return s


def aggregate(results: list[GameResult]) -> tuple[pd.DataFrame, dict]:
    """(per-game table, summary dict)."""
    per_game = pd.DataFrame([game_row(r) for r in results])
    if per_game.empty:
        return per_game, {}
    pnl = per_game["pnl_c"]
    n = len(per_game)
    summary = {
        "games": n,
        "total_pnl_usd": pnl.sum() / 100.0,
        "mean_pnl_per_game_usd": pnl.mean() / 100.0,
        "median_pnl_per_game_usd": pnl.median() / 100.0,
        "game_sharpe": float(pnl.mean() / pnl.std()) if n > 1 and pnl.std() > 0 else np.nan,
        "win_rate": float((pnl > 0).mean()),
        "total_fees_usd": per_game["fees_c"].sum() / 100.0,
        "spread_captured_usd": per_game["spread_captured_c"].sum() / 100.0,
        "inventory_pnl_usd": per_game["inventory_pnl_c"].sum() / 100.0,
        "mean_abs_inv": float(per_game["mean_abs_inv"].mean()),
        "max_abs_inv": float(per_game["max_abs_inv"].max()),
        "mean_terminal_abs_inv": float(per_game["terminal_inv"].abs().mean()),
        "fills_per_game": float(per_game["n_fills"].mean()),
        "maker_qty_per_game": float(per_game["maker_qty"].mean()),
        "latent_qty_share": float(
            per_game["latent_qty"].sum() / max(per_game["maker_qty"].sum(), 1e-9)
        ),
        "worst_game_usd": pnl.min() / 100.0,
        "best_game_usd": pnl.max() / 100.0,
    }
    return per_game, summary


def summary_table(named_summaries: dict[str, dict]) -> pd.DataFrame:
    """Strategies x metrics comparison table."""
    return pd.DataFrame(named_summaries).T
