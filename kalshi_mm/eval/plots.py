"""Standard figures for the writeup. All take per-game tables / GameResults
and save PNGs; matplotlib only, no styling dependencies."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cumulative_pnl(per_game_by_strategy: dict[str, pd.DataFrame], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, pg in per_game_by_strategy.items():
        pnl = pg.sort_values("ticker")["pnl_c"].to_numpy() / 100.0
        ax.plot(np.cumsum(pnl), label=name, lw=1.4)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("game #")
    ax.set_ylabel("cumulative PnL (USD)")
    ax.set_title("Cumulative PnL across games")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pnl_distribution(per_game_by_strategy: dict[str, pd.DataFrame], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, pg in per_game_by_strategy.items():
        ax.hist(pg["pnl_c"] / 100.0, bins=60, alpha=0.5, label=name)
    ax.set_xlabel("per-game PnL (USD)")
    ax.set_ylabel("games")
    ax.set_title("Per-game PnL distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def inventory_profile(per_game_by_strategy: dict[str, pd.DataFrame], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(per_game_by_strategy)
    data = [pg["mean_abs_inv"] for pg in per_game_by_strategy.values()]
    ax.boxplot(data, tick_labels=names, showfliers=False)
    ax.set_ylabel("mean |inventory| (contracts)")
    ax.set_title("Inventory held")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def decomposition_bars(summaries: dict[str, dict], path: str | Path) -> None:
    names = list(summaries)
    spread = [summaries[n]["spread_captured_usd"] for n in names]
    invpnl = [summaries[n]["inventory_pnl_usd"] for n in names]
    fees = [-summaries[n]["total_fees_usd"] for n in names]
    total = [summaries[n]["total_pnl_usd"] for n in names]
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 1.5 * w, spread, w, label="spread captured")
    ax.bar(x - 0.5 * w, invpnl, w, label="inventory PnL")
    ax.bar(x + 0.5 * w, fees, w, label="fees (-)")
    ax.bar(x + 1.5 * w, total, w, label="total PnL", color="k", alpha=0.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, names)
    ax.set_ylabel("USD")
    ax.set_title("PnL decomposition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
