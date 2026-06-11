import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalshi_mm.data.build import iter_games
from kalshi_mm.sim.engine import EngineConfig, run_game
from kalshi_mm.sim.fills import FillConfig
from kalshi_mm.strategies.fixed_spread import FixedSpreadMM

for mode_latent in (False, True):
    print(f"\n--- latent={mode_latent} ---")
    for g in iter_games("data/raw", "nba", "2024-25", pregame_hours=6):
        res = run_game(
            g,
            FixedSpreadMM(half_spread_c=0.5, size=100),
            fill_config=FillConfig(latent=mode_latent, rho=0.5, seed=11),
            config=EngineConfig(max_inventory=500, terminal_mode="liquidate"),
        )
        s = res.summary
        print(
            f"{g.ticker}: pnl={s['pnl_c']:8.1f}c fees={s['fees_c']:7.1f}c "
            f"spreadcap={s['spread_captured_c']:7.1f}c fills={s['n_fills']:3d} "
            f"maker_qty={s['maker_qty']:7.1f} latent={s['latent_qty']:6.1f} "
            f"max|inv|={s['max_abs_inv']:5.1f} term_inv={s['terminal_inv']:6.1f}"
        )
