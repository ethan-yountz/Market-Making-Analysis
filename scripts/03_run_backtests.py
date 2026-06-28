"""Run backtests and produce the comparison table + figures.

    python scripts/03_run_backtests.py --sport nba --season 2024-25 \
        --strategies fixed_spread fixed_spread_skew as --games 100

Fill-mode note: results are produced in tape-only mode (conservative) unless
--latent is passed. Calibrations are read from data/calib (run 02 first);
without them, conservative defaults are used.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalshi_mm.calib.intensity import load_intensity
from kalshi_mm.calib.vol import BoundedVol, load_vol
from kalshi_mm.data.build import iter_games
from kalshi_mm.eval import plots
from kalshi_mm.eval.metrics import aggregate, summary_table
from kalshi_mm.sim.engine import EngineConfig, run_game
from kalshi_mm.sim.fills import FillConfig, IntensityModel
from kalshi_mm.strategies.avellaneda_stoikov import AvellanedaStoikovMM
from kalshi_mm.strategies.fixed_spread import FixedSpreadMM


def build_strategy(name: str, vol, intensity, size: float):
    if name == "fixed_spread":
        return FixedSpreadMM(half_spread_c=0.5, size=size)
    if name == "fixed_spread_wide":
        return FixedSpreadMM(half_spread_c=1.5, size=size)
    if name == "fixed_spread_skew":
        return FixedSpreadMM(half_spread_c=0.5, size=size, skew_c_per_contract=0.01)
    if name == "as":
        return AvellanedaStoikovMM(gamma=0.002, vol=vol, intensity=intensity, size=size)
    if name.startswith("as_g"):  # e.g. as_g0.005
        return AvellanedaStoikovMM(gamma=float(name[4:]), vol=vol, intensity=intensity, size=size)
    if name == "spooner":
        from kalshi_mm.strategies.spooner_rl import SpoonerMM
        return SpoonerMM.load_latest(size=size)
    if name == "drl":
        from kalshi_mm.strategies.drl.policy import DRLMM
        return DRLMM.load_latest(size=size)
    raise SystemExit(f"unknown strategy {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sport", default="nba")
    ap.add_argument("--season", default="2024-25")
    ap.add_argument("--calib-season", default="2024-25",
                    help="season whose calibration files to use")
    ap.add_argument("--strategies", nargs="+",
                    default=["fixed_spread", "fixed_spread_skew", "as"])
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--pregame-hours", type=float, default=6.0)
    ap.add_argument("--size", type=float, default=100.0)
    ap.add_argument("--latent", action="store_true", help="enable layer-2 latent fills")
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--terminal", choices=["liquidate", "settle", "carry"],
                    default="liquidate",
                    help="value leftover inventory at tip: cross out (liquidate), "
                         "hold to the realized outcome fee-free (settle), or mark "
                         "at last mid (carry)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--base-dir", default="data/raw")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    calib_dir = Path("data/calib")
    ipath = calib_dir / f"{args.sport}_{args.calib_season}_intensity.json"
    vpath = calib_dir / f"{args.sport}_{args.calib_season}_vol.json"
    intensity = load_intensity(ipath) if ipath.exists() else IntensityModel()
    vol = load_vol(vpath) if vpath.exists() else BoundedVol([(48.0, 0.2)])
    if not ipath.exists():
        logging.warning("no intensity calibration at %s - using defaults", ipath)

    games = []
    for g in iter_games(args.base_dir, args.sport, args.season,
                        pregame_hours=args.pregame_hours):
        games.append(g)
        if args.games and len(games) >= args.games:
            break
    logging.info("loaded %d games", len(games))
    if not games:
        sys.exit("no games - run 01_download.py first")

    out = Path(args.out_dir) / f"{args.sport}_{args.season}"
    out.mkdir(parents=True, exist_ok=True)

    per_game_by, summaries = {}, {}
    for name in args.strategies:
        strat = build_strategy(name, vol, intensity, args.size)
        results = []
        for g in games:
            results.append(
                run_game(
                    g, strat,
                    fill_config=FillConfig(rho=args.rho, latent=args.latent, seed=11),
                    intensity=intensity,
                    config=EngineConfig(terminal_mode=args.terminal),
                )
            )
        per_game, summary = aggregate(results)
        per_game_by[name], summaries[name] = per_game, summary
        per_game.to_parquet(out / f"per_game_{name}.parquet", index=False)
        logging.info("%s: total %.2f USD over %d games (sharpe %.2f)",
                     name, summary["total_pnl_usd"], summary["games"],
                     summary["game_sharpe"])

    table = summary_table(summaries)
    table.to_csv(out / "summary.csv")
    pd.set_option("display.width", 200)
    print("\n" + table.round(3).to_string())

    plots.cumulative_pnl(per_game_by, out / "cumulative_pnl.png")
    plots.pnl_distribution(per_game_by, out / "pnl_distribution.png")
    plots.inventory_profile(per_game_by, out / "inventory.png")
    plots.decomposition_bars(summaries, out / "decomposition.png")
    print(f"\nfigures + tables in {out}")


if __name__ == "__main__":
    main()
