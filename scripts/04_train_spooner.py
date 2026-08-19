"""Train the Spooner SARSA(lambda) agent on a season of pregame windows.

    python scripts/04_train_spooner.py --season 2024-25 --epochs 4 --games 600

Episodes are single (market, pregame-window) replays. Epsilon and alpha anneal
across epochs; after each epoch the greedy policy is evaluated on a held-out
validation slice. Best-by-validation weights are saved to models/.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalshi_mm.calib.intensity import load_intensity
from kalshi_mm.data.build import iter_games
from kalshi_mm.eval.metrics import aggregate
from kalshi_mm.sim.engine import EngineConfig, run_game
from kalshi_mm.sim.fills import FillConfig, IntensityModel
from kalshi_mm.strategies.spooner_rl import SpoonerConfig, SpoonerMM, model_path


def evaluate(agent, games, fill_cfg, intensity, eng_cfg):
    agent_train, agent.train = agent.train, False
    results = [run_game(g, agent, fill_config=fill_cfg, intensity=intensity,
                        config=eng_cfg) for g in games]
    agent.train = agent_train
    _, summary = aggregate(results)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sport", default="nba")
    ap.add_argument("--season", default="2024-25")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--val-games", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--pregame-hours", type=float, default=6.0)
    ap.add_argument("--size", type=float, default=100.0)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--inv-level-penalty", type=float, default=0.0005,
                    help="per-step coefficient applied to squared inventory")
    ap.add_argument("--terminal", choices=["liquidate", "settle", "carry"],
                    default="settle")
    ap.add_argument("--eps0", type=float, default=0.15)
    ap.add_argument("--eps1", type=float, default=0.02)
    ap.add_argument("--latent", action="store_true")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    calib = Path("data/calib") / f"{args.sport}_{args.season}_intensity.json"
    intensity = load_intensity(calib) if calib.exists() else IntensityModel()

    games = []
    for g in iter_games("data/raw", args.sport, args.season,
                        pregame_hours=args.pregame_hours):
        games.append(g)
        if len(games) >= args.games + args.val_games:
            break
    rng = np.random.default_rng(args.seed)
    rng.shuffle(games)
    val, train = games[: args.val_games], games[args.val_games:]
    logging.info("train=%d val=%d games", len(train), len(val))

    fill_cfg = FillConfig(rho=0.5, latent=args.latent, seed=11)
    eng_cfg = EngineConfig(terminal_mode=args.terminal)

    agent = SpoonerMM(SpoonerConfig(size=args.size, eta=args.eta,
                                    inv_level_penalty=args.inv_level_penalty,
                                    epsilon=args.eps0, seed=args.seed))
    history, best_val, best_path = [], -np.inf, None
    out_path = model_path()

    for epoch in range(args.epochs):
        frac = epoch / max(args.epochs - 1, 1)
        agent.cfg.epsilon = args.eps0 + (args.eps1 - args.eps0) * frac
        agent.cfg.alpha = 0.02 * (1.0 - 0.7 * frac)
        rng.shuffle(train)
        t0, ep_rewards = time.time(), []
        for i, g in enumerate(train):
            run_game(g, agent, fill_config=fill_cfg, intensity=intensity, config=eng_cfg)
            ep_rewards.append(agent.episode_reward)
            if (i + 1) % 100 == 0:
                logging.info("epoch %d: %d/%d games, mean psi=%.2f",
                             epoch, i + 1, len(train), np.mean(ep_rewards[-100:]))
        summary = evaluate(agent, val, fill_cfg, intensity, eng_cfg)
        logging.info(
            "epoch %d done in %.0fs eps=%.3f | val pnl/game=%.2f USD sharpe=%.2f "
            "mean|inv|=%.0f",
            epoch, time.time() - t0, agent.cfg.epsilon,
            summary["mean_pnl_per_game_usd"], summary["game_sharpe"],
            summary["mean_abs_inv"],
        )
        history.append({"epoch": epoch, "mean_psi": float(np.mean(ep_rewards)),
                        **{f"val_{k}": v for k, v in summary.items()}})
        if summary["mean_pnl_per_game_usd"] > best_val:
            best_val = summary["mean_pnl_per_game_usd"]
            agent.save(out_path)
            best_path = out_path
            logging.info("saved best model -> %s", out_path)

    pd.DataFrame(history).to_csv(out_path.with_suffix(".history.csv"), index=False)
    print(f"\nbest val pnl/game: {best_val:.2f} USD; model: {best_path}")


if __name__ == "__main__":
    main()
