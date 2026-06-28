"""Train the deep RL market maker (dueling double DQN, CUDA if available).

    python scripts/05_train_drl.py --season 2024-25 --steps 200000 --reward spooner
    python scripts/05_train_drl.py --reward raw            # ablation
    python scripts/05_train_drl.py --reward inv_penalty    # ablation

Each episode is one pregame window. Validation = greedy rollouts on held-out
games every --eval-every steps; the best-by-validation checkpoint is kept.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalshi_mm.calib.intensity import load_intensity
from kalshi_mm.data.build import iter_games
from kalshi_mm.eval.metrics import aggregate
from kalshi_mm.sim.engine import EngineConfig, run_game
from kalshi_mm.sim.fills import FillConfig, IntensityModel
from kalshi_mm.strategies.drl.dqn import DQNAgent, DQNConfig
from kalshi_mm.strategies.drl.env import N_OBS, KalshiMMEnv
from kalshi_mm.strategies.drl.policy import DRLMM
from kalshi_mm.strategies.drl.rewards import RewardConfig, RewardFn
from kalshi_mm.strategies.spooner_rl import N_ACTIONS


def evaluate(model_path, games, fill_cfg, intensity, eng_cfg, size):
    strat = DRLMM(model_path, size=size)
    results = [run_game(g, strat, fill_config=fill_cfg, intensity=intensity,
                        config=eng_cfg) for g in games]
    _, summary = aggregate(results)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sport", default="nba")
    ap.add_argument("--season", default="2024-25")
    ap.add_argument("--games", type=int, default=900)
    ap.add_argument("--val-games", type=int, default=60)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--pregame-hours", type=float, default=6.0)
    ap.add_argument("--size", type=float, default=100.0)
    ap.add_argument("--reward", default="spooner", choices=["raw", "inv_penalty", "spooner"])
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--lambda-inv", type=float, default=0.001)
    ap.add_argument("--inv-level-penalty", type=float, default=0.0005,
                    help="per-step lambda*inv^2 inventory-level penalty (fix #3)")
    ap.add_argument("--reward-clip", type=float, default=5.0,
                    help="clip |per-step reward| in cents/contract; stabilizes "
                         "the DQN against terminal/inventory spikes (fix #1)")
    ap.add_argument("--terminal", choices=["liquidate", "settle", "carry"],
                    default="settle",
                    help="terminal valuation during training; 'settle' removes "
                         "the artificial liquidation-fee spike (fix #1/#2)")
    ap.add_argument("--terminal-kappa", type=float, default=0.0)
    ap.add_argument("--frame-stack", type=int, default=32)
    ap.add_argument("--tick-seconds", type=float, default=30.0)
    ap.add_argument("--latent", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.info("device: %s", "cuda" if torch.cuda.is_available() else "cpu")

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
    eng_cfg = EngineConfig(terminal_mode=args.terminal, tick_interval_s=args.tick_seconds)
    reward = RewardFn(RewardConfig(name=args.reward, eta=args.eta,
                                   lambda_inv=args.lambda_inv,
                                   inv_level_penalty=args.inv_level_penalty,
                                   terminal_kappa=args.terminal_kappa,
                                   clip=args.reward_clip,
                                   size=args.size))
    env = KalshiMMEnv(train, reward, frame_stack=args.frame_stack, size=args.size,
                      fill_config=fill_cfg, intensity=intensity,
                      engine_config=eng_cfg, seed=args.seed)
    agent = DQNAgent(DQNConfig(n_obs=N_OBS, n_actions=N_ACTIONS,
                               frame_stack=args.frame_stack,
                               eps_decay_steps=args.steps // 2, seed=args.seed))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path("models") / f"drl_{stamp}_{args.reward}.pt"
    best_val, history = -np.inf, []
    obs, _ = env.reset()
    ep_reward, ep_rewards, losses = 0.0, [], []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        action = agent.act(obs)
        next_obs, r, terminated, _, info = env.step(action)
        agent.observe(obs, action, r, next_obs, terminated)
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)
        ep_reward += r
        obs = next_obs
        if terminated:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()

        if step % 5_000 == 0:
            logging.info(
                "step %d/%d eps=%.2f ep_psi(last20)=%.2f loss=%.4f (%.0f steps/s)",
                step, args.steps, agent.epsilon(),
                float(np.mean(ep_rewards[-20:])) if ep_rewards else float("nan"),
                float(np.mean(losses[-200:])) if losses else float("nan"),
                step / (time.time() - t0),
            )
        if step % args.eval_every == 0 or step == args.steps:
            tmp = out.with_suffix(".tmp.pt")
            agent.save(tmp)
            summary = evaluate(tmp, val, fill_cfg, intensity, eng_cfg, args.size)
            logging.info("eval@%d: val pnl/game=%.2f USD sharpe=%.2f mean|inv|=%.0f",
                         step, summary["mean_pnl_per_game_usd"],
                         summary["game_sharpe"], summary["mean_abs_inv"])
            history.append({"step": step, **{f"val_{k}": v for k, v in summary.items()}})
            if summary["mean_pnl_per_game_usd"] > best_val:
                best_val = summary["mean_pnl_per_game_usd"]
                agent.save(out)
                logging.info("saved best model -> %s", out)

    pd.DataFrame(history).to_csv(out.with_suffix(".history.csv"), index=False)
    tmp = out.with_suffix(".tmp.pt")
    if tmp.exists():
        tmp.unlink()
    print(f"\nbest val pnl/game: {best_val:.2f} USD; model: {out}")


if __name__ == "__main__":
    main()
