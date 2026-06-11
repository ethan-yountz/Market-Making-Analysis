"""Market making via reinforcement learning, after Spooner et al. (2018).

SARSA(lambda) with tile-coded linear value functions, an action ladder of
quote offsets around the touch, and the paper's key idea: an
*asymmetrically dampened* reward

    psi_t = dPnL_t - max(0, eta * inv_{t-1} * dmid_t)

which strips speculative (trend-riding) inventory gains while keeping
inventory losses, so the agent is pushed toward spread capture instead of
position taking. Kalshi adaptations: time-to-tip and distance-from-50 in the
state (boundedness), episodes are single pregame windows, and the terminal
liquidation cost at tip is part of the final reward.

The agent learns *online inside the engine* — it is just a Strategy whose
on_event both updates SARSA traces and returns the next QuoteSet.
"""

from __future__ import annotations

import json
import math
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kalshi_mm.data.build import GameData
from kalshi_mm.sim.engine import MarketState, Quote, QuoteSet, Strategy

# Action ladder: (bid_offset, ask_offset) in ticks relative to the touch
# (positive = behind/wider, -1 = improve inside the spread when possible),
# plus one-sided quoting and a taker inventory clear.
LADDER: list[tuple[int, int]] = [
    (0, 0), (1, 1), (2, 2), (3, 3),
    (0, 1), (1, 0), (0, 2), (2, 0),
    (-1, -1),
]
A_BID_ONLY = len(LADDER)
A_ASK_ONLY = len(LADDER) + 1
A_CLEAR = len(LADDER) + 2
A_NOQUOTE = len(LADDER) + 3
N_ACTIONS = len(LADDER) + 4

N_FEATURES = 6


class TileCoder:
    """Classic tile coding over [0,1]^D with hashed indices."""

    def __init__(self, n_dims: int, n_tilings: int = 8, bins: int = 6,
                 table_size: int = 2 ** 18, seed: int = 3):
        self.n_tilings = n_tilings
        self.bins = bins
        self.table_size = table_size
        rng = np.random.default_rng(seed)
        self.offsets = rng.random((n_tilings, n_dims)) / bins

    def indices(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        out = np.empty(self.n_tilings, dtype=np.int64)
        for t in range(self.n_tilings):
            b = np.minimum((x + self.offsets[t]) * self.bins, self.bins - 1).astype(np.int64)
            key = (t.to_bytes(2, "little") + b.tobytes())
            out[t] = zlib.crc32(key) % self.table_size
        return out


@dataclass
class SpoonerConfig:
    size: float = 100.0
    eta: float = 0.5                 # asymmetric dampening strength
    alpha: float = 0.02              # learning rate (per tiling)
    lam: float = 0.85                # eligibility trace decay
    gamma: float = 1.0               # undiscounted within an episode
    epsilon: float = 0.15            # exploration (annealed by trainer)
    no_quote_band_c: float = 3.0
    max_inv_norm: float = 500.0
    seed: int = 5


class SpoonerMM(Strategy):
    def __init__(self, config: SpoonerConfig | None = None, train: bool = True):
        self.cfg = config or SpoonerConfig()
        self.train = train
        self.coder = TileCoder(N_FEATURES, seed=self.cfg.seed)
        self.w = np.zeros((N_ACTIONS, self.coder.table_size), dtype=np.float32)
        self.rng = np.random.default_rng(self.cfg.seed)
        self._reset_episode()

    # ------------------------------------------------------------ features

    def _features(self, s: MarketState) -> np.ndarray:
        inv = np.clip(s.inventory / self.cfg.max_inv_norm, -1, 1)
        tau = np.clip(s.t_to_tip_s / (6 * 3600.0), 0, 1)
        spread = 1.0 if math.isnan(s.spread_c) else np.clip((s.spread_c - 1.0) / 4.0, 0, 1)
        flow = math.tanh(s.flow_5m / 300.0)
        vol = math.tanh(s.vol_c_per_sqrt_min / 1.5)
        d50 = abs(s.mid_c - 50.0) / 49.0 if not math.isnan(s.mid_c) else 0.5
        return np.array(
            [(inv + 1) / 2, tau, spread, (flow + 1) / 2, vol, d50], dtype=np.float64
        )

    # ------------------------------------------------------------ Q machinery

    def _q(self, idx: np.ndarray) -> np.ndarray:
        return self.w[:, idx].sum(axis=1)

    def _choose(self, q: np.ndarray) -> int:
        if self.train and self.rng.random() < self.cfg.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        return int(np.argmax(q))

    def _reset_episode(self) -> None:
        self.traces: dict[tuple[int, int], float] = {}
        self.prev_idx: np.ndarray | None = None
        self.prev_action: int | None = None
        self.prev_equity: float = 0.0
        self.prev_inv: float = 0.0
        self.prev_mid: float = math.nan
        self.episode_reward: float = 0.0

    def reset(self, game: GameData) -> None:
        self._reset_episode()

    # ------------------------------------------------------------ reward

    def _reward(self, s: MarketState) -> float:
        dpnl = s.equity_c - self.prev_equity
        if not math.isnan(self.prev_mid) and not math.isnan(s.mid_c):
            dmid = s.mid_c - self.prev_mid
            damp = max(0.0, self.cfg.eta * self.prev_inv * dmid)
        else:
            damp = 0.0
        return (dpnl - damp) / self.cfg.size  # ~cents per contract quoted

    def _update(self, r: float, idx: np.ndarray | None, action: int | None,
                done: bool) -> None:
        """SARSA(lambda) with replacing traces on hashed tile features."""
        if self.prev_idx is None or self.prev_action is None:
            return
        q_prev = float(self.w[self.prev_action, self.prev_idx].sum())
        q_next = 0.0 if done or idx is None else float(self.w[action, idx].sum())
        delta = r + self.cfg.gamma * q_next - q_prev
        for i in self.prev_idx:
            self.traces[(self.prev_action, int(i))] = 1.0  # replacing
        a_eff = self.cfg.alpha / self.coder.n_tilings
        decay = self.cfg.gamma * self.cfg.lam
        dead = []
        for (a, i), e in self.traces.items():
            self.w[a, i] += a_eff * delta * e
            e *= decay
            if e < 1e-3:
                dead.append((a, i))
            else:
                self.traces[(a, i)] = e
        for key in dead:
            del self.traces[key]

    # ------------------------------------------------------------ acting

    def _quotes_for(self, action: int, s: MarketState) -> QuoteSet:
        m = s.mid_c
        if math.isnan(m) or m < 1 + self.cfg.no_quote_band_c or m > 99 - self.cfg.no_quote_band_c:
            return QuoteSet(None, None)
        bid_ref = s.bid_c if not math.isnan(s.bid_c) else m - 1
        ask_ref = s.ask_c if not math.isnan(s.ask_c) else m + 1
        size = self.cfg.size
        if action == A_CLEAR:
            return QuoteSet(None, None, clear=True)
        if action == A_NOQUOTE:
            return QuoteSet(None, None)
        if action == A_BID_ONLY:
            return QuoteSet(Quote(bid_ref, size), None)
        if action == A_ASK_ONLY:
            return QuoteSet(None, Quote(ask_ref, size))
        ob, oa = LADDER[action]
        return QuoteSet(Quote(bid_ref - ob, size), Quote(ask_ref + oa, size))

    def on_event(self, state: MarketState) -> QuoteSet | None:
        if math.isnan(state.mid_c):
            return None
        idx = self.coder.indices(self._features(state))
        q = self._q(idx)
        action = self._choose(q)
        if self.train:
            r = self._reward(state)
            self.episode_reward += r
            self._update(r, idx, action, done=False)
        self.prev_idx, self.prev_action = idx, action
        self.prev_equity, self.prev_inv, self.prev_mid = (
            state.equity_c, state.inventory, state.mid_c,
        )
        return self._quotes_for(action, state)

    def on_game_end(self, state: MarketState) -> None:
        if self.train:
            r = self._reward(state)  # includes terminal liquidation cost
            self.episode_reward += r
            self._update(r, None, None, done=True)

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, w=self.w)
        meta = {k: getattr(self.cfg, k) for k in vars(self.cfg)}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, train: bool = False) -> "SpoonerMM":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        agent = cls(SpoonerConfig(**meta), train=train)
        agent.w = np.load(path if path.suffix else path.with_suffix(".npz"))["w"]
        return agent

    @classmethod
    def load_latest(cls, models_dir: str | Path = "models", size: float | None = None,
                    train: bool = False) -> "SpoonerMM":
        files = sorted(Path(models_dir).glob("spooner_*.npz"))
        if not files:
            raise FileNotFoundError(
                "no trained Spooner model in models/ - run scripts/04_train_spooner.py"
            )
        agent = cls.load(files[-1], train=train)
        if size is not None:
            agent.cfg.size = size
        return agent


def model_path(models_dir: str | Path = "models") -> Path:
    return Path(models_dir) / f"spooner_{time.strftime('%Y%m%d_%H%M%S')}.npz"
