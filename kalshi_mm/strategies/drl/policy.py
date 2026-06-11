"""Frozen DRL policy as a Strategy, for the standard backtest runner."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
import torch

from kalshi_mm.data.build import GameData
from kalshi_mm.sim.engine import MarketState, QuoteSet, Strategy
from kalshi_mm.strategies.drl.env import obs_features
from kalshi_mm.strategies.drl.networks import DuelingQNet
from kalshi_mm.strategies.spooner_rl import ladder_quotes


class DRLMM(Strategy):
    def __init__(self, model_path: str | Path, size: float = 100.0,
                 no_quote_band_c: float = 3.0, max_inventory: float = 500.0,
                 device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        cfg = ckpt["cfg"]
        self.frame_stack = cfg["frame_stack"]
        self.q = DuelingQNet(cfg["n_obs"], cfg["n_actions"], cfg["frame_stack"],
                             cfg["hidden"]).to(self.device)
        self.q.load_state_dict(ckpt["model"])
        self.q.eval()
        self.size = size
        self.band = no_quote_band_c
        self.max_inv = max_inventory
        self._frames: deque | None = None
        self._prev_mid = math.nan

    def reset(self, game: GameData) -> None:
        self._frames = None
        self._prev_mid = math.nan

    @torch.no_grad()
    def on_event(self, state: MarketState) -> QuoteSet | None:
        if math.isnan(state.mid_c):
            return None
        f = obs_features(state, self._prev_mid, self.max_inv)
        if self._frames is None:
            self._frames = deque([f] * self.frame_stack, maxlen=self.frame_stack)
        else:
            self._frames.append(f)
        self._prev_mid = state.mid_c
        obs = torch.from_numpy(np.stack(self._frames).astype(np.float32))
        action = int(self.q(obs.unsqueeze(0).to(self.device)).argmax(dim=1).item())
        return ladder_quotes(action, state, self.size, self.band)

    @classmethod
    def load_latest(cls, models_dir: str | Path = "models", size: float = 100.0) -> "DRLMM":
        files = sorted(Path(models_dir).glob("drl_*.pt"))
        if not files:
            raise FileNotFoundError(
                "no trained DRL model in models/ - run scripts/05_train_drl.py"
            )
        return cls(files[-1], size=size)
