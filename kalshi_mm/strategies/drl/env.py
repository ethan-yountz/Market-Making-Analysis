"""Gymnasium environment over the backtest engine.

Observation: a (frame_stack, N_OBS) float32 tensor — the recent history of
book/flow/inventory features. Frame stacking stands in for order-book depth
(unavailable historically); once the websocket recorder has accumulated real
LOB data, depth features can be appended without touching the engine.

Action: the same discrete ladder the Spooner agent uses (shared mapping), so
strategy comparisons are apples-to-apples.
"""

from __future__ import annotations

import math
from collections import deque

import gymnasium as gym
import numpy as np

from kalshi_mm.data.build import GameData
from kalshi_mm.sim.engine import EngineConfig, MarketState, episode
from kalshi_mm.sim.fees import FeeSchedule
from kalshi_mm.sim.fills import FillConfig, IntensityModel
from kalshi_mm.strategies.drl.rewards import RewardFn
from kalshi_mm.strategies.spooner_rl import N_ACTIONS, ladder_quotes

N_OBS = 8


def obs_features(s: MarketState, prev_mid: float, max_inv: float) -> np.ndarray:
    mid = s.mid_c if not math.isnan(s.mid_c) else 50.0
    spread = s.spread_c if not math.isnan(s.spread_c) else 2.0
    dmid = 0.0 if math.isnan(prev_mid) else mid - prev_mid
    return np.array(
        [
            mid / 100.0,
            np.clip((spread - 1.0) / 4.0, 0, 1),
            np.clip(s.inventory / max_inv, -1, 1),
            np.clip(s.t_to_tip_s / (6 * 3600.0), 0, 1),
            math.tanh(s.flow_5m / 300.0),
            math.tanh(s.vol_c_per_sqrt_min / 1.5),
            abs(mid - 50.0) / 49.0,
            math.tanh(dmid / 2.0),
        ],
        dtype=np.float32,
    )


class KalshiMMEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        games: list[GameData],
        reward_fn: RewardFn,
        frame_stack: int = 32,
        size: float = 100.0,
        no_quote_band_c: float = 3.0,
        fill_config: FillConfig | None = None,
        intensity: IntensityModel | None = None,
        engine_config: EngineConfig | None = None,
        fees: FeeSchedule | None = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.games = games
        self.reward_fn = reward_fn
        self.frame_stack = frame_stack
        self.size = size
        self.band = no_quote_band_c
        self.fill_config = fill_config or FillConfig()
        self.intensity = intensity
        self.engine_config = engine_config or EngineConfig()
        self.fees = fees
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self._order: list[int] = []
        self._gen = None
        self._state: MarketState | None = None
        self._frames: deque | None = None
        self._prev_mid = math.nan
        self._result = None

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(frame_stack, N_OBS), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

    # ------------------------------------------------------------ helpers

    def _next_game(self) -> GameData:
        if not self._order:
            self._order = list(range(len(self.games)))
            if self.shuffle:
                self.rng.shuffle(self._order)
        return self.games[self._order.pop()]

    def _advance(self, qs) -> MarketState:
        """Send quotes, then keep stepping past mid-less states."""
        state = self._gen.send(qs)
        while math.isnan(state.mid_c) and not state.done:
            state = self._gen.send(None)
        return state

    def _obs(self) -> np.ndarray:
        return np.stack(self._frames).astype(np.float32)

    # ------------------------------------------------------------ gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        game = self._next_game()
        self._gen = episode(
            game,
            fees=self.fees,
            fill_config=self.fill_config,
            intensity=self.intensity,
            config=self.engine_config,
        )
        self._result = None
        self._prev_mid = math.nan
        try:
            state = next(self._gen)
            while math.isnan(state.mid_c) and not state.done:
                state = self._gen.send(None)
        except StopIteration as stop:  # degenerate game with no usable states
            self._result = stop.value
            return self.reset(seed=seed, options=options)
        self._state = state
        f = obs_features(state, self._prev_mid, self.engine_config.max_inventory)
        self._frames = deque([f] * self.frame_stack, maxlen=self.frame_stack)
        self._prev_mid = state.mid_c
        return self._obs(), {"ticker": game.ticker}

    def step(self, action: int):
        assert self._state is not None
        prev = self._state
        qs = ladder_quotes(int(action), prev, self.size, self.band)
        terminated = False
        reward = 0.0
        try:
            state = self._advance(qs)
            reward = self.reward_fn.step(prev, state)
            if state.done:
                reward += self.reward_fn.terminal(state)
                terminated = True
                # exhaust the generator to collect the GameResult
                try:
                    self._gen.send(None)
                except StopIteration as stop:
                    self._result = stop.value
        except StopIteration as stop:
            self._result = stop.value
            state = prev
            terminated = True
        self._state = state
        self._frames.append(
            obs_features(state, self._prev_mid, self.engine_config.max_inventory)
        )
        self._prev_mid = state.mid_c
        info = {"equity_c": state.equity_c, "inventory": state.inventory}
        if terminated and self._result is not None:
            info["game_result"] = self._result
        return self._obs(), float(reward), terminated, False, info
