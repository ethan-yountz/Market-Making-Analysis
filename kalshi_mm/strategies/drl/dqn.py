"""Double DQN with a dueling head, n-step returns, and uniform replay.

Deliberate simplifications vs the full rainbow stack: uniform replay instead
of prioritized (the dueling+double+n-step combination captures most of the
gain at this scale and keeps the code auditable).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from kalshi_mm.strategies.drl.networks import DuelingQNet


@dataclass
class DQNConfig:
    n_obs: int = 8
    n_actions: int = 13
    frame_stack: int = 32
    hidden: int = 128
    gamma: float = 0.999
    n_step: int = 3
    lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 300_000
    target_sync: int = 2_000
    eps0: float = 1.0
    eps1: float = 0.05
    eps_decay_steps: int = 150_000
    seed: int = 0


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, int], seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

    def push(self, obs, action, reward, next_obs, done) -> None:
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def sample(self, batch: int):
        n = len(self)
        ix = self.rng.integers(0, n, size=batch)
        return (self.obs[ix], self.actions[ix], self.rewards[ix],
                self.next_obs[ix], self.dones[ix])


class DQNAgent:
    def __init__(self, cfg: DQNConfig, device: str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        self.q = DuelingQNet(cfg.n_obs, cfg.n_actions, cfg.frame_stack, cfg.hidden).to(self.device)
        self.target = DuelingQNet(cfg.n_obs, cfg.n_actions, cfg.frame_stack, cfg.hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, (cfg.frame_stack, cfg.n_obs), cfg.seed)
        self.nstep_queue: deque = deque()
        self.steps = 0
        self.rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------ acting

    def epsilon(self) -> float:
        c = self.cfg
        f = min(self.steps / c.eps_decay_steps, 1.0)
        return c.eps0 + (c.eps1 - c.eps0) * f

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.cfg.n_actions))
        t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        return int(self.q(t).argmax(dim=1).item())

    # ------------------------------------------------------------ n-step

    def observe(self, obs, action, reward, next_obs, done) -> None:
        """Accumulate n-step transitions, push matured ones to replay."""
        c = self.cfg
        self.nstep_queue.append((obs, action, reward, next_obs, done))
        if len(self.nstep_queue) >= c.n_step or done:
            while self.nstep_queue:
                r, discount = 0.0, 1.0
                for (_, _, ri, _, di) in self.nstep_queue:
                    r += discount * ri
                    discount *= c.gamma
                    if di:
                        break
                o0, a0, _, _, _ = self.nstep_queue[0]
                _, _, _, on, dn = self.nstep_queue[min(len(self.nstep_queue), c.n_step) - 1]
                self.buffer.push(o0, a0, r, on, dn)
                self.nstep_queue.popleft()
                if not done:
                    break
        self.steps += 1

    # ------------------------------------------------------------ learning

    def train_step(self) -> float | None:
        c = self.cfg
        if len(self.buffer) < max(c.batch_size * 4, 5_000):
            return None
        obs, act, rew, nxt, done = self.buffer.sample(c.batch_size)
        obs_t = torch.from_numpy(obs).to(self.device)
        nxt_t = torch.from_numpy(nxt).to(self.device)
        act_t = torch.from_numpy(act).to(self.device)
        rew_t = torch.from_numpy(rew).to(self.device)
        done_t = torch.from_numpy(done).to(self.device)

        q = self.q(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best = self.q(nxt_t).argmax(dim=1, keepdim=True)        # double DQN
            q_next = self.target(nxt_t).gather(1, best).squeeze(1)
            gamma_n = c.gamma ** c.n_step
            target = rew_t + gamma_n * q_next * (1.0 - done_t)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        if self.steps % c.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    # ------------------------------------------------------------ persistence

    def save(self, path) -> None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": self.q.state_dict(), "cfg": vars(self.cfg)}, path
        )
