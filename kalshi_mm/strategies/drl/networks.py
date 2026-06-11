"""Dueling Q-network over stacked feature frames.

A small temporal-conv encoder reads the (T, F) history (the stand-in for an
order-book image in the LOB papers), followed by dueling value/advantage
heads (Wang et al. 2016). Sized to train comfortably on a single consumer
GPU in minutes-to-hours, not days.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DuelingQNet(nn.Module):
    def __init__(self, n_obs: int, n_actions: int, frame_stack: int, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_obs, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),  # 64*4
        )
        enc_out = 64 * 4
        self.value = nn.Sequential(
            nn.Linear(enc_out, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(enc_out, hidden), nn.ReLU(), nn.Linear(hidden, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> conv over time expects (B, F, T)
        z = self.encoder(x.transpose(1, 2))
        v = self.value(z)
        a = self.advantage(z)
        return v + a - a.mean(dim=1, keepdim=True)
