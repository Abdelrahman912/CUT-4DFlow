"""Acceleration-factor conditioning for the cascade gates.

Two variants:

- ``ConditioningTable``: an anchored lookup table that adds a per-stage offset to the WA gate
  logit as a function of the (categorical) acceleration R. Anchoring subtracts a reference row
  so the table represents only the tilt across R, leaving the base gate to own the operating
  point. Zero-initialised, so the model starts identical to the unconditioned baseline.
- ``ConditioningMLP``: a small MLP mapping a normalized conditioning vector to per-stage gate
  offsets (and, optionally, FiLM scales for the denoiser norms).
"""
from __future__ import annotations

import torch
import torch.nn as nn

R_MAX = 50.0
B0_MAX = 3.0


def build_m(inputs, R, B0=None, device=None):
    """Normalized conditioning vector (1, len(inputs)) from R / B0. Empty inputs -> None."""
    if not inputs:
        return None
    vals = []
    for name in inputs:
        if name == 'R':
            vals.append(float(R) / R_MAX)
        elif name == 'B0':
            if B0 is None:
                raise ValueError("conditioning input 'B0' requested but B0 is None")
            vals.append(float(B0) / B0_MAX)
        else:
            raise ValueError(f"unknown conditioning input {name!r} (expected 'R' or 'B0')")
    return torch.tensor([vals], dtype=torch.float32, device=device)


class ConditioningTable(nn.Module):
    """Anchored per-stage WA-gate offset as a lookup table over R.

    ``delta_para(R) = table[idx(R)] - table[idx(anchor_R)]``; zero-initialised.
    """

    def __init__(self, r_values, n_stages: int, anchor_R=None):
        super().__init__()
        self.r_values = [int(r) for r in r_values]
        if not self.r_values:
            raise ValueError('ConditioningTable needs a non-empty r_values')
        anchor_R = self.r_values[0] if anchor_R is None else int(anchor_R)
        if anchor_R not in self.r_values:
            raise ValueError(f'anchor_R={anchor_R} not in r_values={self.r_values}')
        self.anchor_R = anchor_R
        self.anchor_idx = self.r_values.index(anchor_R)
        self.n_stages = int(n_stages)
        self.dp_table = nn.Parameter(torch.zeros(len(self.r_values), self.n_stages))

    def index_of(self, R) -> int:
        R = int(R)
        if R in self.r_values:
            return self.r_values.index(R)
        return min(range(len(self.r_values)), key=lambda i: abs(self.r_values[i] - R))

    def forward(self, R) -> torch.Tensor:
        i = self.index_of(R)
        return self.dp_table[i] - self.dp_table[self.anchor_idx]

    @torch.no_grad()
    def as_table(self) -> torch.Tensor:
        """Anchored table (n_R, n_stages) for inspecting the learned tilt."""
        return self.dp_table - self.dp_table[self.anchor_idx:self.anchor_idx + 1]

    def extra_repr(self) -> str:
        return f'r_values={self.r_values}, anchor_R={self.anchor_R}, n_stages={self.n_stages}'


class ConditioningMLP(nn.Module):
    """m -> {delta_v, delta_para, gamma}. Head final layers are zero-initialised."""

    def __init__(self, input_dim, n_stages, n_blocks, d_model, hidden=16, use_film=False):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_stages = int(n_stages)
        self.n_blocks = int(n_blocks)
        self.d_model = int(d_model)
        self.use_film = bool(use_film)

        self.body = nn.Sequential(
            nn.Linear(self.input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_dv = nn.Linear(hidden, self.n_stages)
        self.head_dp = nn.Linear(hidden, self.n_stages)
        self.head_gamma = (
            nn.Linear(hidden, self.n_blocks * self.d_model) if self.use_film else None
        )

        for head in (self.head_dv, self.head_dp, self.head_gamma):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, m: torch.Tensor) -> dict:
        h = self.body(m)
        out = {
            'delta_v': self.head_dv(h),
            'delta_para': self.head_dp(h),
            'gamma': None,
        }
        if self.head_gamma is not None:
            out['gamma'] = self.head_gamma(h).view(-1, self.n_blocks, self.d_model)
        return out
