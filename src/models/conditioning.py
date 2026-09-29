"""Ada-MoDL-style multi-R (+ optional B0) conditioning for the CUT-4DFlow cascade.

A single small MLP reads the conditional vector ``m`` and emits, ONCE per forward:
  - ``delta_v``    (n_stages,)          logit offsets for the per-stage DC gates
  - ``delta_para`` (n_stages,)          logit offsets for the per-stage WA gates
  - ``gamma``      (n_blocks, d_model)  real per-channel scale for the denoiser's
                                        block norms (FiLM scale, Phase 2; only when
                                        ``use_film=True``)

``m = [R/50, (B0/3)]`` — normalized undersampling rate and (optionally) field
strength; both are known exactly at train and test time, so no estimation.

Injection (see cascade / dc_wa / complex_norm):
  DC   : v[i]    = sigmoid(noise_lvl[i] + delta_v[i])      (offset, pre-sigmoid)
  WA   : para[i] = sigmoid(para0[i]     + delta_para[i])
  norm : out     = (1 + gamma[block]) * (zeta.whiten(x) + beta)   (real scale, phase-safe)

Zero-init on the final layer of every head => at step 0 all offsets/gamma are 0, so
the conditioned network is BIT-IDENTICAL to the unconditioned baseline and only learns
to deviate. Grounding: Ada-MoDL (Pramanik 2023), FiLM (Perez 2018), UPCMR, FlowVN.
"""
from __future__ import annotations

import torch
import torch.nn as nn

R_MAX = 50.0        # challenge undersampling range upper bound
B0_MAX = 3.0        # field strength normalizer (3.0 T)


def build_m(inputs, R, B0=None, device=None):
    """Normalized conditioning vector ``(1, len(inputs))`` from raw R / B0.

    ``inputs`` : ordered list from {'R', 'B0'}. Empty/None -> returns None.
    """
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
    """R-conditioning as a LOOKUP TABLE on the WA gate only (v2).

    Why a table and not an MLP: R is a *categorical* variable with 5 values
    (10/20/30/40/50). A table is the EXACT representation of R -> delta_para;
    an MLP approximates a continuous curve that is never queried between the
    5 points, and its hidden layer is what let p1 go wrong (see below).

    ANCHORING (the important part). ``para`` and ``delta_para`` both add into
    the same logit, ``mu = sigmoid(para + delta_para)``, so they are redundant
    and can fight over the same job. In p1 the conditioning head won and
    absorbed a large CONSTANT (-5.96 +/- 0.21 -> ~96% constant, ~4% actual
    R-tilt), which drove the gate toward saturation. Subtracting the anchor row
    makes a constant algebraically impossible:

        delta_para(R) = table[idx(R)] - table[idx(anchor_R)]

    so ``para`` alone owns the operating point and the table owns only the
    tilt across R. Zero-init => delta_para == 0 for every R at step 0, i.e. the
    conditioned network starts BIT-IDENTICAL to the unconditioned baseline.

    Cost: len(r_values) x n_stages scalars (5 x 10 = 50), vs 660 for the MLP.
    Use :meth:`as_table` after training to read the learned tilt directly.
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
        # Unseen R (shouldn't happen for this challenge) -> nearest, so inference
        # degrades gracefully instead of raising.
        return min(range(len(self.r_values)), key=lambda i: abs(self.r_values[i] - R))

    def forward(self, R) -> torch.Tensor:
        """R: int-like. Returns the anchored ``(n_stages,)`` delta_para."""
        i = self.index_of(R)
        return self.dp_table[i] - self.dp_table[self.anchor_idx]

    @torch.no_grad()
    def as_table(self) -> torch.Tensor:
        """Anchored table ``(n_R, n_stages)`` — print this to read the learned tilt."""
        return self.dp_table - self.dp_table[self.anchor_idx:self.anchor_idx + 1]

    def extra_repr(self) -> str:
        return f'r_values={self.r_values}, anchor_R={self.anchor_R}, n_stages={self.n_stages}'


class ConditioningMLP(nn.Module):
    """m -> {delta_v, delta_para, gamma}.

    Body: 2 shared FC (hidden=16) + ReLU, then one FC head per output (=3 FC deep;
    Ada-MoDL uses 5x16, but a 1-2-D input needs less). Head final layers zero-init.
    """

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
        self.head_dv = nn.Linear(hidden, self.n_stages)             # delta_v
        self.head_dp = nn.Linear(hidden, self.n_stages)             # delta_para
        self.head_gamma = (
            nn.Linear(hidden, self.n_blocks * self.d_model) if self.use_film else None
        )

        # Zero-init the FINAL layer of every head -> identity at step 0.
        for head in (self.head_dv, self.head_dp, self.head_gamma):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, m: torch.Tensor) -> dict:
        """m: (B, input_dim) real. Returns dict with (B, ...) heads; gamma or None."""
        h = self.body(m)
        out = {
            'delta_v': self.head_dv(h),        # (B, n_stages)
            'delta_para': self.head_dp(h),     # (B, n_stages)
            'gamma': None,
        }
        if self.head_gamma is not None:
            out['gamma'] = self.head_gamma(h).view(-1, self.n_blocks, self.d_model)
        return out
