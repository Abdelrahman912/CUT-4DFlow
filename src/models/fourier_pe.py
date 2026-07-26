"""Fourier-feature positional encodings (temporal and spatial).

Positions are encoded as normalized coordinates in [0, 1) through K sin/cos harmonics and
projected to ``d_model`` by a complex linear layer. Using normalized coordinates makes the
encoding subject-invariant across variable cardiac lengths and spatial sizes.
"""

import math

import torch
import torch.nn as nn

from src.models.complex_ops import ComplexLinear


class FourierCardiacPhasePE(nn.Module):
    """Cardiac-phase encoding of ``phi = frame / N_cycle`` in [0, 1)."""

    def __init__(self, d_model: int, K: int = 4, learnable: bool = False):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.learnable = bool(learnable)
        freqs = torch.arange(1, K + 1, dtype=torch.float32)
        if self.learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs, persistent=False)
        self.complex_linear = ComplexLinear(2 * K, d_model, bias=True)

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        """phases: (..., T) in [0, 1)  ->  (..., T, d_model) complex."""
        ks = self.freqs.to(device=phases.device, dtype=phases.dtype)
        angles = 2 * math.pi * phases.unsqueeze(-1) * ks
        feats = torch.cat([angles.cos(), angles.sin()], dim=-1).to(torch.complex64)
        return self.complex_linear(feats)


class FourierSpatialPE(nn.Module):
    """Separable spatial encoding of normalized row/column positions.

    Each axis is encoded independently and added; the same anatomical position maps to the
    same encoding regardless of the per-subject grid size.
    """

    def __init__(self, d_model: int, K: int = 4, learnable: bool = False):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.learnable = bool(learnable)
        freqs = torch.arange(1, K + 1, dtype=torch.float32)
        if self.learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs, persistent=False)
        self.proj_h = ComplexLinear(2 * K, d_model, bias=True)
        self.proj_w = ComplexLinear(2 * K, d_model, bias=True)

    @staticmethod
    def _fourier(phi, ks):
        angles = 2 * math.pi * phi.unsqueeze(-1) * ks
        feats = torch.cat([angles.cos(), angles.sin()], dim=-1)
        return feats.to(torch.complex64)

    def forward(self, Hp: int, Wp: int, device) -> tuple:
        """Returns (pe_h, pe_w) of shapes (Hp, d) and (Wp, d), both complex."""
        ks = self.freqs.to(device=device, dtype=torch.float32)
        phi_h = torch.arange(Hp, device=device, dtype=torch.float32) / Hp
        phi_w = torch.arange(Wp, device=device, dtype=torch.float32) / Wp
        pe_h = self.proj_h(self._fourier(phi_h, ks))
        pe_w = self.proj_w(self._fourier(phi_w, ks))
        return pe_h, pe_w
