"""
Fourier-feature normalized-cardiac-phase positional encoding.

Replaces the fixed-size discrete lookup `nn.Parameter(T, d)` with a
subject-invariant continuous encoding that handles variable N_cycle
across subjects (15..24 in the aortic dataset) and random window
start positions (with wrap-around).

Encodes phi = frame_idx / N_cycle in [0, 1) through K sin/cos
harmonics, projected via a learnable complex linear to d_model.
"""

import math

import torch
import torch.nn as nn

from src.models.complex_ops import ComplexLinear


class FourierCardiacPhasePE(nn.Module):
    """Continuous cardiac-phase positional encoding (K Fourier harmonics)."""

    def __init__(self, d_model: int, K: int = 4, learnable: bool = False):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.learnable = bool(learnable)
        # Frequencies: fixed integer harmonics [1..K] (default) OR learnable
        # (LFF, Li et al. NeurIPS 2021) — the K *values* are trained, count fixed.
        freqs = torch.arange(1, K + 1, dtype=torch.float32)
        if self.learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs, persistent=False)
        self.complex_linear = ComplexLinear(2 * K, d_model, bias=True)
        # complex_linear weight is initialized by ComplexLinear to N(0, 0.02)
        # and bias to zero — that is the default for "everything else" in
        # _init_weights. Leave it alone.

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        """
        phases: (..., T) real tensor of normalized cardiac phases in [0, 1).
        returns: (..., T, d_model) complex
        """
        ks = self.freqs.to(device=phases.device, dtype=phases.dtype)
        angles = 2 * math.pi * phases.unsqueeze(-1) * ks          # (..., T, K)
        fourier_feats = torch.cat([angles.cos(), angles.sin()], dim=-1)
        fourier_feats = fourier_feats.to(torch.complex64)          # (..., T, 2K) complex
        return self.complex_linear(fourier_feats)                  # (..., T, d_model)


class FourierSpatialPE(nn.Module):
    """Separable Fourier-feature spatial positional encoding for variable-size patch grids.

    Encodes normalized positions phi_h = h_idx / Hp and phi_w = w_idx / Wp
    in [0, 1) through K Fourier harmonics, projected to d_model via complex linear.
    Each axis projected independently; tokens get pe_h + pe_w (additive).

    Subject-invariant: the same anatomical mid-aorta position (~ phi=0.5) maps
    to the same PE regardless of whether Hp=46 or Hp=60, which is the right
    inductive bias since the FOV is roughly anatomy-aligned across subjects.
    """

    def __init__(self, d_model: int, K: int = 4, learnable: bool = False):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.learnable = bool(learnable)
        # Fixed integer harmonics [1..K] (default) OR learnable frequencies
        # (LFF, Li et al. NeurIPS 2021): the K values are trained, count fixed.
        freqs = torch.arange(1, K + 1, dtype=torch.float32)
        if self.learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs, persistent=False)
        self.proj_h = ComplexLinear(2 * K, d_model, bias=True)
        self.proj_w = ComplexLinear(2 * K, d_model, bias=True)

    @staticmethod
    def _fourier(phi, ks):
        # phi: (N,)  ks: (K,)  → returns (N, 2K) complex
        angles = 2 * math.pi * phi.unsqueeze(-1) * ks
        feats = torch.cat([angles.cos(), angles.sin()], dim=-1)
        return feats.to(torch.complex64)

    def forward(self, Hp: int, Wp: int, device) -> tuple:
        """Returns (pe_h, pe_w) of shapes (Hp, d) and (Wp, d), both complex."""
        ks = self.freqs.to(device=device, dtype=torch.float32)
        phi_h = torch.arange(Hp, device=device, dtype=torch.float32) / Hp
        phi_w = torch.arange(Wp, device=device, dtype=torch.float32) / Wp
        pe_h = self.proj_h(self._fourier(phi_h, ks))   # (Hp, d) complex
        pe_w = self.proj_w(self._fourier(phi_w, ks))   # (Wp, d) complex
        return pe_h, pe_w
