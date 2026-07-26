"""Unrolled reconstruction cascade.

Wires a shared (weight-tied) complex transformer denoiser together with per-stage
data-consistency (DC) and weighted-average (WA) blocks. Per stage:

    delta   = denoiser(x_curr)
    x_pre   = x_curr + delta
    Sx      = DC(x_curr)             # DC reads the previous iterate
    x_next  = WA(x_pre, Sx)

Optional acceleration conditioning adds a per-stage offset to the WA gate.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as _ckpt

from src.models.transformer_denoiser import CMRxTransformerDenoiser
from src.models.dc_wa import dataConsistencyTerm, weightedAverageTerm
from src.models.conditioning import ConditioningMLP, ConditioningTable


class CMRxTransformerCascade(nn.Module):
    """Unrolled cascade: shared denoiser + per-stage DC and WA.

    forward(x_init, y_theta, sens, mask_theta, phases) -> x_recon
      x_init   : (B, V, T, SPE, PE) complex64 zero-filled image
      y_theta  : (B, V, C, T, SPE, PE) complex64 acquired k-space
      sens     : (B, C, SPE, PE) complex64 coil sensitivities
      phases   : (B, T) normalized cardiac phase in [0, 1)
    """

    def __init__(
        self,
        n_stages: int = 10,
        d_model: int = 48,
        n_heads: int = 4,
        n_blocks: int = 2,
        mlp_ratio: int = 2,
        patch_size: int = 2,
        in_channels: int = 6,
        K_pe: int = 4,
        K_cardiac: int = 4,
        dc_min_v: float = 0.0,
        wa_min_para: float = 0.0,
        init_scheme: str = "uniform",
        pe_learnable: bool = False,
        grad_check: bool = False,
        attn_modes=None,
        conditioning: Optional[dict] = None,
        **_,
    ) -> None:
        super().__init__()
        if n_stages < 1:
            raise ValueError(f"n_stages must be >= 1; got {n_stages}")
        if K_pe != K_cardiac:
            raise ValueError(f"requires K_pe == K_cardiac; got {K_pe}, {K_cardiac}")

        self.n_stages = int(n_stages)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_blocks = int(n_blocks)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.K_pe = int(K_pe)
        self.K_cardiac = int(K_cardiac)
        self.init_scheme = init_scheme
        self.pe_learnable = bool(pe_learnable)
        self.grad_check = bool(grad_check)   # recompute the denoiser in backward to save memory

        # shared denoiser, unrolled n_stages times
        self.denoiser = CMRxTransformerDenoiser(
            in_channels=self.in_channels, d_model=self.d_model, n_heads=self.n_heads,
            n_blocks=self.n_blocks, mlp_ratio=int(mlp_ratio), patch_size=self.patch_size,
            fourier_K=self.K_pe, init_scheme=self.init_scheme, pe_learnable=self.pe_learnable,
            attn_modes=attn_modes,
        )

        # one independent DC and WA per stage
        self.dc = nn.ModuleList([dataConsistencyTerm(-2.2, min_v=dc_min_v) for _ in range(self.n_stages)])
        self.wa = nn.ModuleList([weightedAverageTerm(-2.2, min_para=wa_min_para) for _ in range(self.n_stages)])

        # optional conditioning (zero-initialised: identical to baseline at start)
        cond = conditioning or {}
        self.conditioning_enabled = bool(cond.get('enabled', False))
        self.cond_mode = str(cond.get('mode', 'mlp')).lower()
        self.conditioning_inputs = list(cond.get('inputs', ['R'])) if self.conditioning_enabled else []
        self.use_film = (bool(cond.get('use_film', False))
                         and self.conditioning_enabled and self.cond_mode != 'table')
        self.cond_mlp = None
        self.cond_table = None
        if self.conditioning_enabled and self.cond_mode == 'table':
            self.cond_table = ConditioningTable(
                r_values=cond.get('r_values', [10, 20, 30, 40, 50]),
                n_stages=self.n_stages, anchor_R=cond.get('anchor_R'),
            )
        elif self.conditioning_enabled:
            self.cond_mlp = ConditioningMLP(
                input_dim=len(self.conditioning_inputs), n_stages=self.n_stages,
                n_blocks=self.n_blocks, d_model=self.d_model, use_film=self.use_film,
            )

    def forward(
        self,
        x_init: torch.Tensor,
        y_theta: torch.Tensor,
        sens: torch.Tensor,
        mask_theta: Optional[torch.Tensor],
        phases: torch.Tensor,
        return_stages: bool = False,
        m: Optional[torch.Tensor] = None,
        R: Optional[int] = None,
    ) -> torch.Tensor:
        if x_init.dim() != 5 or not torch.is_complex(x_init):
            raise ValueError(f"x_init must be 5-D complex; got dim={x_init.dim()}, dtype={x_init.dtype}")
        B, V, T, SPE, PE = x_init.shape
        if V != self.in_channels:
            raise ValueError(f"x_init channel dim V={V} != in_channels={self.in_channels}")
        if phases.shape != (B, T):
            raise ValueError(f"phases must be (B={B}, T={T}); got {tuple(phases.shape)}")

        # conditioning offsets (computed once). Table conditions the WA gate only.
        delta_v = delta_para = gammas = None
        if self.cond_table is not None and R is not None:
            delta_para = self.cond_table(R)
        elif self.conditioning_enabled and self.cond_mlp is not None and m is not None:
            off = self.cond_mlp(m)
            delta_v = off['delta_v'][0]
            delta_para = off['delta_para'][0]
            if self.use_film and off['gamma'] is not None:
                gammas = off['gamma'][0]

        x_curr = x_init
        stages = [] if return_stages else None
        for i in range(self.n_stages):
            if self.grad_check and self.training:
                delta = _ckpt.checkpoint(self.denoiser, x_curr, phases, gammas, use_reentrant=False)
            else:
                delta = self.denoiser(x_curr, phases, gammas)
            x_pre = x_curr + delta

            # DC reads the previous iterate x_curr. Reshape k0/coils to (N, C, V, T, D, H, W).
            k0 = y_theta.unsqueeze(4) if y_theta.dim() == 6 else y_theta
            c = sens.unsqueeze(2) if sens.dim() == 4 else sens
            k0_swapped = torch.swapaxes(k0, 1, 2)
            Sx = self.dc[i].perform(x_curr, k0_swapped, c,
                                    delta=None if delta_v is None else delta_v[i])
            if Sx.dim() == 6:
                Sx = Sx[:, :, :, 0]

            x_curr = self.wa[i].perform(x_pre, Sx, delta=None if delta_para is None else delta_para[i])
            if return_stages:
                stages.append(x_curr)

        if return_stages:
            return torch.stack(stages, dim=0)
        return x_curr
