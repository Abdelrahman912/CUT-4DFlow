"""Complex transformer denoiser for one cascade stage.

Patch-embeds the complex input, adds cardiac-phase and spatial Fourier positional encodings,
applies factorized axial transformer blocks (temporal -> row -> column attention + MLP, each
with complex layer norm), and un-embeds to a complex residual of the input shape.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.attention import MultiHeadCAtt
from src.models.complex_norm import ComplexLayerNorm2x2
from src.models.complex_ops import (
    ComplexConv2d,
    ComplexLinear,
    ComplexUpsampleConv2d,
    modReLU,
)
from src.models.fourier_pe import FourierCardiacPhasePE, FourierSpatialPE


class _FactorizedBlock(nn.Module):
    """One factorized complex transformer block (axial attention + MLP)."""

    def __init__(self, d_model: int = 32, n_heads: int = 4, mlp_ratio: int = 2,
                 attn_mode: str = "series"):
        super().__init__()
        if attn_mode not in ("series", "parallel"):
            raise ValueError(f"attn_mode must be 'series' or 'parallel'; got {attn_mode!r}")
        self.attn_mode = attn_mode          # 'series' = axial t->r->c; 'parallel' = summed
        self.norm_t = ComplexLayerNorm2x2(d_model)
        self.attn_t = MultiHeadCAtt(d_model, n_heads)
        self.norm_r = ComplexLayerNorm2x2(d_model)
        self.attn_r = MultiHeadCAtt(d_model, n_heads)
        self.norm_c = ComplexLayerNorm2x2(d_model)
        self.attn_c = MultiHeadCAtt(d_model, n_heads)

        self.norm_mlp = ComplexLayerNorm2x2(d_model)
        d_mlp = d_model * mlp_ratio
        self.mlp_fc1 = ComplexLinear(d_model, d_mlp)
        self.mlp_act = modReLU(d_mlp)
        self.mlp_fc2 = ComplexLinear(d_mlp, d_model)

    def forward(self, z: torch.Tensor, gamma: torch.Tensor | None = None) -> torch.Tensor:
        """z: (B, T, d, Hp, Wp) complex64 -> same shape. ``gamma``: optional per-block scale."""
        B, T, d, Hp, Wp = z.shape

        if self.attn_mode == "parallel":
            z_t = z.permute(0, 3, 4, 1, 2).contiguous().reshape(B * Hp * Wp, T, d)
            dt = self.attn_t(self.norm_t(z_t, gamma)).reshape(B, Hp, Wp, T, d).permute(0, 3, 4, 1, 2).contiguous()
            z_r = z.permute(0, 1, 3, 4, 2).contiguous().reshape(B * T * Hp, Wp, d)
            dr = self.attn_r(self.norm_r(z_r, gamma)).reshape(B, T, Hp, Wp, d).permute(0, 1, 4, 2, 3).contiguous()
            z_c = z.permute(0, 1, 4, 3, 2).contiguous().reshape(B * T * Wp, Hp, d)
            dc = self.attn_c(self.norm_c(z_c, gamma)).reshape(B, T, Wp, Hp, d).permute(0, 1, 4, 3, 2).contiguous()
            z = z + dt + dr + dc
        else:
            z_t = z.permute(0, 3, 4, 1, 2).contiguous().reshape(B * Hp * Wp, T, d)
            z_t = z_t + self.attn_t(self.norm_t(z_t, gamma))
            z = z_t.reshape(B, Hp, Wp, T, d).permute(0, 3, 4, 1, 2).contiguous()

            z_r = z.permute(0, 1, 3, 4, 2).contiguous().reshape(B * T * Hp, Wp, d)
            z_r = z_r + self.attn_r(self.norm_r(z_r, gamma))
            z = z_r.reshape(B, T, Hp, Wp, d).permute(0, 1, 4, 2, 3).contiguous()

            z_c = z.permute(0, 1, 4, 3, 2).contiguous().reshape(B * T * Wp, Hp, d)
            z_c = z_c + self.attn_c(self.norm_c(z_c, gamma))
            z = z_c.reshape(B, T, Wp, Hp, d).permute(0, 1, 4, 3, 2).contiguous()

        z_m = z.permute(0, 1, 3, 4, 2).contiguous().reshape(B * T * Hp * Wp, d)
        z_mlp = self.norm_mlp(z_m, gamma)
        z_mlp = self.mlp_fc1(z_mlp)
        z_mlp = self.mlp_act(z_mlp)
        z_mlp = self.mlp_fc2(z_mlp)
        z_m = z_m + z_mlp
        z = z_m.reshape(B, T, Hp, Wp, d).permute(0, 1, 4, 2, 3).contiguous()
        return z


class CMRxTransformerDenoiser(nn.Module):
    """Complex transformer denoiser.

    Input:  x (B, V, T, SPE, PE) complex64 ; phases (B, T) in [0, 1).
    Output: delta_x (B, V, T, SPE, PE) complex64 residual.
    """

    def __init__(
        self,
        in_channels: int = 6,
        d_model: int = 48,
        n_heads: int = 4,
        n_blocks: int = 2,
        mlp_ratio: int = 2,
        patch_size: int = 2,
        fourier_K: int = 4,
        init_scheme: str = "uniform",
        pe_learnable: bool = False,
        attn_modes=None,
        **_,
    ):
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1; got {n_blocks}")
        self.in_channels = int(in_channels)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_blocks = int(n_blocks)
        self.patch_size = int(patch_size)
        self.fourier_K = int(fourier_K)
        if init_scheme != "uniform":
            raise ValueError(f"init_scheme must be 'uniform'; got {init_scheme!r}")
        self.init_scheme = init_scheme
        self.pe_learnable = bool(pe_learnable)
        if attn_modes is None:
            attn_modes = ["series"] * self.n_blocks
        if len(attn_modes) != self.n_blocks:
            raise ValueError(f"attn_modes must have n_blocks={self.n_blocks} entries; got {attn_modes}")
        self.attn_modes = list(attn_modes)

        self.patch_embed = ComplexConv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.temporal_pe = FourierCardiacPhasePE(d_model, K=fourier_K, learnable=self.pe_learnable)
        self.spatial_pe = FourierSpatialPE(d_model, K=fourier_K, learnable=self.pe_learnable)
        self.blocks = nn.ModuleList([
            _FactorizedBlock(d_model, n_heads, mlp_ratio, attn_mode=self.attn_modes[i])
            for i in range(n_blocks)
        ])
        self.patch_unembed = ComplexUpsampleConv2d(d_model, in_channels, scale_factor=patch_size, kernel_size=3)

        self._init_weights()

    def _init_weights(self) -> None:
        """Every complex weight ~ N(0, sigma=0.012); all biases zero. CLN params keep their own init."""
        sigma_uniform = 0.012
        for name, p in self.named_parameters():
            if any(x in name for x in ['raw_a', 'raw_b', 'raw_c', 'beta_re', 'beta_im']):
                continue
            if not p.is_complex():
                if "bias" in name:
                    p.data.zero_()
                continue
            if "bias" in name:
                p.data.zero_()
            else:
                p.data.real.normal_(0, sigma_uniform)
                p.data.imag.normal_(0, sigma_uniform)

    def forward(
        self,
        x: torch.Tensor,
        phases: torch.Tensor,
        gammas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, V, T, SPE, PE) complex64 ; phases: (B, T) -> delta_x same shape."""
        if x.dim() != 5 or not torch.is_complex(x):
            raise ValueError(f"x must be 5-D complex; got dim={x.dim()}, dtype={x.dtype}")
        B, V, T, W, H = x.shape
        if V != self.in_channels:
            raise ValueError(f"x channel dim V={V} != in_channels={self.in_channels}")
        if phases.shape != (B, T):
            raise ValueError(f"phases must be (B={B}, T={T}); got {tuple(phases.shape)}")

        # pad spatial dims to a multiple of patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        if pad_w > 0 or pad_h > 0:
            x_padded = torch.zeros(B, V, T, W + pad_w, H + pad_h, dtype=x.dtype, device=x.device)
            x_padded[:, :, :, :W, :H] = x
            x_fwd = x_padded
        else:
            x_fwd = x

        # patch embed (per time step)
        W_p, H_p = x_fwd.shape[3], x_fwd.shape[4]
        x_e = x_fwd.permute(0, 2, 1, 3, 4).contiguous().reshape(B * T, V, W_p, H_p)
        z = self.patch_embed(x_e)
        d, Hp, Wp = z.shape[1], z.shape[2], z.shape[3]
        z = z.reshape(B, T, d, Hp, Wp)

        # positional encodings
        pe = self.temporal_pe(phases)
        z = z + pe[:, :, :, None, None]
        pe_h, pe_w = self.spatial_pe(Hp=Hp, Wp=Wp, device=z.device)
        z = z + pe_h.permute(1, 0)[None, None, :, :, None]
        z = z + pe_w.permute(1, 0)[None, None, :, None, :]

        for b, block in enumerate(self.blocks):
            z = block(z, None if gammas is None else gammas[b])

        # patch unembed (per time step)
        z = z.reshape(B * T, d, Hp, Wp)
        delta = self.patch_unembed(z)
        delta = delta.reshape(B, T, V, W_p, H_p).permute(0, 2, 1, 3, 4).contiguous()

        if pad_w > 0 or pad_h > 0:
            delta = delta[:, :, :, :W, :H]
        return delta
