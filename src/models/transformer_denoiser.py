"""CMRx Transformer Denoiser (non-FiLM, normalized-index spatial PE).

Complex-valued, factorised transformer denoiser for one cascade stage. The
spatial positional encoding is the normalized-index Fourier PE
(:class:`src.models.fourier_pe.FourierSpatialPE`): the coordinate is ``i/N``
per axis, which self-normalises across the varying per-subject spatial dims —
no physical voxel resolution is required. LayerNorm is the learnable
:class:`src.models.complex_norm.ComplexLayerNorm2x2`; there is no FiLM
conditioning anywhere in the network.

Architecture
============

  Pad spatial dims -> patch_embed (ComplexConv2d, einsum-only) ->
  cardiac-phase Fourier PE (additive) ->
  normalized-index spatial Fourier PE (additive) ->
  ``n_blocks`` x factorised complex transformer blocks
       (temporal -> row -> col axial CMHA -> CMLP, each with CLN2x2) ->
  patch_unembed (NN-upsample + ComplexConv2d 3x3) -> crop to original spatial.

Returns: unweighted complex residual ``delta_x`` of the same shape as input.

Constraints
===========

  - ROCm gfx906: no ``F.conv2d``, no MIOpen — uses :mod:`src.models.complex_ops`.
  - Re-uses :class:`src.models.attention.MultiHeadCAtt` unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as _ckpt

from src.models.attention import MultiHeadCAtt
from src.models.complex_norm import ComplexLayerNorm2x2
from src.models.complex_ops import (
    ComplexConv2d,
    ComplexLinear,
    ComplexUpsampleConv2d,
    modReLU,
)
from src.models.fourier_pe import FourierCardiacPhasePE, FourierSpatialPE


# ─────────────────────────────────────────────────────────────────────────────
# Factorised block with learnable ComplexLayerNorm2x2
# ─────────────────────────────────────────────────────────────────────────────


class _MLPSublayer(nn.Module):
    """One complex MLP sublayer: CLN2x2 -> fc1 -> modReLU -> fc2, residual-added.

    Used only when the CMLP is unshared across cascade stages (one per stage). The
    shared/default path keeps the flat ``norm_mlp``/``mlp_fc1``/``mlp_act``/``mlp_fc2``
    attributes on the block so existing checkpoints load unchanged.
    """

    def __init__(self, d_model: int, mlp_ratio: int):
        super().__init__()
        self.norm = ComplexLayerNorm2x2(d_model)
        d_mlp = d_model * mlp_ratio
        self.fc1 = ComplexLinear(d_model, d_mlp)
        self.act = modReLU(d_mlp)
        self.fc2 = ComplexLinear(d_mlp, d_model)

    def forward(self, z_flat: torch.Tensor, gamma: torch.Tensor | None) -> torch.Tensor:
        h = self.norm(z_flat, gamma)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return z_flat + h


class _AxialAttnSet(nn.Module):
    """Container for one stage's axial-attention path (3 norms + 3 attentions).
    Used only when the attention is unshared across cascade stages."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm_t = ComplexLayerNorm2x2(d_model)
        self.attn_t = MultiHeadCAtt(d_model, n_heads)
        self.norm_r = ComplexLayerNorm2x2(d_model)
        self.attn_r = MultiHeadCAtt(d_model, n_heads)
        self.norm_c = ComplexLayerNorm2x2(d_model)
        self.attn_c = MultiHeadCAtt(d_model, n_heads)


class _FactorizedBlock(nn.Module):
    """One factorised complex transformer block (non-FiLM).

    Attention path (t/r/c + their CLN2x2): shared across stages by default (flat
    ``norm_*``/``attn_*`` attributes, unchanged checkpoint keys) or per-stage
    (``attn_stages`` ModuleList) when ``unshared_attn=True``. The MLP sublayer is
    likewise shared by default or per-stage (``mlp_stages``) when ``unshared_mlp=True``.
    """

    def __init__(self, d_model: int = 32, n_heads: int = 4, mlp_ratio: int = 2,
                 attn_mode: str = "series", unshared_mlp: bool = False,
                 n_stages: int = 1, unshared_attn: bool = False):
        super().__init__()
        if attn_mode not in ("series", "parallel"):
            raise ValueError(f"attn_mode must be 'series' or 'parallel'; got {attn_mode!r}")
        self.attn_mode = attn_mode          # 'series' = axial t->r->c; 'parallel' = t,r,c from same input, summed
        self.unshared_attn = bool(unshared_attn)
        if self.unshared_attn:
            if int(n_stages) < 1:
                raise ValueError(f"n_stages must be >= 1 for unshared attention; got {n_stages}")
            self.attn_stages = nn.ModuleList(
                [_AxialAttnSet(d_model, n_heads) for _ in range(int(n_stages))]
            )
        else:
            # Temporal attention
            self.norm_t = ComplexLayerNorm2x2(d_model)
            self.attn_t = MultiHeadCAtt(d_model, n_heads)

            # Row attention
            self.norm_r = ComplexLayerNorm2x2(d_model)
            self.attn_r = MultiHeadCAtt(d_model, n_heads)

            # Col attention
            self.norm_c = ComplexLayerNorm2x2(d_model)
            self.attn_c = MultiHeadCAtt(d_model, n_heads)

        # MLP — shared (default; unchanged keys) or per-stage unshared.
        self.unshared_mlp = bool(unshared_mlp)
        if self.unshared_mlp:
            if int(n_stages) < 1:
                raise ValueError(f"n_stages must be >= 1 for unshared MLP; got {n_stages}")
            self.mlp_stages = nn.ModuleList(
                [_MLPSublayer(d_model, mlp_ratio) for _ in range(int(n_stages))]
            )
        else:
            self.norm_mlp = ComplexLayerNorm2x2(d_model)
            d_mlp = d_model * mlp_ratio
            self.mlp_fc1 = ComplexLinear(d_model, d_mlp)
            self.mlp_act = modReLU(d_mlp)
            self.mlp_fc2 = ComplexLinear(d_mlp, d_model)

    def forward(self, z: torch.Tensor, gamma: torch.Tensor | None = None,
                stage_idx: int = 0) -> torch.Tensor:
        """z: (B, T, Hp, Wp, d) complex64 CHANNEL-LAST -> same shape.

        The block works in channel-last because every sublayer consumes tokens as
        (..., d): the row attention and the MLP then need NO permute at all (plain
        reshapes of a contiguous tensor), and only the t/c axial passes move axes.
        The previous channel-first (B,T,d,Hp,Wp) contract forced a
        permute+contiguous round-trip through the channel-first layout between
        EVERY sublayer — 8 full-tensor copies per block instead of 4, i.e. tens of
        GB of pure copy traffic per window at d=160. The denoiser converts once on
        each side of the block stack instead (see CMRxTransformerDenoiser.forward).

        gamma: optional (d,) real conditional scale applied at ALL of this block's
        norms (shared per block). None -> unconditioned (baseline).
        stage_idx: cascade-stage index; selects the per-stage MLP/attention when unshared."""
        B, T, Hp, Wp, d = z.shape
        a = self.attn_stages[stage_idx] if self.unshared_attn else self

        if self.attn_mode == "parallel":
            # --- Parallel axial: t, r, c all from the SAME z, summed once ---
            z_t = z.permute(0, 2, 3, 1, 4).contiguous().reshape(B * Hp * Wp, T, d)
            dt = a.attn_t(a.norm_t(z_t, gamma)).reshape(B, Hp, Wp, T, d).permute(0, 3, 1, 2, 4)
            z_r = z.reshape(B * T * Hp, Wp, d)                       # already contiguous
            dr = a.attn_r(a.norm_r(z_r, gamma)).reshape(B, T, Hp, Wp, d)
            z_c = z.permute(0, 1, 3, 2, 4).contiguous().reshape(B * T * Wp, Hp, d)
            dc = a.attn_c(a.norm_c(z_c, gamma)).reshape(B, T, Wp, Hp, d).permute(0, 1, 3, 2, 4)
            z = z + dt + dr + dc
        else:
            # --- Series axial: t -> r -> c sequential (default; unchanged math) ---
            z_t = z.permute(0, 2, 3, 1, 4).contiguous().reshape(B * Hp * Wp, T, d)
            z_t = z_t + a.attn_t(a.norm_t(z_t, gamma))
            z = z_t.reshape(B, Hp, Wp, T, d).permute(0, 3, 1, 2, 4).contiguous()

            z_r = z.reshape(B * T * Hp, Wp, d)                       # no permute needed
            z_r = z_r + a.attn_r(a.norm_r(z_r, gamma))
            z = z_r.reshape(B, T, Hp, Wp, d)

            z_c = z.permute(0, 1, 3, 2, 4).contiguous().reshape(B * T * Wp, Hp, d)
            z_c = z_c + a.attn_c(a.norm_c(z_c, gamma))
            z = z_c.reshape(B, T, Wp, Hp, d).permute(0, 1, 3, 2, 4).contiguous()

        # --- MLP (shared, or per-stage unshared) — channel-last: plain reshape ---
        z_m = z.reshape(B * T * Hp * Wp, d)
        if self.unshared_mlp:
            z_m = self.mlp_stages[stage_idx](z_m, gamma)
        else:
            z_mlp = self.norm_mlp(z_m, gamma)
            z_mlp = self.mlp_fc1(z_mlp)
            z_mlp = self.mlp_act(z_mlp)
            z_mlp = self.mlp_fc2(z_mlp)
            z_m = z_m + z_mlp
        return z_m.reshape(B, T, Hp, Wp, d)


# ─────────────────────────────────────────────────────────────────────────────
# Denoiser
# ─────────────────────────────────────────────────────────────────────────────


class CMRxTransformerDenoiser(nn.Module):
    """CMRx non-FiLM complex transformer denoiser (normalized-index spatial PE).

    Input / output
    --------------
    Input
        x         : (B, V, T, SPE, PE) complex64
        phases    : (B, T) real float — normalised cardiac phase in [0, 1)

    Output
        delta_x   : (B, V, T, SPE, PE) complex64 — unweighted residual.
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
        n_stages: int = 1,
        unshared_mlp: bool = False,
        unshared_attn: bool = False,
        mlp_neg_bias_init=None,
        embed_kernel: int = 1,
        block_ckpt: bool = False,
        **_,
    ):
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1; got {n_blocks}")
        self.in_channels = int(in_channels)
        # block_ckpt: gradient-checkpoint each transformer BLOCK (training only) —
        # finer than the cascade's per-stage grad_check and nests under it, so the
        # backward peak is ONE block's graph instead of the whole stage's. For
        # matrices where even grad_check OOMs (S2 Carotid/Cerebro). ~+30% recompute.
        self.block_ckpt = bool(block_ckpt)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_blocks = int(n_blocks)
        self.patch_size = int(patch_size)
        self.fourier_K = int(fourier_K)
        if init_scheme != "uniform":
            raise ValueError(
                "init_scheme must be 'uniform' (the only supported scheme; "
                f"'asymmetric' was removed). Got {init_scheme!r}."
            )
        self.init_scheme = init_scheme
        self.pe_learnable = bool(pe_learnable)
        # per-block attention wiring: None -> all 'series' (unchanged default)
        if attn_modes is None:
            attn_modes = ["series"] * self.n_blocks
        if len(attn_modes) != self.n_blocks:
            raise ValueError(f"attn_modes must have n_blocks={self.n_blocks} entries; got {attn_modes}")
        self.attn_modes = list(attn_modes)

        # Patch embed. Default (embed_kernel=1) = the original patchify ComplexConv2d
        # (kernel==stride==patch_size), keys unchanged. embed_kernel>1 uses a same-
        # resolution overlapping k x k complex conv (ComplexUpsampleConv2d at scale=1)
        # so each per-pixel token aggregates its k x k neighbourhood.
        self.embed_kernel = int(embed_kernel)
        _embed_in = self.in_channels
        if self.embed_kernel == 1:
            self.patch_embed = ComplexConv2d(
                _embed_in, d_model, kernel_size=patch_size, stride=patch_size,
            )
        else:
            if patch_size != 1:
                raise ValueError(
                    "embed_kernel > 1 requires patch_size == 1 (same-resolution "
                    f"overlapping conv); got embed_kernel={self.embed_kernel}, patch_size={patch_size}"
                )
            self.patch_embed = ComplexUpsampleConv2d(
                _embed_in, d_model, scale_factor=patch_size, kernel_size=self.embed_kernel,
            )

        # Cardiac-phase Fourier PE (normalized frame/Nt; optionally learnable freqs)
        self.temporal_pe = FourierCardiacPhasePE(
            d_model, K=fourier_K, learnable=self.pe_learnable,
        )

        # Spatial Fourier PE — normalized i/N coordinate (self-normalising across
        # the varying per-subject spatial dims), optionally learnable freqs.
        self.spatial_pe = FourierSpatialPE(
            d_model, K=fourier_K, learnable=self.pe_learnable,
        )

        # Factorised transformer blocks. Attention path shared across stages; the
        # MLP sublayer is per-stage when unshared_mlp (needs n_stages copies).
        self.unshared_mlp = bool(unshared_mlp)
        self.unshared_attn = bool(unshared_attn)
        self.blocks = nn.ModuleList([
            _FactorizedBlock(d_model, n_heads, mlp_ratio, attn_mode=self.attn_modes[i],
                             unshared_mlp=self.unshared_mlp, n_stages=int(n_stages),
                             unshared_attn=self.unshared_attn)
            for i in range(n_blocks)
        ])

        # Patch unembed (NN-upsample + einsum 3x3 conv, gfx906-safe)
        self.patch_unembed = ComplexUpsampleConv2d(
            d_model, in_channels, scale_factor=patch_size, kernel_size=3,
        )

        self._init_weights()

        # Optional negative modReLU-bias init: makes the magnitude-shrinkage
        # nonlinearity active from step 0 (bias=0 => identity => linear MLP).
        # Applied AFTER _init_weights (which zeros every real bias). None => zeros
        # (unchanged default), so existing checkpoints are unaffected.
        if mlp_neg_bias_init is not None:
            lo, hi = float(mlp_neg_bias_init[0]), float(mlp_neg_bias_init[1])
            for m in self.modules():
                if isinstance(m, modReLU):
                    nn.init.uniform_(m.bias, lo, hi)

    # ───────────────────────────────────────────────── _init_weights ────────
    def _init_weights(self) -> None:
        """Uniform (BCRNN-style) init: every complex weight ~ N(0, σ=0.012),
        all biases zero.

        Note: ``ComplexLayerNorm2x2`` has its own properly initialised
        parameters (``raw_a``, ``raw_b``, ``raw_c``, ``beta_re``, ``beta_im``)
        — we skip those here to avoid clobbering their init.
        """
        sigma_uniform = 0.012
        for name, p in self.named_parameters():
            # Skip CLN parameters (already properly initialized in CLN.__init__)
            if any(x in name for x in ['raw_a', 'raw_b', 'raw_c',
                                        'beta_re', 'beta_im']):
                continue

            # Real parameters (modReLU bias) -> zero
            if not p.is_complex():
                if "bias" in name:
                    p.data.zero_()
                continue

            # Complex parameters: biases zero, weights ~ N(0, σ=0.012)
            if "bias" in name:
                p.data.zero_()
            else:
                p.data.real.normal_(0, sigma_uniform)
                p.data.imag.normal_(0, sigma_uniform)

    # ───────────────────────────────────────────────────── forward ─────────
    def forward(
        self,
        x: torch.Tensor,
        phases: torch.Tensor,
        gammas: torch.Tensor | None = None,
        stage_idx: int = 0,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, V, T, SPE, PE) complex64
            Input complex feature map for one cascade stage.
        phases : (B, T) float
            Normalised cardiac phase per cardiac bin, in [0, 1).
        gammas : (n_blocks, d_model) real, optional
            Per-block conditional norm scale; ``gammas[b]`` is shared across block
            b's norms. None -> unconditioned (baseline).

        Returns
        -------
        delta_x : (B, V, T, SPE, PE) complex64 — unweighted residual.
        """
        # ── Validation ─────────────────────────────────────────────────────
        if x.dim() != 5 or not torch.is_complex(x):
            raise ValueError(
                f"x must be 5-D complex; got dim={x.dim()}, dtype={x.dtype}"
            )
        B, V, T, W, H = x.shape
        if V != self.in_channels:
            raise ValueError(
                f"x channel dim V={V} != in_channels={self.in_channels}"
            )
        if phases.shape != (B, T):
            raise ValueError(
                f"phases must be (B={B}, T={T}); got {tuple(phases.shape)}"
            )
        # ── Pad spatial dims to be divisible by patch_size ─────────────────
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        if pad_w > 0 or pad_h > 0:
            x_padded = torch.zeros(
                B, V, T, W + pad_w, H + pad_h,
                dtype=x.dtype, device=x.device,
            )
            x_padded[:, :, :, :W, :H] = x
            x_fwd = x_padded
        else:
            x_fwd = x

        # ── Patch Embed (per time step) ────────────────────────────────────
        W_p, H_p = x_fwd.shape[3], x_fwd.shape[4]
        x_e = x_fwd.permute(0, 2, 1, 3, 4).contiguous()      # (B, T, V, W_p, H_p)
        x_e = x_e.reshape(B * T, V, W_p, H_p)
        z = self.patch_embed(x_e)                            # (B*T, d, Hp, Wp)
        d, Hp, Wp = z.shape[1], z.shape[2], z.shape[3]
        # ONE conversion to the blocks' channel-last working layout (and one back
        # after the stack) instead of a round-trip inside every sublayer.
        z = z.reshape(B, T, d, Hp, Wp).permute(0, 1, 3, 4, 2).contiguous()   # (B,T,Hp,Wp,d)

        # ── Cardiac-phase Fourier PE (unchanged from Arch-A) ───────────────
        pe = self.temporal_pe(phases)                        # (B, T, d) complex
        z = z + pe[:, :, None, None, :]                      # broadcast Hp, Wp

        # ── Spatial Fourier PE (normalized i/N coordinate) ─────────────────
        pe_h, pe_w = self.spatial_pe(Hp=Hp, Wp=Wp, device=z.device)
        z = z + pe_h[None, None, :, None, :]                 # (1, 1, Hp, 1, d)
        z = z + pe_w[None, None, None, :, :]                 # (1, 1, 1, Wp, d)

        # ── Transformer Blocks ────────────────────────────────────────────
        for b, block in enumerate(self.blocks):
            gamma_b = None if gammas is None else gammas[b]
            if self.block_ckpt and self.training:
                # NOTE: block/gamma/stage bound by value (default args) — see the
                # cascade's grad_check note; nesting under it is supported with
                # use_reentrant=False.
                z = _ckpt.checkpoint(
                    lambda zz, _blk=block, _gm=gamma_b, _s=stage_idx:
                        _blk(zz, _gm, stage_idx=_s),
                    z, use_reentrant=False,
                )
            else:
                z = block(z, gamma_b, stage_idx=stage_idx)

        # ── Patch Unembed (per time step) ──────────────────────────────────
        z = z.permute(0, 1, 4, 2, 3).contiguous().reshape(B * T, d, Hp, Wp)   # back to channel-first
        delta = self.patch_unembed(z)                        # (B*T, in_channels, W_p, H_p)
        delta = delta.reshape(B, T, self.in_channels, W_p, H_p)
        delta = delta.permute(0, 2, 1, 3, 4).contiguous()    # (B, V, T, W_p, H_p)

        # ── Crop back to original spatial size ────────────────────────────
        if pad_w > 0 or pad_h > 0:
            delta = delta[:, :, :, :W, :H]

        return delta
