"""CMRx unrolled cascade (non-FiLM, normalized-index spatial PE).

Wires the CMRx transformer denoiser together with DC + WA into a single
end-to-end unrolled recon network. There is no FiLM conditioning:
``ComplexLayerNorm2x2`` (learnable, no external conditioning) is used
throughout, and the spatial PE is the normalized-index
:class:`src.models.fourier_pe.FourierSpatialPE` (coordinate ``i/N`` per
axis — self-normalising across the varying per-subject spatial dims, so no
physical voxel resolution is needed).

Architecture per cascade stage (weight-shared denoiser, weights unrolled
``n_stages`` times):

    delta   = denoiser(x_curr, phases)
    x_pre   = x_curr + delta
    Sx      = DC_i(x_pre, y_theta, sens)        (k-space consistency)
    x_next  = WA_i(x_pre, Sx)                   (image-space blend)

Inputs / outputs
================

forward(x_init, y_theta, sens, mask_theta, phases) -> x_recon

  x_init   : (B, V, T, SPE, PE)            complex64 — zero-filled image
  y_theta  : (B, V, C, T, SPE, PE)         complex64 — Theta-masked k-space
  sens     : (B, C, SPE, PE)               complex64 — coil sensitivities
  mask_theta : currently unused (k0 already carries the mask via zeros)
               kept in the signature for future use / clarity.
  phases   : (B, T) float — normalised cardiac phase in [0, 1).

Returns
  x_recon  : (B, V, T, SPE, PE) complex64 — refined image after ``n_stages``
             unrolled iterations.

Constraints (ROCm gfx906)
=========================

  - No complex conv via cuDNN; the denoiser uses einsum-only
    :class:`src.models.complex_ops.ComplexConv2d`.
  - DC / WA reuse Arch-A's verified blocks unchanged.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as _ckpt

from src.models.transformer_denoiser import CMRxTransformerDenoiser
from src.models.dc_wa import dataConsistencyTerm, weightedAverageTerm
from src.models.conditioning import ConditioningMLP, ConditioningTable


# ─────────────────────────────────────────────────────────────────────────────
# CMRxTransformerCascade
# ─────────────────────────────────────────────────────────────────────────────


class CMRxTransformerCascade(nn.Module):
    """Unrolled cascade with non-FiLM transformer denoiser + DC + WA.

    Parameters
    ----------
    n_stages : int
        Number of unrolling iterations. The denoiser is *shared* across
        stages (weight-tied). One independent DC and WA per stage.
    d_model, n_heads, n_blocks, mlp_ratio, patch_size : int
        Forwarded to :class:`CMRxTransformerDenoiser`.
    in_channels : int
        Number of complex channels V in the input image. Default 6 for CMRx.
    K_pe : int
        Number of Fourier harmonics for the spatial PE (= ``fourier_K`` in
        the denoiser).
    K_cardiac : int
        Number of Fourier harmonics for the cardiac-phase PE. Currently
        equals K_pe internally because :class:`CMRxTransformerDenoiser`
        uses a single ``fourier_K`` for both. Kept as a separate
        argument for forward-compatibility; raises ``ValueError`` if it
        differs from ``K_pe`` to avoid silent misconfiguration.
    dc_min_v, wa_min_para : float
        Optional lower-bound floors on the DC / WA sigmoid weights.
    init_scheme : str
        Forwarded to the denoiser.
    pe_learnable : bool
        Whether the Fourier PE frequencies are learnable. Forwarded to the
        denoiser.
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
        unshared_mlp: bool = False,
        unshared_attn: bool = False,
        mlp_neg_bias_init=None,
        embed_kernel: int = 1,
        denoiser: str = "transformer",
        frame_chunk: int = 0,
        bcrnn_nf: int = 24,
        bcrnn_per_encoding: bool = True,
        block_ckpt: bool = False,
        conditioning: Optional[dict] = None,
        **_,
    ) -> None:
        super().__init__()
        if n_stages < 1:
            raise ValueError(f"n_stages must be >= 1; got {n_stages}")
        if K_pe != K_cardiac:
            raise ValueError(
                "CMRxTransformerCascade currently requires K_pe == K_cardiac "
                "because the denoiser uses a single fourier_K for both spatial "
                f"and cardiac PEs. Got K_pe={K_pe}, K_cardiac={K_cardiac}."
            )

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
        self.grad_check = bool(grad_check)   # gradient checkpointing: recompute the denoiser in backward

        # ── Denoiser (shared, weight-tied across stages) ────────────────
        # 'transformer' (default) = CMRxTransformerDenoiser (unchanged).
        # 'bcrnn' = FlowMRI-Net's BCRNN regularizer (paper reviewer comparison):
        # stateful across stages (ih2ih recurrence), so grad_check is refused.
        self.denoiser_type = str(denoiser).lower()
        if self.denoiser_type == "bcrnn":
            if self.grad_check:
                raise ValueError(
                    "denoiser='bcrnn' is stateful across stages (ih2ih recurrence); "
                    "grad_check recomputation would advance the state twice. Set grad_check: false."
                )
            from src.models.bcrnn_denoiser import BCRNNDenoiser
            self.denoiser = BCRNNDenoiser(
                in_channels=self.in_channels, nf=int(bcrnn_nf),
                per_encoding=bool(bcrnn_per_encoding),
            )
        elif self.denoiser_type == "transformer":
            self.denoiser = CMRxTransformerDenoiser(
                in_channels=self.in_channels,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_blocks=self.n_blocks,
                mlp_ratio=int(mlp_ratio),
                patch_size=self.patch_size,
                fourier_K=self.K_pe,
                init_scheme=self.init_scheme,
                pe_learnable=self.pe_learnable,
                attn_modes=attn_modes,
                n_stages=self.n_stages,          # for per-stage (unshared) CMLP/attention copies
                unshared_mlp=unshared_mlp,
                unshared_attn=unshared_attn,
                mlp_neg_bias_init=mlp_neg_bias_init,
                embed_kernel=embed_kernel,
                block_ckpt=bool(block_ckpt),
            )
        else:
            raise ValueError(f"denoiser must be 'transformer' or 'bcrnn'; got {denoiser!r}")

        # ── DC + WA: one independent (learnable) block per stage ────────
        self.dc = nn.ModuleList([
            dataConsistencyTerm(-2.2, min_v=dc_min_v) for _ in range(self.n_stages)
        ])
        self.wa = nn.ModuleList([
            weightedAverageTerm(-2.2, min_para=wa_min_para)
            for _ in range(self.n_stages)
        ])

        # ── Conditioning (Ada-MoDL-style, opt-in; zero-init => baseline at step 0) ──
        cond = conditioning or {}
        self.conditioning_enabled = bool(cond.get('enabled', False))
        # mode 'table' (v2): anchored lookup table on the WA gate ONLY.
        #   - DC is NOT conditioned: it saturates to sigma(-16) ~ 8e-8, where a
        #     0.7-logit offset changes the output by ~1e-9, i.e. 80x BELOW float32
        #     epsilon. Measured on p1: max|v(R=10)-v(R=50)| == 0.000000.
        #   - FiLM is NOT used: the denoiser is R-invariant (spectral distance
        #     between R8/R16-trained prototypes was ~4x SMALLER than between two
        #     independent runs), so conditioning it adds capacity for nothing.
        self.cond_mode = str(cond.get('mode', 'mlp')).lower()
        self.conditioning_inputs = (
            list(cond.get('inputs', ['R'])) if self.conditioning_enabled else []
        )
        self.use_film = (bool(cond.get('use_film', False))
                         and self.conditioning_enabled and self.cond_mode != 'table')
        self.cond_mlp = None
        self.cond_table = None
        if self.conditioning_enabled and self.cond_mode == 'table':
            self.cond_table = ConditioningTable(
                r_values=cond.get('r_values', [10, 20, 30, 40, 50]),
                n_stages=self.n_stages,
                anchor_R=cond.get('anchor_R'),
            )
        elif self.conditioning_enabled:
            self.cond_mlp = ConditioningMLP(
                input_dim=len(self.conditioning_inputs),
                n_stages=self.n_stages, n_blocks=self.n_blocks,
                d_model=self.d_model, use_film=self.use_film,
            )

    # ──────────────────────────────────────────────────────────── forward
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
        """Run the unrolled cascade.

        See class docstring for parameter shapes. ``mask_theta`` is
        currently unused — k0 already encodes the sampling mask via its
        zero locations. Kept in the signature for API forward compatibility.
        """
        # ── Validate ─────────────────────────────────────────────────────
        if x_init.dim() != 5 or not torch.is_complex(x_init):
            raise ValueError(
                f"x_init must be 5-D complex (B, V, T, SPE, PE); got "
                f"dim={x_init.dim()}, dtype={x_init.dtype}."
            )
        B, V, T, SPE, PE = x_init.shape
        if V != self.in_channels:
            raise ValueError(
                f"x_init channel dim V={V} != in_channels={self.in_channels}."
            )
        if phases.shape != (B, T):
            raise ValueError(
                f"phases must be (B={B}, T={T}); got {tuple(phases.shape)}."
            )
        # ── Conditioning offsets (computed ONCE per forward) ──────────────
        delta_v = delta_para = gammas = None
        if self.cond_table is not None and R is not None:
            # v2: anchored table -> WA gate ONLY. delta_v stays None (the DC gate
            # is saturated, so conditioning it is a no-op) and gammas stays None
            # (the denoiser is R-invariant).
            delta_para = self.cond_table(R)        # (n_stages,)
        elif self.conditioning_enabled and self.cond_mlp is not None and m is not None:
            off = self.cond_mlp(m)
            delta_v = off['delta_v'][0]            # (n_stages,)
            delta_para = off['delta_para'][0]      # (n_stages,)
            if self.use_film and off['gamma'] is not None:
                gammas = off['gamma'][0]           # (n_blocks, d_model)

        # ── Unrolled loop ───────────────────────────────────────────────
        x_curr = x_init
        stages = [] if return_stages else None
        for i in range(self.n_stages):
            # Denoiser: returns the residual delta_x. Optionally gradient-checkpointed
            # (recompute in backward) so only one stage's activations live at a time —
            # the flowmri-net trick that lets heavy (patch_size=1) models fit / run two-up.
            if self.grad_check and self.training:
                # NOTE: stage_idx MUST be bound by value (default arg), not closed over.
                # use_reentrant=False re-invokes this lambda during BACKWARD, when the
                # loop variable i has its final value — a closure would recompute every
                # stage with the LAST stage's per-stage modules and corrupt their grads.
                delta = _ckpt.checkpoint(
                    lambda xc, ph, gm, _i=i: self.denoiser(xc, ph, gm, stage_idx=_i),
                    x_curr, phases, gammas, use_reentrant=False,
                )
            else:
                delta = self.denoiser(x_curr, phases, gammas, stage_idx=i)
            x_pre = x_curr + delta

            # Data consistency in k-space. The DC block expects:
            #   x  : (N, V, T, H, W)              -- our (B, V, T, SPE, PE)
            #   k0 : (N, C, V, T, D, H, W)        -- y_theta swap-axes + D=1
            #   c  : (N, C, D, H, W)              -- sens + D=1
            # dc-prev (fixed, FlowMRI-correct): DC operates on the PREVIOUS iterate
            # x_curr (=x^{n-1}), not the denoised x_pre. WA still blends x_pre with Sx.
            x_dc_in = x_curr
            k0 = y_theta
            c = sens
            # Add D=1 dim to k0 and c at the right place, then swap V<->C in k0.
            if k0.dim() == 6:
                # (B, V, C, T, SPE, PE) -> (B, V, C, T, 1, SPE, PE)
                k0 = k0.unsqueeze(4)
            if c.dim() == 4:
                # (B, C, SPE, PE) -> (B, C, 1, SPE, PE)
                c = c.unsqueeze(2)
            # DC expects (N, C, V, T, D, H, W).
            k0_swapped = torch.swapaxes(k0, 1, 2)

            Sx = self.dc[i].perform(
                x_dc_in, k0_swapped, c,
                delta=None if delta_v is None else delta_v[i],
            )
            # DC returns (N, V, T, D, H, W); drop the D=1 dim.
            if Sx.dim() == 6:
                Sx = Sx[:, :, :, 0]

            # Weighted average.
            x_curr = self.wa[i].perform(
                x_pre, Sx, delta=None if delta_para is None else delta_para[i],
            )
            if return_stages:
                stages.append(x_curr)

        if return_stages:
            return torch.stack(stages, dim=0)   # (n_stages, B, V, T, SPE, PE) — deep supervision
        return x_curr
