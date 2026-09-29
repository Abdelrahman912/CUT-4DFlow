"""Ω-domain reconstruction utilities (reusable across notebooks / CLIs).

At TEST time a trained net is fed the FULL acquired Ω k-space (= Θ + Λ), not the
SSDU Θ partition. These helpers build the Ω inputs from a dataset sample and run a
trained cascade over the whole cardiac cycle in T=5 windows (the trained window
size), stitched back into the full cycle.

A dataset ``sample`` here is the ndarray dict returned by
``src.data.dataset.CMRx4DFlowDataset.__getitem__`` (with D_size=-1 for the whole
volume). Shapes: img_* (V,T,SPE,PE,FE) complex; kdata_* (V,T,C,SPE,PE,FE);
mask_* (1,T,1,SPE,PE,1); coilmap (C,SPE,PE,FE).
"""
from __future__ import annotations

import numpy as np
import torch

from src.data.dataset import _k2i_numpy
from src.models.cascade import CMRxTransformerCascade
from src.models.conditioning import build_m
from src.training.train_cmrx import _slice_to_tensors


def build_cascade(cfg: dict, device):
    """Build a CMRxTransformerCascade from a checkpoint's ``meta`` cfg dict."""
    return CMRxTransformerCascade(
        n_stages=cfg["n_stages"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_blocks=cfg["n_blocks"], mlp_ratio=int(cfg.get("mlp_ratio", 2)),
        patch_size=cfg["patch_size"], in_channels=cfg["in_channels"],
        K_pe=cfg["K_pe"], K_cardiac=cfg["K_pe"], init_scheme=cfg["init_scheme"],
        pe_learnable=cfg["pe_learnable"], grad_check=bool(cfg.get("grad_check", False)),
        attn_modes=cfg.get("attn_modes"),
        unshared_mlp=bool(cfg.get("unshared_mlp", False)),
        unshared_attn=bool(cfg.get("unshared_attn", False)),
        mlp_neg_bias_init=cfg.get("mlp_neg_bias_init"),
        embed_kernel=int(cfg.get("embed_kernel", 1)),
        denoiser=str(cfg.get("denoiser", "transformer")),
        frame_chunk=int(cfg.get("frame_chunk", 0)),
        bcrnn_nf=int(cfg.get("bcrnn_nf", 24)),
        bcrnn_per_encoding=bool(cfg.get("bcrnn_per_encoding", True)),
        block_ckpt=bool(cfg.get("block_ckpt", False)),   # training-only; inert in eval
        conditioning=cfg.get("conditioning"),
    ).to(device)

_T_FIELDS = ["img_zf", "img_gt", "kdata_theta", "kdata_lambda", "mask_theta", "mask_lambda"]
_BATCH_KEYS = ["img_zf", "img_gt", "kdata_theta", "kdata_lambda",
               "mask_theta", "mask_lambda", "coilmap"]


def omega_zero_filled(sample, supervised: bool = False) -> np.ndarray:
    """Zero-filled image from the acquired Ω k-space. SSDU: Ω=Θ+Λ; supervised: Ω=Θ. (V,T,SPE,PE,FE)."""
    k_omega = sample["kdata_theta"] if supervised else (sample["kdata_theta"] + sample["kdata_lambda"])
    return np.sum(_k2i_numpy(k_omega, ax=[-2, -3]) * np.conj(sample["coilmap"]),
                  axis=-4).astype(np.complex64)


def build_omega_sample(sample, supervised: bool = False) -> dict:
    """Copy of ``sample`` whose Θ slots carry the acquired Ω k-space + its zero-fill.

    SSDU: Ω = Θ+Λ (recombine the two acquired subsets). Supervised: Θ is already the full
    acquired Ω, so feed it as-is — Λ is the UN-acquired complement and is never added at test.

    """
    s = dict(sample)
    if supervised:
        s["img_zf"] = omega_zero_filled(sample, supervised=True)
    else:
        s["kdata_theta"] = sample["kdata_theta"] + sample["kdata_lambda"]
        s["img_zf"] = omega_zero_filled(sample)
        s["mask_theta"] = sample["mask_omega"]
    return s


def slice_frames(sample, frames) -> dict:
    """Slice all time-dependent fields to ``frames`` (T axis)."""
    fr = np.asarray(frames)
    s = dict(sample)
    for k in _T_FIELDS:
        s[k] = sample[k][:, fr]
    s["cardiac_bins"] = np.asarray(sample["cardiac_bins"])[fr]
    return s


def _volume_batch(sample, device, fe_list) -> dict:
    singles = [_slice_to_tensors(sample, device, fe_idx=int(fe)) for fe in fe_list]
    keys = _BATCH_KEYS
    vb = {k: torch.cat([s[k] for s in singles], dim=0) for k in keys}
    vb["cardiac_bins"] = singles[0]["cardiac_bins"]
    vb["R"] = singles[0]["R"]                 # constant across a volume's FE slices
    vb["B0"] = singles[0].get("B0")
    return vb


def _forward(model, vb) -> torch.Tensor:
    B = vb["img_zf"].shape[0]
    Nt = int(vb["kdata_theta"].shape[3])
    phases = (vb["cardiac_bins"].to(torch.float32) / max(Nt, 1)
              ).unsqueeze(0).expand(B, -1).contiguous()
    m = None                                   # conditioning: one m per volume (R,B0 fixed)
    if getattr(model, "conditioning_enabled", False):
        m = build_m(model.conditioning_inputs, vb["R"], vb.get("B0"), vb["img_zf"].device)
    # R= is REQUIRED for table-mode conditioning (v2); without it the table is
    # silently inert and the model reconstructs as if unconditioned.
    return model(vb["img_zf"], vb["kdata_theta"], vb["coilmap"], vb["mask_theta"], phases,
                 m=m, R=vb["R"])


@torch.no_grad()
def recon_full_volume(model, sample, device, chunk: int = 8) -> np.ndarray:
    """Reconstruct every FE slice for the sample's T frames -> (V,T,SPE,PE,FE)."""
    model.eval()
    V, T, SPE, PE, FE = sample["img_gt"].shape
    out = np.zeros((V, T, SPE, PE, FE), dtype=np.complex64)
    for s0 in range(0, FE, chunk):
        fe = list(range(s0, min(s0 + chunk, FE)))
        x = _forward(model, _volume_batch(sample, device, fe)).cpu().numpy()   # (B,V,T,SPE,PE)
        out[..., fe] = np.transpose(x, (1, 2, 3, 4, 0))
    return out


@torch.no_grad()
def recon_net_full_cycle(model, sample, device, chunk: int = 8, overlap: int = 0,
                         circular: bool = False, supervised: bool = False) -> np.ndarray:
    """Net recon over the whole cardiac cycle, Ω input, tiled in T=5 windows.

    overlap=0 (default, production path): non-overlapping 5-frame windows — the tail
    window slides back to stay a full 5 frames and its frames take the last window's
    recon. overlap>0: windows stride by (5-overlap) and any frame covered by more than
    one window is AVERAGED (complex mean), smoothing the temporal seam between windows.
    circular=True (with overlap>0): windows WRAP around the cardiac cycle (mod Nt), so the
    end-of-cycle -> start transition is bridged like every other seam — the cycle is
    periodic and the model was trained on wrap-around windows, so this is in-distribution.
    Returns (V,T,SPE,PE,FE) complex.
    """
    so = build_omega_sample(sample, supervised=supervised)
    V, Nt, SPE, PE, FE = sample["img_gt"].shape
    out = np.zeros((V, Nt, SPE, PE, FE), dtype=np.complex64)
    if overlap <= 0 or Nt <= 5:
        for s0 in range(0, Nt, 5):
            win = list(range(s0, min(s0 + 5, Nt)))
            if len(win) < 5 and Nt >= 5:
                win = list(range(Nt - 5, Nt))
            rv = recon_full_volume(model, slice_frames(so, win), device, chunk=chunk)
            for li, gf in enumerate(win):
                out[:, gf] = rv[:, li]
        return out
    # overlapping: stride < 5; frames seen by >1 window are complex-averaged
    stride = max(1, 5 - int(overlap))
    if circular:
        windows = [[(s0 + i) % Nt for i in range(5)] for s0 in range(0, Nt, stride)]
    else:
        windows = [list(range(s0, s0 + 5))
                   for s0 in sorted({min(s0, Nt - 5) for s0 in range(0, Nt, stride)})]
    cnt = np.zeros(Nt, dtype=np.float32)
    for win in windows:
        rv = recon_full_volume(model, slice_frames(so, win), device, chunk=chunk)
        for li, gf in enumerate(win):
            out[:, gf] += rv[:, li]; cnt[gf] += 1.0
    cnt[cnt == 0] = 1.0
    return out / cnt[None, :, None, None, None]
