"""Evaluation with MSAC background-phase correction.

The background (eddy-current) phase is fit on the ground truth and the same correction is applied
to the ground truth and every reconstruction before decoding magnitude / flow, matching the
challenge's evaluation.
"""
from __future__ import annotations

import numpy as np
import torch

from src.utils.cmrx_metrics import SSIM, SSIM3D, _to_tensor, nRMSE, RelErr, AngErr, complex2magflow
from src.utils.utils_bgc import execute_MSAC


def _ssim_gated(mag_p, mag_g, segmask, device):
    """SSIM (same math as cmrx_metrics.SSIM) computed on ``device``, falling back to CPU on error."""
    if segmask is None:
        segmask = np.ones(mag_g.shape[-3:], dtype=bool)
    fn = SSIM3D(window_size=11, size_average=False)
    p = mag_p * segmask[None, None]
    g = mag_g * segmask[None, None]
    gmax = np.max(g)
    pt = _to_tensor(p / gmax).float()
    gt = _to_tensor(g / gmax).float()
    try:
        smap = fn(pt.to(device), gt.to(device))
    except RuntimeError:
        smap = fn(pt, gt)                      # CPU fallback
    roi = _to_tensor(segmask.astype(bool)).to(smap.device).unsqueeze(0).unsqueeze(0)
    roi_cnt = roi.sum().clamp_min(1.0) * g.shape[1] * g.shape[0]
    return float(((smap * roi).sum() / roi_cnt).item())


def msac_correction(gt: np.ndarray, corr_fit_order: int = 3, th: float = 0.1) -> np.ndarray:
    """Background-phase correction map fit on GT. Shape (Nv-1, 1, SPE, PE, FE)."""
    return execute_MSAC(np.asarray(gt), corr_fit_order=corr_fit_order, th=th)


def apply_correction(img: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Apply an MSAC correction to the velocity channels (indices 1..) of a copy."""
    out = np.asarray(img).copy()
    out[1:] = out[1:] * np.exp(-1j * corr)          # corr is (Nv-1,1,...) -> broadcasts over Nt
    return out


def _ssim_roi_crop(mag_p, mag_g, segmask, margin: int = 5):
    """SSIM gated by segmask, computed only on the segmask bounding box plus a margin.

    Each SSIM voxel depends only on its 11^3 neighbourhood, so cropping to the ROI bbox padded by
    the window radius leaves every in-mask voxel's value unchanged while being much faster.
    """
    seg = np.asarray(segmask)
    if seg is None or not seg.any():
        return float(SSIM(mag_p, mag_g, segmask))
    zz, yy, xx = np.where(seg)
    sl = tuple(slice(max(0, a.min() - margin), min(d, a.max() + 1 + margin))
               for a, d in zip((zz, yy, xx), seg.shape))
    return float(SSIM(mag_p[:, :, sl[0], sl[1], sl[2]],
                      mag_g[:, :, sl[0], sl[1], sl[2]], seg[sl[0], sl[1], sl[2]]))


def official_metrics(pred, gt, segmask, venc, corr_fit_order: int = 3, th: float = 0.1,
                     corr: np.ndarray | None = None, ssim_device=None,
                     compute_ssim: bool = True) -> dict:
    """MSAC-corrected, segmask-gated SSIM / nRMSE / RelErr / AngErr.

    pred, gt : (Nv, Nt, SPE, PE, FE) complex ; segmask : (SPE, PE, FE) bool ; venc : (Nv-1,) cm/s.
    ``corr``: optional precomputed MSAC map (reuse one GT fit across recons).
    ``ssim_device``: run the 3-D SSIM on this device; None uses CPU.
    """
    gt = np.asarray(gt); pred = np.asarray(pred)
    if corr is None:
        corr = msac_correction(gt, corr_fit_order=corr_fit_order, th=th)
    gt_c = apply_correction(gt, corr)
    pred_c = apply_correction(pred, corr)
    mag_g, flow_g = complex2magflow(gt_c, venc)
    mag_p, flow_p = complex2magflow(pred_c, venc)
    if not compute_ssim:                       # SSIM (3D conv) is the slow metric; skip for speed
        ssim = float("nan")
    elif ssim_device is not None:
        ssim = _ssim_gated(mag_p, mag_g, segmask, ssim_device)
    else:
        ssim = _ssim_roi_crop(mag_p, mag_g, segmask)
    return {
        "SSIM":   ssim,
        "nRMSE":  float(nRMSE(mag_p, mag_g, segmask)),
        "RelErr": float(RelErr(flow_p, flow_g, segmask)),
        "AngErr": float(AngErr(flow_p, flow_g, segmask)),
    }
