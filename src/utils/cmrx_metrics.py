"""Evaluation metrics (SSIM, nRMSE, RelErr, AngErr), flow decoding, and sparse .npz I/O.

Metric formulas follow the CMRxRecon 4D-flow evaluation so local numbers match the challenge scorer.
"""

from __future__ import annotations

import os
from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def _gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window_3D(window_size, channel):
    _1D = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t())
    _3D = (_1D.mm(_2D.reshape(1, -1)).reshape(window_size, window_size, window_size)
           .float().unsqueeze(0).unsqueeze(0))
    return _3D.expand(channel, 1, window_size, window_size, window_size).contiguous()


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = _create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        Nv, Nt = img1.shape[0], img1.shape[1]
        img1 = rearrange(img1, 'v t s p f -> (v t) 1 s p f')
        img2 = rearrange(img2, 'v t s p f -> (v t) 1 s p f')
        channel = img1.size(1)
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = _create_window_3D(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        out = _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)
        return rearrange(out, '(v t) 1 s p f -> v t s p f', v=Nv, t=Nt)


def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def SSIM(pred, gt, segmask=None):
    """SSIM within segmask. pred, gt: (Nv, Nt, SPE, PE, FE) magnitudes."""
    if segmask is None:
        segmask = np.ones(gt.shape[-3:], dtype=bool)
    ssim_fn = SSIM3D(window_size=11, size_average=False)
    pred = pred * segmask[None, None]
    gt = gt * segmask[None, None]
    gt_max = np.max(gt)
    pred = pred / gt_max
    gt = gt / gt_max
    ssim_map = ssim_fn(_to_tensor(pred).float(), _to_tensor(gt).float())
    roi = _to_tensor(segmask.astype(bool)).to(ssim_map.device).unsqueeze(0).unsqueeze(0)
    roi_cnt = roi.sum().clamp_min(1.0) * gt.shape[1] * gt.shape[0]
    return ((ssim_map * roi).sum() / roi_cnt).item()


def nRMSE(pred, gt, segmask=None, eps=1e-12):
    """Normalized RMSE within segmask. pred, gt: (Nv, Nt, SPE, PE, FE) magnitudes."""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if segmask is None:
        segmask = np.ones(gt.shape[-3:], dtype=bool)
    segmask = np.asarray(segmask, dtype=bool)
    while segmask.ndim < gt.ndim:
        segmask = segmask[None, ...]
    mask = np.broadcast_to(segmask, gt.shape).astype(np.float32)
    n = np.sum(mask)
    mse = np.sum(((pred - gt) ** 2) * mask) / (n + eps)
    denom = np.max(gt * mask) + eps
    return np.sqrt(mse) / denom


def RelErr(pred, gt, segmask=None, eps=1e-12):
    """Relative magnitude error for vector fields. pred, gt: (Nv-1, Nt, SPE, PE, FE)."""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if segmask is None:
        segmask = np.ones(gt.shape[-3:], dtype=bool)
    segmask = np.asarray(segmask, dtype=bool)
    gt_mag = np.linalg.norm(gt, axis=0)
    pred_mag = np.linalg.norm(pred, axis=0)
    while segmask.ndim < gt_mag.ndim:
        segmask = segmask[None, ...]
    mask = np.broadcast_to(segmask, gt_mag.shape).astype(np.float32)
    numerator = np.sum(((gt_mag - pred_mag) ** 2) * mask)
    denominator = np.sum((gt_mag ** 2) * mask) + eps
    return np.sqrt(numerator / denominator)


def AngErr(pred, gt, segmask=None, eps=1e-8):
    """Angular error (degrees) for vector fields. pred, gt: (Nv-1, Nt, SPE, PE, FE)."""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if segmask is None:
        segmask = np.ones(gt.shape[-3:], dtype=bool)
    segmask = np.asarray(segmask, dtype=bool)
    dot = np.sum(pred * gt, axis=0)
    norm_p = np.linalg.norm(pred, axis=0)
    norm_g = np.linalg.norm(gt, axis=0)
    cos_sim = np.clip(dot / (norm_p * norm_g + eps), -1.0, 1.0)
    error_map = np.arccos(cos_sim)
    while segmask.ndim < error_map.ndim:
        segmask = segmask[None, ...]
    mask = np.broadcast_to(segmask, error_map.shape).astype(np.float32)
    n_valid = np.sum(mask) + 1e-12
    return (np.sum(error_map * mask) / n_valid) / np.pi * 180.0


def complex2magflow(x, venc=None):
    """(Nv, Nt, SPE, PE, FE) complex -> (magnitude, flow). Flow in radians unless ``venc`` is given."""
    mag = np.abs(x)
    flow = np.angle(x[1:] * np.conj(x[0:1]))
    if venc is not None:
        flow = flow / np.pi * np.asarray(venc)[:, None, None, None, None]
    return mag, flow


def save_coo_npz(path, arr):
    """Save a complex array as a sparse COO .npz (coords, data, shape)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = np.asarray(arr).astype('complex64')
    coords = np.argwhere(arr != 0).astype(np.int32)
    data = arr[tuple(coords.T)] if coords.size else arr.reshape(-1)[:0]
    np.savez_compressed(path, coords=coords, data=data, shape=np.array(arr.shape, dtype=np.int64))


def load_coo_npz(path, as_dense=True):
    z = np.load(path)
    coords, data, shape = z["coords"], z["data"], tuple(z["shape"])
    if not as_dense:
        return coords, data, shape
    out = np.zeros(shape, dtype=data.dtype)
    if coords.size:
        out[tuple(coords.T)] = data
    return out
