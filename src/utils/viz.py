"""Visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.utils.cmrx_metrics import complex2magflow


def plot_ktgaussian_mask(mask: np.ndarray, t0: int = 0, z0: int | None = None,
                         R: float | None = None, figsize: tuple[float, float] = (6, 6)):
    """Plot a kt-Gaussian undersampling mask: Y-Z plane at frame ``t0`` and Y-T at slice ``z0``.

    ``mask`` is either the canonical (1, Nt, 1, SPE, PE, 1) layout from
    ``make_ktgaussian_mask`` or an already-squeezed (Nt, SPE, PE) array.
    ``z0`` defaults to the centre kz slice; ``R`` defaults to the achieved
    acceleration computed from the mask. Pixels are kept square so the panel
    proportions follow the data shape. Returns ``(fig, axes)``.
    """
    mask_tzy = np.asarray(mask).squeeze()
    if mask_tzy.ndim != 3:
        raise ValueError(f'expected mask squeezable to (Nt, SPE, PE), got shape {mask.shape}')
    Nt, SPE, PE = mask_tzy.shape
    if z0 is None:
        z0 = SPE // 2
    if R is None:
        R = mask_tzy.size / mask_tzy.sum()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(mask_tzy[t0].T, cmap='gray', aspect='equal', interpolation='nearest')
    axes[0].set(title=f'frame t={t0}', xlabel='Z (SPE)', ylabel='Y (PE)')

    axes[1].imshow(mask_tzy[:, z0, :].T, cmap='gray', aspect='equal', interpolation='nearest')
    axes[1].set(title=f'$k_z$={z0}', xlabel='T (frame)', ylabel='Y (PE)')

    fig.suptitle(f'kt-Gaussian mask, R={R:.3g}', y=0.98)
    fig.tight_layout()
    return fig, axes


def plot_mag_velocity(img: np.ndarray, venc=None, t0: int = 0, z0: int | None = None,
                      cmap: str = 'jet', upright: bool = True, vmax=None,
                      segmask: np.ndarray | None = None, crop: bool | int = 4,
                      figsize: tuple[float, float] = (10, 5)):
    """Plot magnitude and velocity-norm maps of a 4D-flow image at frame ``t0``, slice ``z0``.

    ``img`` is (Nv=4, Nt, SPE, PE, FE) complex: encoding 0 is the flow-compensated
    reference, encodings 1..3 the three velocity directions. Magnitude is averaged
    over encodings; the velocity norm is the L2 norm of the three phase-difference
    components (cm/s if ``venc`` is given, e.g. ``params['VENC']``, else radians).

    ``z0`` defaults to the centre slice. ``upright=True`` puts the head-foot axis
    (FE) vertical, as in the anatomical views used in the paper figures; set it to
    False to keep the raw (PE, FE) array layout.

    Pass ``segmask`` (SPE, PE, FE bool) to blank the velocity map outside the vessel
    — outside the mask the magnitude is ~0 so the phase difference is pure noise,
    which otherwise dominates the colour scale. ``crop`` then trims to the mask
    bounding box plus that many voxels of margin (False keeps the full field of
    view). ``vmax`` defaults to the 99th percentile of the in-mask speed, rounded
    up to a round number. Returns ``(fig, axes)``.
    """
    img = np.asarray(img)
    if img.ndim != 5 or img.shape[0] != 4:
        raise ValueError(f'expected img of shape (4, Nt, SPE, PE, FE), got {img.shape}')
    if z0 is None:
        z0 = img.shape[2] // 2

    mag, flow = complex2magflow(img[:, t0:t0 + 1], venc=venc)
    mag_sl = mag[:, 0, z0].mean(axis=0)                        # (PE, FE)
    speed_sl = np.linalg.norm(flow[:, 0, z0], axis=0)          # (PE, FE)
    seg_sl = np.asarray(segmask)[z0].astype(bool) if segmask is not None else None

    if upright:                                                # -> (FE, PE), head-foot vertical
        mag_sl, speed_sl = mag_sl.T, speed_sl.T
        seg_sl = seg_sl.T if seg_sl is not None else None

    if seg_sl is not None:
        if vmax is None and seg_sl.any():
            p99 = np.percentile(speed_sl[seg_sl], 99)
            step = 10 ** np.floor(np.log10(max(p99, 1e-6)))
            vmax = float(np.ceil(p99 / (step / 2)) * (step / 2))
        speed_sl = np.where(seg_sl, speed_sl, np.nan)          # NaN renders as the 'bad' colour
        if crop is not False and seg_sl.any():
            m = int(crop)
            rows, cols = np.where(seg_sl)
            rs = slice(max(rows.min() - m, 0), min(rows.max() + m + 1, seg_sl.shape[0]))
            cs = slice(max(cols.min() - m, 0), min(cols.max() + m + 1, seg_sl.shape[1]))
            mag_sl, speed_sl = mag_sl[rs, cs], speed_sl[rs, cs]

    ylabel, xlabel = ('FE (X)', 'PE (Y)') if upright else ('PE (Y)', 'FE (X)')

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(mag_sl, cmap='gray', aspect='equal')
    axes[0].set(title='magnitude', xlabel=xlabel, ylabel=ylabel)

    speed_cmap = plt.get_cmap(cmap).copy()
    speed_cmap.set_bad('black')
    im = axes[1].imshow(speed_sl, cmap=speed_cmap, aspect='equal', vmin=0, vmax=vmax)
    axes[1].set(title=r'velocity norm  $v=|\mathbf{v}|$', xlabel=xlabel, ylabel=ylabel)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04,
                 label='[cm/s]' if venc is not None else '[rad]')

    fig.suptitle(f'frame t={t0}, slice z={z0}', y=0.98)
    fig.tight_layout()
    return fig, axes
