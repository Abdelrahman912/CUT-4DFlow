"""kt-Gaussian undersampling mask generator for CMRx4DFlow.

Ported from CMRx4DFlow2026/CMRx4DFlowMaskGeneration/ktgaussian.py with two
deliberate changes:

1. Fixed the "ellipse branch" of ``fun_mask_gen_2d``. Upstream uses
   ``((width - (width - 1) / 2) / center_radius_x) ** 2`` which evaluates to
   a SCALAR (not a per-pixel grid), so the fully-sampled centre collapses
   to a single point. We replace it with a proper ``np.meshgrid``-based
   ellipse so the ACS region behaves like an ellipse of radii
   ``(center_radius_x, center_radius_y)``.

2. Added ``make_ktgaussian_mask`` — a thin wrapper that:
     - takes the parameters the dataloader actually has (SPE, PE, Nt, R)
     - reshapes the (SPE, PE, T) output to the canonical
       (1, T, 1, SPE, PE, 1) broadcast shape
     - exposes an explicit ``seed`` arg for reproducibility (upstream uses
       the global numpy RNG)
     - uses ``center_radius_x = center_radius_y = 0.5`` (single-pixel ACS
       fallback) to match the SHIPPED validation masks. Verified directly
       from the .mat files: every shipped ``usmask_ktGaussian{R}.mat``
       has exactly 1 always-sampled centre pixel and realised R within
       1% of the requested R. The FlowVN training dataloader uses the
       same convention.

Why kt-Gaussian:
    - "kt" = different 2D mask per cardiac frame -> temporal incoherence
      that compressed-sensing / deep recon can exploit
    - "Gaussian" = sampling probability higher at k-space centre, lower at
      edges -> preserves SNR + low-frequency content where it matters
    - Poisson-disk constraint prevents clustering -> samples are spread
    - Centre ellipse is ALWAYS sampled (ACS-equivalent for SENSE calibration
      and SNR baseline)
"""

from __future__ import annotations

import math

import numpy as np


# ─── Step 1: Gaussian weight matrix (port verbatim — already correct) ────
def create_gaussian_weight_matrix(
    mask_size: tuple[int, int], sigma_x: float, sigma_y: float
) -> np.ndarray:
    """2D anisotropic Gaussian density on the (PE, SPE) grid.

    Higher near centre, lower at edges. Used as the sampling-probability map.

    Parameters
    ----------
    mask_size : (PE, SPE)
        Width = PE, Height = SPE.
    sigma_x, sigma_y :
        Standard deviations along PE and SPE.

    Returns
    -------
    weight : (SPE, PE) float64
    """
    width, height = mask_size
    x = np.linspace(1 - (width + 1) / 2, width - (width + 1) / 2, width)
    y = np.linspace(1 - (height + 1) / 2, height - (height + 1) / 2, height)
    X, Y = np.meshgrid(x, y)
    return np.exp(-(X ** 2 / (2 * sigma_x ** 2) + Y ** 2 / (2 * sigma_y ** 2)))


# ─── Step 2: Weighted Poisson-disk sampler (process-isolated RNG) ─────────
def random_sampling_optimized(
    mask_size: tuple[int, int],
    total_points: int,
    weight: np.ndarray,
    min_dist_lookup: np.ndarray,
    existing_mask: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw ``total_points`` samples from ``weight`` while honouring a
    per-location minimum-distance (Poisson-disk) constraint.

    Returns (N, 2) array of 1-based [x, y] coordinates, where N <= total_points.

    ``rng`` is a ``numpy.random.Generator``; if ``None``, a fresh
    ``np.random.default_rng()`` (process-isolated) is constructed so the
    global ``np.random`` state is never mutated.
    """
    if rng is None:
        rng = np.random.default_rng()
    width, height = mask_size
    sampled_points: list[list[int]] = []

    # Exclude already-selected points from the probability mass.
    current_weight = weight * (1 - existing_mask)
    if np.sum(current_weight) <= 0:
        return np.array([])

    flat_weight = current_weight.ravel()
    prob = flat_weight / flat_weight.sum()

    forbidden_mask = np.zeros((height, width), dtype=bool)

    # Draw more candidates than needed to amortise rejections.
    batch_size = max(total_points * 2, 1000)
    indices = rng.choice(width * height, size=batch_size, p=prob)

    count = 0
    idx_ptr = 0
    Y_grid, X_grid = np.ogrid[:height, :width]

    while count < total_points and idx_ptr < batch_size:
        idx = int(indices[idx_ptr])
        idx_ptr += 1

        y, x = divmod(idx, width)

        if forbidden_mask[y, x] or existing_mask[y, x]:
            continue

        # Store 1-based to match the original implementation's convention.
        sampled_points.append([x + 1, y + 1])
        count += 1

        # Mark neighbourhood within radius d as forbidden.
        d = float(min_dist_lookup[y, x])
        if d > 0:
            y_min, y_max = max(0, int(y - d)), min(height, int(y + d + 1))
            x_min, x_max = max(0, int(x - d)), min(width, int(x + d + 1))

            region_y = Y_grid[y_min:y_max, 0]
            region_x = X_grid[0, x_min:x_max]

            dist_sq = (region_y[:, np.newaxis] - y) ** 2 + (region_x - x) ** 2
            forbidden_mask[y_min:y_max, x_min:x_max] |= dist_sq < d ** 2

        # Out of candidates? Resample from remaining valid region.
        if idx_ptr >= batch_size and count < total_points:
            current_weight = weight * (1 - existing_mask) * (1 - forbidden_mask)
            sw = current_weight.sum()
            if sw <= 0:
                break
            prob = current_weight.ravel() / sw
            indices = rng.choice(width * height, size=batch_size, p=prob)
            idx_ptr = 0

    return np.array(sampled_points)


# ─── Step 3: Main generator (with the ellipse-branch fix) ─────────────────
def fun_mask_gen_2d(
    mask_size: tuple[int, int],
    total_points: int,
    pattern_num: int,
    sigma_x: float,
    sigma_y: float,
    min_dist_factor: float = 3.0,
    rep_decay_factor: float = 0.5,
    center_radius_x: float = 0.5,
    center_radius_y: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate ``pattern_num`` 2D kt-Gaussian masks of shape (SPE, PE).

    Parameters
    ----------
    mask_size : (PE, SPE)
        (width, height). Note the (width, height) order, NOT (height, width).
    total_points :
        Target samples per frame INCLUDING the forced centre ellipse.
        For target R: ``total_points = PE * SPE // R``.
    pattern_num :
        Number of frames (Nt). One independent mask per frame.
    sigma_x, sigma_y :
        Gaussian sigmas along PE and SPE. Typical: PE/5, SPE/5.
    min_dist_factor :
        Multiplies the per-location Poisson-disk exclusion radius. Larger ->
        more spread-out samples.
    rep_decay_factor :
        Weight applied to a location after it has been sampled (for the
        next frame). < 1.0 discourages resampling the same spot across frames.
    center_radius_x, center_radius_y :
        Radii of the fully-sampled centre ellipse, in pixels along PE and SPE.
        If either is <= 0.5, the fallback "single centre pixel" branch fires.

    Returns
    -------
    masks : (SPE, PE, pattern_num) float32, values {0, 1}
    """
    if rng is None:
        rng = np.random.default_rng()
    width, height = mask_size
    masks = np.zeros((height, width, pattern_num), dtype=np.float32)

    initial_weight = create_gaussian_weight_matrix(mask_size, sigma_x, sigma_y)
    weight = initial_weight.copy()

    # Larger exclusion radii in low-weight (outer) regions, smaller near centre.
    min_dist_lookup = min_dist_factor * ((1.0 - initial_weight) / 2.0 + 0.5)

    # Forced fully-sampled centre region.
    if (center_radius_x <= 0.5) or (center_radius_y <= 0.5):
        # Fallback: single centre pixel.
        center_ellipse = np.zeros((height, width), dtype=bool)
        center_ellipse[height // 2, width // 2] = True
    else:
        # FIX vs upstream: use a real meshgrid so the ellipse spans actual pixels.
        # The condition (x/cx)^2 + (y/cy)^2 <= 1 with x,y centered on the grid.
        cy = height // 2
        cx = width // 2
        x_coord = np.arange(width) - cx
        y_coord = np.arange(height) - cy
        Xc, Yc = np.meshgrid(x_coord, y_coord)
        center_ellipse = (
            (Xc / center_radius_x) ** 2 + (Yc / center_radius_y) ** 2 <= 1
        )

    num_center_points = int(np.sum(center_ellipse))

    for p in range(pattern_num):
        mask = np.zeros((height, width), dtype=np.float32)
        # Always include the forced centre.
        mask[center_ellipse] = 1

        # First pass: fill toward target count.
        needed = total_points - num_center_points
        if needed > 0:
            points = random_sampling_optimized(
                mask_size, needed, weight, min_dist_lookup, mask, rng=rng,
            )
            if len(points) > 0:
                xs = points[:, 0].astype(int) - 1
                ys = points[:, 1].astype(int) - 1
                mask[ys, xs] = 1
                weight[ys, xs] *= rep_decay_factor

        # Second pass: top up if still short.
        curr_total = int(np.sum(mask))
        if curr_total < total_points:
            extra = total_points - curr_total
            extra_points = random_sampling_optimized(
                mask_size, extra, weight, min_dist_lookup, mask, rng=rng,
            )
            if len(extra_points) > 0:
                xs = extra_points[:, 0].astype(int) - 1
                ys = extra_points[:, 1].astype(int) - 1
                mask[ys, xs] = 1
                weight[ys, xs] *= rep_decay_factor

        masks[:, :, p] = mask

    return masks


# ─── A7 — Dataloader-shaped wrapper ───────────────────────────────────────
def make_ktgaussian_mask(
    SPE: int,
    PE: int,
    Nt: int,
    R: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a kt-Gaussian undersampling mask in the canonical 6D layout.

    Parameters
    ----------
    SPE, PE :
        Phase-encode dims. The mask is dense in (SPE, PE); FE is broadcast.
    Nt :
        Number of cardiac frames. One independent mask per frame.
    R :
        Target undersampling factor in {10, 20, 30, 40, 50}.
    seed :
        If given, builds a process-isolated ``np.random.default_rng(seed)``
        and threads it through the generator. The global ``np.random`` state
        is *never* mutated, so dataloader workers and parallel processes
        don't poison each other.

    Returns
    -------
    mask : (1, Nt, 1, SPE, PE, 1) float32 in {0, 1}
        Broadcastable over (Nv, Nc, FE). Multiply with kdata to undersample.
    """
    rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()

    # Match the shipped validation masks: 1-pixel ACS (the upstream
    # `center_radius <= 0.5` fallback) and PE/5, SPE/5 Gaussian widths.
    masks_spe_pe_t = fun_mask_gen_2d(
        mask_size=(PE, SPE),                 # (width, height)
        total_points=PE * SPE // R,
        pattern_num=Nt,
        sigma_x=PE / 5.0,
        sigma_y=SPE / 5.0,
        min_dist_factor=3,
        rep_decay_factor=0.5,
        center_radius_x=0.5,                 # ← single-pixel ACS fallback
        center_radius_y=0.5,
        rng=rng,
    )                                          # (SPE, PE, Nt)

    # Reshape to canonical (1, Nt, 1, SPE, PE, 1)
    mask = np.moveaxis(masks_spe_pe_t, -1, 0)  # (Nt, SPE, PE)
    return mask[None, :, None, :, :, None]     # (1, Nt, 1, SPE, PE, 1)
