"""kt-Gaussian undersampling mask generator.

A different variable-density (Gaussian, Poisson-disk-constrained) mask per cardiac frame, with a
fully-sampled centre. ``make_ktgaussian_mask`` returns the canonical (1, T, 1, SPE, PE, 1) layout.
"""

from __future__ import annotations

import numpy as np


def create_gaussian_weight_matrix(mask_size, sigma_x, sigma_y):
    """2-D anisotropic Gaussian sampling-density map on the (PE, SPE) grid -> (SPE, PE)."""
    width, height = mask_size
    x = np.linspace(1 - (width + 1) / 2, width - (width + 1) / 2, width)
    y = np.linspace(1 - (height + 1) / 2, height - (height + 1) / 2, height)
    X, Y = np.meshgrid(x, y)
    return np.exp(-(X ** 2 / (2 * sigma_x ** 2) + Y ** 2 / (2 * sigma_y ** 2)))


def random_sampling_optimized(mask_size, total_points, weight, min_dist_lookup, existing_mask, rng=None):
    """Draw up to ``total_points`` samples from ``weight`` with a per-location minimum-distance rule.

    Returns an (N, 2) array of 1-based [x, y] coordinates.
    """
    if rng is None:
        rng = np.random.default_rng()
    width, height = mask_size
    sampled_points: list[list[int]] = []

    current_weight = weight * (1 - existing_mask)
    if np.sum(current_weight) <= 0:
        return np.array([])
    prob = current_weight.ravel() / current_weight.sum()

    forbidden_mask = np.zeros((height, width), dtype=bool)
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

        sampled_points.append([x + 1, y + 1])
        count += 1

        d = float(min_dist_lookup[y, x])
        if d > 0:
            y_min, y_max = max(0, int(y - d)), min(height, int(y + d + 1))
            x_min, x_max = max(0, int(x - d)), min(width, int(x + d + 1))
            region_y = Y_grid[y_min:y_max, 0]
            region_x = X_grid[0, x_min:x_max]
            dist_sq = (region_y[:, np.newaxis] - y) ** 2 + (region_x - x) ** 2
            forbidden_mask[y_min:y_max, x_min:x_max] |= dist_sq < d ** 2

        if idx_ptr >= batch_size and count < total_points:
            current_weight = weight * (1 - existing_mask) * (1 - forbidden_mask)
            sw = current_weight.sum()
            if sw <= 0:
                break
            prob = current_weight.ravel() / sw
            indices = rng.choice(width * height, size=batch_size, p=prob)
            idx_ptr = 0

    return np.array(sampled_points)


def fun_mask_gen_2d(mask_size, total_points, pattern_num, sigma_x, sigma_y,
                    min_dist_factor=3.0, rep_decay_factor=0.5,
                    center_radius_x=0.5, center_radius_y=0.5, rng=None):
    """Generate ``pattern_num`` masks of shape (SPE, PE), one per frame.

    ``mask_size`` is (PE, SPE); ``total_points`` includes the forced centre. A centre ellipse of
    radii (center_radius_x, center_radius_y) is always sampled (single pixel if a radius <= 0.5).
    """
    if rng is None:
        rng = np.random.default_rng()
    width, height = mask_size
    masks = np.zeros((height, width, pattern_num), dtype=np.float32)

    initial_weight = create_gaussian_weight_matrix(mask_size, sigma_x, sigma_y)
    weight = initial_weight.copy()
    min_dist_lookup = min_dist_factor * ((1.0 - initial_weight) / 2.0 + 0.5)

    if (center_radius_x <= 0.5) or (center_radius_y <= 0.5):
        center_ellipse = np.zeros((height, width), dtype=bool)
        center_ellipse[height // 2, width // 2] = True
    else:
        cy, cx = height // 2, width // 2
        Xc, Yc = np.meshgrid(np.arange(width) - cx, np.arange(height) - cy)
        center_ellipse = (Xc / center_radius_x) ** 2 + (Yc / center_radius_y) ** 2 <= 1

    num_center_points = int(np.sum(center_ellipse))

    for p in range(pattern_num):
        mask = np.zeros((height, width), dtype=np.float32)
        mask[center_ellipse] = 1

        needed = total_points - num_center_points
        if needed > 0:
            points = random_sampling_optimized(mask_size, needed, weight, min_dist_lookup, mask, rng=rng)
            if len(points) > 0:
                xs = points[:, 0].astype(int) - 1
                ys = points[:, 1].astype(int) - 1
                mask[ys, xs] = 1
                weight[ys, xs] *= rep_decay_factor

        curr_total = int(np.sum(mask))
        if curr_total < total_points:
            extra_points = random_sampling_optimized(mask_size, total_points - curr_total,
                                                     weight, min_dist_lookup, mask, rng=rng)
            if len(extra_points) > 0:
                xs = extra_points[:, 0].astype(int) - 1
                ys = extra_points[:, 1].astype(int) - 1
                mask[ys, xs] = 1
                weight[ys, xs] *= rep_decay_factor

        masks[:, :, p] = mask

    return masks


def make_ktgaussian_mask(SPE: int, PE: int, Nt: int, R: int, seed: int | None = None) -> np.ndarray:
    """kt-Gaussian mask of shape (1, Nt, 1, SPE, PE, 1) for undersampling factor R.

    Uses a process-isolated RNG so the global numpy state is never mutated.
    """
    rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()
    masks_spe_pe_t = fun_mask_gen_2d(
        mask_size=(PE, SPE), total_points=PE * SPE // R, pattern_num=Nt,
        sigma_x=PE / 5.0, sigma_y=SPE / 5.0, min_dist_factor=3, rep_decay_factor=0.5,
        center_radius_x=0.5, center_radius_y=0.5, rng=rng,
    )
    mask = np.moveaxis(masks_spe_pe_t, -1, 0)
    return mask[None, :, None, :, :, None]
