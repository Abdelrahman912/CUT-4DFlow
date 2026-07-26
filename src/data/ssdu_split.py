"""SSDU partitioning: split the acquired mask Omega into disjoint Theta (input) and Lambda (loss).

The network reconstructs from Theta and the loss is evaluated on the held-out Lambda samples.
An ACS centre disk (radius sqrt(r2)) is always kept in Theta.
"""

from __future__ import annotations

import numpy as np


def gen_center_disk(SPE: int, PE: int, r2: int = 9) -> np.ndarray:
    """Boolean disk of radius sqrt(r2) at the centre of the (SPE, PE) plane."""
    sp = np.arange(SPE) - SPE // 2
    pe = np.arange(PE) - PE // 2
    SP, PEi = np.meshgrid(sp, pe, indexing='ij')
    return SP ** 2 + PEi ** 2 < r2


def uniform_disjoint_selection(
    mask: np.ndarray,
    rho: float = 0.2,
    r2: int = 9,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition a (1, T, 1, SPE, PE, 1) mask into (theta, lambda) with a fraction ``rho`` in Lambda.

    Guarantees theta + lambda == mask, theta * lambda == 0, and the ACS disk stays in theta.
    """
    if mask.ndim != 6:
        raise ValueError(f'mask must be 6D (1, T, 1, SPE, PE, 1); got shape {mask.shape}')
    if mask.shape[0] != 1 or mask.shape[2] != 1 or mask.shape[5] != 1:
        raise ValueError(f'broadcast axes 0, 2, 5 must each be 1; got shape {mask.shape}')
    if not (0.0 < rho < 1.0):
        raise ValueError(f'rho must be in (0, 1); got {rho}')

    _, T, _, SPE, PE, _ = mask.shape
    mask_3d = mask[0, :, 0, :, :, 0]
    centre = gen_center_disk(SPE, PE, r2=r2)

    eligible = (mask_3d > 0).copy()
    eligible[:, centre] = False
    n_eligible = int(eligible.sum())
    n_lambda = int(round(n_eligible * rho))

    rng = np.random.RandomState(seed)
    eligible_indices = np.flatnonzero(eligible.ravel())
    chosen = rng.choice(eligible_indices, size=n_lambda, replace=False)

    lambda_flat = np.zeros(eligible.size, dtype=np.float32)
    lambda_flat[chosen] = 1.0
    lambda_3d = lambda_flat.reshape(eligible.shape)
    theta_3d = mask_3d - lambda_3d

    theta_6d = theta_3d[None, :, None, :, :, None].astype(np.float32)
    lambda_6d = lambda_3d[None, :, None, :, :, None].astype(np.float32)
    return theta_6d, lambda_6d
