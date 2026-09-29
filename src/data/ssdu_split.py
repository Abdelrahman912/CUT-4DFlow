"""SSDU partitioner — splits the acquired k-space mask M_Omega into M_Theta and M_Lambda.

In Self-Supervised learning via Data Undersampling (SSDU, Yaman 2020), the loss
cannot be computed against a ground-truth image because none exists at submission
time. Instead, the acquired k-space mask Omega is partitioned into two disjoint sets:

    Omega = Theta U Lambda      (Theta cap Lambda = empty)

- M_Theta: fed to the network as input  (the "training" mask, typically 80%)
- M_Lambda: held out for the loss term   (the "loss" mask, typically 20%)

The network reconstructs an image from kdata * M_Theta, then projects back to
k-space; the SSDU loss compares the prediction at M_Lambda points against the
truly-acquired samples there.

Ported from flowmri_net-main/utils/partitioning.py:uniform_disjoint_selection with
three deliberate changes:

1. Works natively on our 6D mask shape ``(1, T, 1, SPE, PE, 1)``.
2. Default ``rho=0.2`` (80/20 split: 80% Theta, 20% Lambda) — matches the
   FlowMRI-Net / SSDU paper convention.
3. Uses ``np.random.RandomState(seed)`` instead of mutating the global RNG.

The ACS centre disk (default r2=9 -> radius sqrt(9) = 3 pixels) is ALWAYS kept
in Theta and never assigned to Lambda. This guards the DC anchor against being
moved to the loss side.
"""

from __future__ import annotations

import numpy as np


def gen_center_disk(SPE: int, PE: int, r2: int = 9) -> np.ndarray:
    """Boolean disk at the centre of the (SPE, PE) plane, radius sqrt(r2) px.

    Used to identify pixels that must always remain in Theta (never moved to Lambda).

    Parameters
    ----------
    SPE, PE :
        Phase-encode dimensions.
    r2 :
        Squared radius. Default r2=9 -> radius 3 px, matching the FlowVN /
        FlowMRI-Net protocol.

    Returns
    -------
    disk : (SPE, PE) bool — True inside the centre disk.
    """
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
    """Partition an acquired k-space mask into disjoint Theta + Lambda sets.

    Parameters
    ----------
    mask :
        Mask of shape ``(1, T, 1, SPE, PE, 1)`` with values in {0, 1}. Typically
        the output of ``src.data.ktgaussian.make_ktgaussian_mask``.
    rho :
        Fraction of non-ACS acquired points assigned to Lambda. Default 0.2
        gives an 80/20 split (80% Theta, 20% Lambda).
    r2 :
        Squared radius of the ACS centre disk that stays in Theta. Default 9.
    seed :
        If provided, the split is fully reproducible. If None, uses fresh randomness.

    Returns
    -------
    theta_mask : (1, T, 1, SPE, PE, 1) float32 — network input mask.
    lambda_mask : (1, T, 1, SPE, PE, 1) float32 — loss target mask.

    Guarantees
    ----------
    - ``theta_mask + lambda_mask == mask``  (no points lost or duplicated)
    - ``theta_mask * lambda_mask == 0``     (mutually exclusive)
    - ACS centre disk pixels never appear in ``lambda_mask``
    """
    if mask.ndim != 6:
        raise ValueError(
            f'mask must be 6D (1, T, 1, SPE, PE, 1); got shape {mask.shape}'
        )
    if mask.shape[0] != 1 or mask.shape[2] != 1 or mask.shape[5] != 1:
        raise ValueError(
            f'broadcast axes 0, 2, 5 must each be 1; got shape {mask.shape}'
        )
    if not (0.0 < rho < 1.0):
        raise ValueError(f'rho must be in (0, 1); got {rho}')

    _, T, _, SPE, PE, _ = mask.shape

    # Squeeze to (T, SPE, PE) for the partitioning logic
    mask_3d = mask[0, :, 0, :, :, 0]                  # (T, SPE, PE) float32

    # ACS centre disk: pixels here are always in Theta.
    centre = gen_center_disk(SPE, PE, r2=r2)           # (SPE, PE) bool

    # Eligible-for-Lambda = acquired AND NOT in the centre disk.
    eligible = (mask_3d > 0).copy()                    # (T, SPE, PE) bool
    eligible[:, centre] = False

    n_eligible = int(eligible.sum())
    n_lambda = int(round(n_eligible * rho))

    rng = np.random.RandomState(seed)
    eligible_indices = np.flatnonzero(eligible.ravel())
    chosen = rng.choice(eligible_indices, size=n_lambda, replace=False)

    # Build the Lambda mask (and Theta as the complement within the acquired set)
    lambda_flat = np.zeros(eligible.size, dtype=np.float32)
    lambda_flat[chosen] = 1.0
    lambda_3d = lambda_flat.reshape(eligible.shape)    # (T, SPE, PE)

    theta_3d = mask_3d - lambda_3d                     # acquired \ lambda

    # Re-broadcast to the 6D layout the dataloader / loss expect
    theta_6d  = theta_3d[None, :, None, :, :, None].astype(np.float32)
    lambda_6d = lambda_3d[None, :, None, :, :, None].astype(np.float32)

    return theta_6d, lambda_6d
