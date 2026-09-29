"""Orientation canonicalisation for CMRx4DFlow data.

The CMRx training set spans 6 scanners with 4 distinct ``spatial_order``
conventions. We pick one canonical convention and flip axes per-patient
so that downstream code sees consistent orientation.

Canonical convention:
    CANONICAL_SPATIAL = ['HF', 'AP', 'LR']

Axis-position to numpy-axis mapping (load-bearing — verified from FOV
arithmetic: ``108 voxels x 2.4mm = 259.2mm = FOV[0]``):

    spatial_order[0]  ->  FE direction  ->  array axis -1
    spatial_order[1]  ->  PE direction  ->  array axis -2
    spatial_order[2]  ->  SPE direction ->  array axis -3

A1 (``_axis_is_reversed``) is the boolean decision per axis.
A2 (``canonicalise_spatial``) is the actual array-flipping function the
dataloader will call on every kdata / coilmap / segmask / image tensor.
"""

from __future__ import annotations

import numpy as np


CANONICAL_SPATIAL: list[str] = ['HF', 'AP', 'LR']
"""Canonical direction tags, one per axis position (FE, PE, SPE)."""

REVERSED_PAIRS: dict[str, str] = {'HF': 'FH', 'AP': 'PA', 'LR': 'RL'}
"""Maps canonical tag -> its reversed twin."""

# NOTE: tag here is a string like 'HF' or 'PA', not the whole spatial_order list.
# NOTE: axis_position is 0, 1, or 2.
def _axis_is_reversed(tag: str, axis_position: int) -> bool:
    """Return True if ``tag`` is the reversed twin of the canonical at this position.

    Parameters
    ----------
    tag :
        One of the 6 direction tags: HF/FH, AP/PA, LR/RL.
    axis_position :
        Position in ``spatial_order`` (0 = FE, 1 = PE, 2 = SPE).

    Returns
    -------
    bool
        False if ``tag`` already matches the canonical at this position;
        True if it is the reversed twin (axis must be flipped to canonicalise).

    Raises
    ------
    ValueError
        If ``tag`` is neither the canonical nor its reversed twin at this
        position. This catches both unknown tags (e.g. new scanner) and known
        tags placed at the wrong position (e.g. ``LR`` at position 0).
    """
    if axis_position not in (0, 1, 2):
        raise ValueError(
            f'axis_position must be 0, 1, or 2; got {axis_position!r}'
        )
    canonical = CANONICAL_SPATIAL[axis_position]
    if tag == canonical:
        return False
    if tag == REVERSED_PAIRS[canonical]:
        return True
    raise ValueError(
        f'unknown or misplaced spatial_order tag {tag!r} at position '
        f'{axis_position} (canonical here is {canonical!r}, '
        f'reversed twin is {REVERSED_PAIRS[canonical]!r})'
    )


def flip_origin(arr: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Reverse ``arr`` about the DFT origin along each axis in ``axes``.

    Why not ``np.flip``? ``np.flip`` reverses about the array's geometric midpoint,
    which on an EVEN-length axis sits half a voxel off the DFT centre. Applied to
    **k-space** that produces a ~1-voxel image shift (a linear phase) — i.e. a
    k-space ``np.flip`` does NOT equal flipping the image. Reversing about the DFT
    origin instead commutes with the Fourier transform with NO phase term, so it is
    **exact in both image and k-space for any axis length**, and it is its own
    inverse (involution). Proven in ``cmrx/DebugCMR.ipynb`` (Issue #1).

    Implementation: ifftshift (DC -> index 0) -> reverse keeping index 0 fixed
    (``x[-n]`` = ``roll(flip, 1)``) -> fftshift (DC -> centre).
    """
    a = np.fft.ifftshift(arr, axes=axes)
    a = np.roll(np.flip(a, axis=axes), 1, axis=axes)
    return np.fft.fftshift(a, axes=axes)


# ─── A2 ───────────────────────────────────────────────────────────────────
# Actually flip the array axes to canonical (HF, AP, LR) orientation.
# Works on any tensor with spatial axes at positions (-3, -2, -1) = (SPE, PE, FE):
#   kdata    (Nv, Nt, Nc, SPE, PE, FE)
#   coilmap  (Nc,         SPE, PE, FE)
#   segmask  (             SPE, PE, FE)
#   image    (Nv, Nt,     SPE, PE, FE)
# NOTE: FE -> X, PE -> Y, SPE -> Z, undersampling is always in Y-Z direction
def canonicalise_spatial(arr: np.ndarray, spatial_order: list[str]) -> np.ndarray:
    """Flip ``arr`` so its last 3 axes are in canonical (HF, AP, LR) orientation.

    Parameters
    ----------
    arr :
        Array with at least 3 spatial axes at positions ``(-3, -2, -1)`` =
        ``(SPE, PE, FE)``. Leading dimensions are preserved verbatim. Works
        on kdata (6D), coilmap (4D), segmask (3D), recon image (5D), etc.
    spatial_order :
        The 3-element direction list from ``params.csv``, e.g. ``['HF','AP','LR']``
        or ``['FH','PA','RL']``.

    Returns
    -------
    np.ndarray
        A new C-contiguous array with the same dtype and shape as ``arr``.
        Axes whose ``spatial_order`` tag is the reversed twin of canonical
        have been flipped via ``flip_origin`` (reverse about the DFT origin;
        exact in both image & k-space, unlike ``np.flip`` on even-length axes).

    Raises
    ------
    ValueError
        - ``len(spatial_order) != 3``.
        - Any tag is unknown or at the wrong position (raised by ``_axis_is_reversed``).
        - ``arr.ndim < 3`` (not enough axes for the 3 spatial dims).
    """
    if len(spatial_order) != 3:
        raise ValueError(
            f'spatial_order must have 3 entries; got {len(spatial_order)} '
            f'({spatial_order!r})'
        )
    if arr.ndim < 3:
        raise ValueError(
            f'arr must have at least 3 axes (SPE, PE, FE); got {arr.ndim}D '
            f'with shape {arr.shape}'
        )

    # spatial_order[i] -> numpy axis (FE=-1, PE=-2, SPE=-3)
    axis_map = [-1, -2, -3]
    flip_axes = tuple(
        axis_map[i]
        for i, tag in enumerate(spatial_order)
        if _axis_is_reversed(tag, i)
    )

    if not flip_axes:
        # No flips needed — return an explicit copy so the caller owns the array
        # and modifying it cannot alias back into ``arr``. ``ascontiguousarray``
        # would have returned ``arr`` unchanged for a contiguous input.
        return arr.copy()

    # Reverse about the DFT origin (NOT np.flip): exact in both image & k-space for
    # any axis length, so the k-space flip matches the image flip and canonicalise/
    # decanonicalise round-trip exactly. See flip_origin docstring + DebugCMR Issue #1.
    flipped = flip_origin(arr, flip_axes)
    return np.ascontiguousarray(flipped)


# ─── A3 ───────────────────────────────────────────────────────────────────
# Inverse of canonicalise_spatial: un-flips back to the patient's native
# orientation. Called by the submission writer so that img_ktGaussian{R}.npz
# is in the same orientation as the organiser's hidden GT (which is the
# patient's native on-disk orientation).
#
# Mathematical fact: axis flips are involutions — flipping the same axis
# twice returns the original. So the inverse of canonicalise_spatial under
# the same spatial_order is structurally identical to canonicalise_spatial.
# We keep it as a separate named function for call-site readability:
#     canonicalise_spatial(arr, so)    # input side
#     decanonicalise_spatial(arr, so)  # output side
def decanonicalise_spatial(arr: np.ndarray, spatial_order: list[str]) -> np.ndarray:
    """Inverse of ``canonicalise_spatial`` — un-flips back to native orientation.

    Used by the submission writer:

    .. code-block:: python

        img_canon  = network(kdata_canon)
        img_native = decanonicalise_spatial(img_canon, spatial_order)
        save_coo_npz(out_path, img_native * segmask_native[None, None])

    Axis flips are involutions, so this delegates to ``canonicalise_spatial``.
    A separate name documents intent at call sites.

    Parameters
    ----------
    arr :
        Array with at least 3 spatial axes at positions ``(-3, -2, -1)`` =
        ``(SPE, PE, FE)``, currently in canonical (HF, AP, LR) orientation.
    spatial_order :
        The 3-element ``params.csv`` direction list of the **target native**
        orientation. Same ``spatial_order`` that was used at canonicalise time.

    Returns
    -------
    np.ndarray
        A new C-contiguous array in the orientation described by ``spatial_order``.
    """
    return canonicalise_spatial(arr, spatial_order)
