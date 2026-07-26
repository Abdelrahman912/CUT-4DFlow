"""Spatial-orientation canonicalisation.

The dataset spans scanners with different axis-direction conventions. We flip axes per patient
to a canonical orientation (HF, AP, LR) so downstream code sees consistent data. Only axis flips
are handled here (not permutations).
"""

from __future__ import annotations

import numpy as np


CANONICAL_SPATIAL: list[str] = ['HF', 'AP', 'LR']
REVERSED_PAIRS: dict[str, str] = {'HF': 'FH', 'AP': 'PA', 'LR': 'RL'}


def _axis_is_reversed(tag: str, axis_position: int) -> bool:
    """True if ``tag`` is the reversed twin of the canonical direction at this axis position."""
    if axis_position not in (0, 1, 2):
        raise ValueError(f'axis_position must be 0, 1, or 2; got {axis_position!r}')
    canonical = CANONICAL_SPATIAL[axis_position]
    if tag == canonical:
        return False
    if tag == REVERSED_PAIRS[canonical]:
        return True
    raise ValueError(f'unknown or misplaced spatial_order tag {tag!r} at position {axis_position}')


def flip_origin(arr: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Reverse ``arr`` about the DFT origin along ``axes``.

    Unlike ``np.flip`` (which reverses about the geometric midpoint and shifts by half a voxel on
    even-length axes), this commutes with the Fourier transform, so it is exact in both image and
    k-space and is its own inverse.
    """
    a = np.fft.ifftshift(arr, axes=axes)
    a = np.roll(np.flip(a, axis=axes), 1, axis=axes)
    return np.fft.fftshift(a, axes=axes)


def canonicalise_spatial(arr: np.ndarray, spatial_order: list[str]) -> np.ndarray:
    """Flip ``arr`` (last three axes = SPE, PE, FE) to canonical (HF, AP, LR) orientation."""
    if len(spatial_order) != 3:
        raise ValueError(f'spatial_order must have 3 entries; got {spatial_order!r}')
    if arr.ndim < 3:
        raise ValueError(f'arr must have at least 3 axes; got {arr.ndim}D')

    axis_map = [-1, -2, -3]                                 # spatial_order[i] -> numpy axis (FE, PE, SPE)
    flip_axes = tuple(axis_map[i] for i, tag in enumerate(spatial_order) if _axis_is_reversed(tag, i))
    if not flip_axes:
        return arr.copy()
    return np.ascontiguousarray(flip_origin(arr, flip_axes))


def decanonicalise_spatial(arr: np.ndarray, spatial_order: list[str]) -> np.ndarray:
    """Inverse of ``canonicalise_spatial`` (axis flips are involutions)."""
    return canonicalise_spatial(arr, spatial_order)
