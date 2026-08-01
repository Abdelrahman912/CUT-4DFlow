"""CG-SENSE baseline (Pruessmann et al., 2001).

Solves the Tikhonov-damped SENSE normal equations

    (E^H E + lam I) x = E^H y,     E = M F S

with conjugate gradients, where ``S`` multiplies by the coil sensitivities, ``F`` is the
centred orthonormal FFT over the three spatial axes (SPE, PE, FE) and ``M`` applies the
undersampling mask. The operators are matrix-free: only ``E`` and ``E^H`` are ever applied.

Every routine works on arrays whose last three axes are (SPE, PE, FE) and whose leading
axes are free batch dimensions, so a full (Nv, Nt, ...) volume and a single frame use the
same code. Each batch element is an independent system, so the CG scalars are computed
per element rather than globally.
"""

from __future__ import annotations

import numpy as np
import scipy.fft as sfft

SPATIAL_AXES = (-3, -2, -1)
COIL_AXIS = -4


def _fftc(x, axes=SPATIAL_AXES):
    """Centred orthonormal forward FFT."""
    return sfft.fftshift(
        sfft.fftn(sfft.ifftshift(x, axes=axes), axes=axes, norm='ortho', workers=-1), axes=axes)


def _ifftc(x, axes=SPATIAL_AXES):
    """Centred orthonormal inverse FFT."""
    return sfft.fftshift(
        sfft.ifftn(sfft.ifftshift(x, axes=axes), axes=axes, norm='ortho', workers=-1), axes=axes)


def forward_op(x: np.ndarray, coilmap: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """E x: image (..., SPE, PE, FE) -> sampled multi-coil k-space (..., Nc, SPE, PE, FE)."""
    k = _fftc(np.expand_dims(x, COIL_AXIS) * coilmap)
    return k if mask is None else k * mask


def adjoint_op(k: np.ndarray, coilmap: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """E^H y: multi-coil k-space (..., Nc, SPE, PE, FE) -> coil-combined image (..., SPE, PE, FE).

    With ``mask`` applied to raw k-space this is exactly the zero-filled SENSE recon.
    """
    if mask is not None:
        k = k * mask
    return np.sum(_ifftc(k) * np.conj(coilmap), axis=COIL_AXIS)


def zero_filled(kdata: np.ndarray, coilmap: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Zero-filled reconstruction — one application of the adjoint. Alias of ``adjoint_op``."""
    return adjoint_op(kdata, coilmap, mask)


def normal_op(x: np.ndarray, coilmap: np.ndarray, mask: np.ndarray | None = None,
              lam: float = 0.0) -> np.ndarray:
    """(E^H E + lam I) x — the operator CG is run on."""
    out = adjoint_op(forward_op(x, coilmap, mask), coilmap, mask)
    return out + lam * x if lam else out


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-batch-element real inner product Re<a, b>, reduced over the spatial axes."""
    return np.real(np.sum(np.conj(a) * b, axis=SPATIAL_AXES, keepdims=True))


def cg_sense(kdata: np.ndarray, coilmap: np.ndarray, mask: np.ndarray | None = None,
             n_iter: int = 10, lam: float = 0.0, tol: float = 1e-5,
             x0: np.ndarray | None = None, verbose: bool = False,
             return_history: bool = False):
    """Reconstruct by conjugate gradients on the SENSE normal equations.

    Parameters
    ----------
    kdata : (..., Nc, SPE, PE, FE) complex
        Acquired k-space. Pass the fully-sampled array together with ``mask``, or an
        already-masked array with ``mask=None``.
    coilmap : (Nc, SPE, PE, FE) complex
        Coil sensitivities, broadcast against ``kdata``.
    mask : broadcastable, optional
        Sampling mask, e.g. the (1, Nt, 1, SPE, PE, 1) kt-Gaussian layout.
    n_iter : int
        Maximum CG iterations. Without ``lam`` this is the regularisation: CG-SENSE is
        semi-convergent, so too many iterations amplify noise.
    lam : float
        Tikhonov damping, relative to the scale of ``E^H E`` (try 1e-3 ... 1e-1).
    tol : float
        Stop once every batch element has relative residual below this.
    x0 : optional
        Initial guess; defaults to zeros (so the first residual is the zero-filled image).
    verbose : bool
        Print the max relative residual each iteration.
    return_history : bool
        Also return the list of per-iteration max relative residuals.

    Returns
    -------
    x : (..., SPE, PE, FE) complex64, and the residual history if requested.
    """
    b = adjoint_op(kdata, coilmap, mask).astype(np.complex64)

    if x0 is None:
        x = np.zeros_like(b)
        r = b.copy()
    else:
        x = x0.astype(np.complex64).copy()
        r = b - normal_op(x, coilmap, mask, lam)

    p = r.copy()
    rs = _dot(r, r)
    b_norm = np.sqrt(_dot(b, b))
    b_norm[b_norm == 0] = 1.0

    history = []
    eps = np.finfo(np.float32).tiny
    for it in range(n_iter):
        Ap = normal_op(p, coilmap, mask, lam)
        alpha = rs / np.maximum(_dot(p, Ap), eps)
        x += alpha * p
        r -= alpha * Ap
        rs_new = _dot(r, r)

        rel = float(np.max(np.sqrt(rs_new) / b_norm))
        history.append(rel)
        if verbose:
            print(f'  cg iter {it + 1:3d}/{n_iter}  max rel. residual {rel:.3e}', flush=True)
        if rel < tol:
            break

        p = r + (rs_new / np.maximum(rs, eps)) * p
        rs = rs_new

    return (x, history) if return_history else x


def adjoint_test(shape, coilmap: np.ndarray, mask: np.ndarray | None = None,
                 seed: int | None = 0) -> float:
    """Dot-product test <E x, y> == <x, E^H y>; returns the relative mismatch (should be ~1e-6).

    ``shape`` is the image shape (..., SPE, PE, FE).
    """
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    y_shape = shape[:-3] + coilmap.shape[-4:]
    y = (rng.standard_normal(y_shape) + 1j * rng.standard_normal(y_shape)).astype(np.complex64)

    lhs = np.vdot(forward_op(x, coilmap, mask), y)
    rhs = np.vdot(x, adjoint_op(y, coilmap, mask))
    return float(abs(lhs - rhs) / max(abs(lhs), 1e-30))
