"""Per-patient loader for CMRx4DFlow.

Two layers:

A9 — I/O primitives (purely about reading bytes off disk):
  - ``h5_complex(path, key)`` reads a MATLAB v7.3 (HDF5) compound ``(real, imag)``
    dataset and returns numpy ``complex64``. Falls through for non-compound
    datasets (e.g. ``segmask.mat`` is plain ``uint8``).
  - ``read_params_csv(path)`` parses the per-patient ``params.csv`` into a typed
    dict.

A10 — composer:
  - ``load_patient(patient_dir)`` reads kdata, coilmap, segmask, params, then
    canonicalises the spatial orientation via ``canonicalise_spatial`` (A2).
    Returns a dict; deterministic; no randomness, no mask generation,
    no SSDU split. Those live in the Dataset class (A11-A14).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Peak extra RAM for the compound slab buffer in h5_complex (bytes).
_SLAB_BYTES = int(os.environ.get('CMRX_H5_SLAB_MB', '1024')) * 1024 * 1024

from src.data.orientation import canonicalise_spatial


# ─── A9.1 — h5 complex loader ─────────────────────────────────────────────
def h5_complex(path: str | Path, key: str) -> np.ndarray:
    """Load an HDF5 dataset, decoding compound ``(real, imag)`` to complex64.

    MATLAB v7.3 ``.mat`` files store complex values as a structured dtype with
    fields ``real`` and ``imag``. h5py exposes that structure as a numpy
    structured array; this function unwraps it to a real ``np.complex64``.

    For non-complex datasets (e.g. ``segmask.mat`` storing ``uint8``), the
    dataset is returned as-is.

    Parameters
    ----------
    path :
        Path to the ``.mat`` file.
    key :
        Top-level dataset name inside the file
        (e.g. ``'kdata_full'``, ``'coilmap'``, ``'segmask'``).

    Returns
    -------
    arr : np.ndarray
        Complex64 if the source was compound; otherwise the raw dtype.
    """
    with h5py.File(str(path), 'r') as f:
        ds = f[key]
        if ds.dtype.names == ('real', 'imag'):
            # Assemble complex64 in SLABS along the slowest axis. Two constraints
            # collide here and slabbing satisfies both:
            #   * peak RAM — `a['real'].astype(f32) + 1j*a['imag']...` held the whole
            #     2-field float64 structured array plus temporaries (~70 GB on the big
            #     S2 volumes, which OOM-killed the recon), and
            #   * decompression cost — reading `ds['real']` then `ds['imag']` asks h5py
            #     for one field at a time, and each pass INFLATES EVERY GZIP CHUNK of
            #     the dataset again (the compound fields are interleaved inside a
            #     chunk), so the whole file is decompressed TWICE. gzip inflate is the
            #     dominant per-case load cost (~92 MB/s), so that doubled ~15-19 s on
            #     an Aorta case and ~45-60 s on a Carotid/Cerebro volume.
            # Reading a slab as the COMPOUND dtype inflates each chunk exactly once and
            # keeps only that slab's float64 bytes alive: single-pass I/O, bounded peak.
            out = np.empty(ds.shape, dtype=np.complex64)
            n0 = ds.shape[0]
            row_bytes = max(int(np.prod(ds.shape[1:])) * ds.dtype.itemsize, 1)
            rows = int(max(1, min(n0, _SLAB_BYTES // row_bytes)))
            for i0 in range(0, n0, rows):
                i1 = min(i0 + rows, n0)
                blk = ds[i0:i1]            # ONE inflate of these chunks (compound)
                out.real[i0:i1] = blk['real']
                out.imag[i0:i1] = blk['imag']
                del blk
            return out
        return np.ascontiguousarray(ds[()])


# ─── A9.2 — params.csv parser ─────────────────────────────────────────────
def read_params_csv(path: str | Path) -> dict[str, Any]:
    """Parse the per-patient ``params.csv`` into a typed dict.

    Semicolon-delimited values become lists (of floats if all numeric, else of
    strings). Numeric scalars become ``float``; everything else stays ``str``.

    Returns a dict with keys like ``FA``, ``TE``, ``TR``, ``FOV``, ``matrix_size``,
    ``resolution``, ``VENC``, ``spatial_order``, ``venc_order``, ``system_model``,
    ``field_strength``.
    """
    with open(str(path)) as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError(f'{path} has fewer than 2 lines (header + values expected)')
    hdr, vals = rows[0], rows[1]
    out: dict[str, Any] = {}
    for k, v in zip(hdr, vals):
        if ';' in v:
            try:
                out[k] = [float(x) for x in v.split(';')]
            except ValueError:
                out[k] = v.split(';')                # direction tags etc.
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v                            # system_model, field_strength label
    return out


# ─── A10 — load + canonicalise one patient ────────────────────────────────
def load_patient(patient_dir: str | Path, canonicalise: bool = True) -> dict[str, Any]:
    """Load one patient's data and (optionally) canonicalise its spatial orientation.

    ``canonicalise=True`` (default, orientation='canonical'): flip to canonical
    (HF,AP,LR) via ``flip_origin``. ``canonicalise=False`` (orientation='native'):
    return the patient's NATIVE orientation untouched (for the canonical-vs-native
    training comparison).

    Reads the four files in ``patient_dir``:

      - ``kdata_full.mat``  -> complex64 (Nv, Nt, Nc, SPE, PE, FE)
      - ``coilmap.mat``     -> complex64 (Nc, SPE, PE, FE)
      - ``segmask.mat``     -> bool      (SPE, PE, FE)
      - ``params.csv``      -> dict

    Then applies ``canonicalise_spatial`` to the three array tensors so the
    downstream code sees canonical (HF, AP, LR) orientation regardless of the
    patient's native ``spatial_order``.

    This is the **deterministic** load step. All randomness (R, cardiac window,
    mask, SSDU split) lives in the Dataset class (A11-A14).

    Parameters
    ----------
    patient_dir :
        Path to a single patient directory containing the 4 files above.

    Returns
    -------
    dict with keys:
        - ``kdata``   (Nv, Nt, Nc, SPE, PE, FE) complex64, canonical
        - ``coilmap`` (Nc, SPE, PE, FE)         complex64, canonical
        - ``segmask`` (SPE, PE, FE)             bool,      canonical
        - ``params``  dict from ``params.csv``  (untouched)
    """
    p = Path(patient_dir)
    if not p.is_dir():
        raise FileNotFoundError(f'patient_dir does not exist: {p}')

    # A9 I/O
    kdata   = h5_complex(p / 'kdata_full.mat', 'kdata_full')
    coilmap = h5_complex(p / 'coilmap.mat',    'coilmap')
    segmask = h5_complex(p / 'segmask.mat',    'segmask').astype(bool)
    params  = read_params_csv(p / 'params.csv')

    # Validate shapes against params.csv before canonicalisation
    expected = tuple(int(x) for x in params['matrix_size'])           # (Nv,Nt,Nc,SPE,PE,FE)
    if kdata.shape != expected:
        # Most commonly: cardiac Nt off by 1 (organizer metadata vs actual file).
        # The file shape is authoritative; matrix_size is informational metadata.
        # Tolerate small mismatches (delta <= 2 per axis) and use the file shape.
        max_delta = max(abs(a - b) for a, b in zip(kdata.shape, expected))
        if max_delta > 2:
            raise ValueError(
                f'kdata.shape {kdata.shape} != matrix_size from params.csv {expected} '
                f'(max axis delta = {max_delta}). Refusing to silently load — too large to be metadata-only drift.'
            )
        import warnings
        warnings.warn(
            f'Patient kdata.shape {kdata.shape} differs from params.csv matrix_size '
            f'{expected} (max delta {max_delta}). Using file shape (authoritative).',
            UserWarning,
            stacklevel=2,
        )
    if coilmap.shape != expected[2:]:
        raise ValueError(
            f'coilmap.shape {coilmap.shape} != matrix_size[2:] {expected[2:]}'
        )
    if segmask.shape != expected[3:]:
        raise ValueError(
            f'segmask.shape {segmask.shape} != matrix_size[3:] {expected[3:]}'
        )

    # A2 — canonicalise spatial orientation on all three array tensors.
    # Skipped entirely for orientation='native' (canonicalise=False).
    so = params['spatial_order']
    if canonicalise:
        kdata   = canonicalise_spatial(kdata,   so)
        coilmap = canonicalise_spatial(coilmap, so)
        segmask = canonicalise_spatial(segmask, so)

    return {
        'kdata':   kdata,
        'coilmap': coilmap,
        'segmask': segmask,
        'params':  params,
        'orientation': 'canonical' if canonicalise else 'native',
    }
