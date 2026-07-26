"""Per-patient loader: reads k-space, coil maps, segmentation and parameters, and canonicalises."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from src.data.orientation import canonicalise_spatial


def h5_complex(path: str | Path, key: str) -> np.ndarray:
    """Load an HDF5 dataset, decoding a compound ``(real, imag)`` dtype to complex64."""
    with h5py.File(str(path), 'r') as f:
        ds = f[key]
        if ds.dtype.names == ('real', 'imag'):
            out = np.empty(ds.shape, dtype=np.complex64)
            out.real = ds['real']
            out.imag = ds['imag']
            return out
        return np.ascontiguousarray(ds[()])


def read_params_csv(path: str | Path) -> dict[str, Any]:
    """Parse the per-patient ``params.csv`` into a typed dict (lists for ';'-separated values)."""
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
                out[k] = v.split(';')
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def load_patient(patient_dir: str | Path, canonicalise: bool = True) -> dict[str, Any]:
    """Load one patient (kdata, coilmap, segmask, params) and optionally canonicalise orientation.

    Returns a dict with keys ``kdata`` (Nv,Nt,Nc,SPE,PE,FE), ``coilmap`` (Nc,SPE,PE,FE),
    ``segmask`` (SPE,PE,FE) bool, ``params`` dict, and ``orientation``.
    """
    p = Path(patient_dir)
    if not p.is_dir():
        raise FileNotFoundError(f'patient_dir does not exist: {p}')

    kdata = h5_complex(p / 'kdata_full.mat', 'kdata_full')
    coilmap = h5_complex(p / 'coilmap.mat', 'coilmap')
    segmask = h5_complex(p / 'segmask.mat', 'segmask').astype(bool)
    params = read_params_csv(p / 'params.csv')

    expected = tuple(int(x) for x in params['matrix_size'])
    if kdata.shape != expected:
        max_delta = max(abs(a - b) for a, b in zip(kdata.shape, expected))
        if max_delta > 2:
            raise ValueError(f'kdata.shape {kdata.shape} != matrix_size {expected} (delta {max_delta})')
        import warnings
        warnings.warn(f'kdata.shape {kdata.shape} differs from matrix_size {expected}; using file shape.',
                      UserWarning, stacklevel=2)
    if coilmap.shape != expected[2:]:
        raise ValueError(f'coilmap.shape {coilmap.shape} != {expected[2:]}')
    if segmask.shape != expected[3:]:
        raise ValueError(f'segmask.shape {segmask.shape} != {expected[3:]}')

    so = params['spatial_order']
    if canonicalise:
        kdata = canonicalise_spatial(kdata, so)
        coilmap = canonicalise_spatial(coilmap, so)
        segmask = canonicalise_spatial(segmask, so)

    return {'kdata': kdata, 'coilmap': coilmap, 'segmask': segmask, 'params': params,
            'orientation': 'canonical' if canonicalise else 'native'}
