"""Pre-compute per-patient cache for the CMRx training pipeline.

For each patient directory under ``--src``, this script:

1. Loads the patient via ``src.data.h5_reader.load_patient`` (which already
   applies ``canonicalise_spatial`` to kdata / coilmap / segmask).
2. Performs the FE-axis (last spatial axis) inverse FFT on ``kdata`` with
   centred-orthonormal convention, producing the ``kdata_hybrid`` tensor used
   inside ``CMRx4DFlowDataset.__getitem__`` (after that step the dataset only
   does FE slicing and 2D IFFT over SPE/PE — cheap relative to the disk load
   + canonicalise + FE IFFT).
3. Saves ``{kdata_hybrid, sens, segmask, params}`` to ``<dst>/<rel>.pt``.

At runtime the dataset can ``torch.load`` instead of re-loading the .mat files
and re-doing the FE IFFT every step.

Usage
-----
    python src/data/precompute_patient_cache.py
    python src/data/precompute_patient_cache.py --overwrite
    python src/data/precompute_patient_cache.py --src /path/to/Aorta --dst /tmp/cache
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import scipy.fft
import torch

from src.data.dataset import find_valid_patients
from src.data.h5_reader import load_patient


DEFAULT_SRC = 'Data/TaskR1R2/TrainSet/Aorta'
DEFAULT_DST = 'outputs/patient_cache'


def _fe_ifft_numpy(kdata: np.ndarray) -> np.ndarray:
    """Centred orthonormal IFFT along the LAST spatial axis (FE).

    ``kdata`` has shape (Nv, Nt, Nc, SPE, PE, FE); the FE axis is axis -1.
    Matches the recipe in ``src.data.dataset._k2i_numpy``.
    """
    ax = (-1,)
    return scipy.fft.fftshift(
        scipy.fft.ifftn(
            scipy.fft.ifftshift(kdata, axes=ax),
            axes=ax, norm='ortho', workers=-1,
        ),
        axes=ax,
    )


def _patient_rel_path(patient_dir: Path, src_root: Path) -> Path:
    """Return ``centre/vendor/PXXX`` relative to ``src_root``."""
    return patient_dir.relative_to(src_root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Pre-compute per-patient .pt cache for CMRx training.',
    )
    parser.add_argument('--src', type=str, default=DEFAULT_SRC,
                        help='Root containing centre/vendor/PXXX patient dirs.')
    parser.add_argument('--dst', type=str, default=DEFAULT_DST,
                        help='Destination cache root.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-cache patients even if the .pt file exists.')
    parser.add_argument('--orientation', choices=['canonical', 'native'], default='canonical',
                        help="'canonical' (flip_origin to HF,AP,LR) or 'native' (no flip). Stamped "
                             "into each .pt; the dataset refuses a cache whose tag != its orientation. "
                             "Use a DIFFERENT --dst for native so the two caches don't collide.")
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    if not src_root.is_dir():
        raise SystemExit(f'[precompute] --src does not exist: {src_root}')

    print(f'[precompute] src: {src_root}', flush=True)
    print(f'[precompute] dst: {dst_root}', flush=True)
    print(f'[precompute] orientation: {args.orientation}', flush=True)
    print('[precompute] scanning for valid patients ...', flush=True)

    t_scan = time.time()
    patient_dirs = find_valid_patients([src_root])
    print(f'[precompute] {len(patient_dirs)} valid patients found '
          f'({time.time() - t_scan:.1f}s)', flush=True)
    if not patient_dirs:
        raise SystemExit('[precompute] no valid patients under --src; check path/structure')

    n_done = 0
    n_skipped = 0
    total_bytes = 0
    t_total_start = time.time()

    for i, pdir in enumerate(patient_dirs):
        rel = _patient_rel_path(pdir, src_root)
        out_path = dst_root / rel.with_suffix('.pt')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            sz = out_path.stat().st_size
            total_bytes += sz
            n_skipped += 1
            print(
                f'  [{i + 1:4d}/{len(patient_dirs)}] skip {rel} '
                f'({sz / 1e6:.1f} MB existing)', flush=True
            )
            continue

        print(f'  [{i + 1:4d}/{len(patient_dirs)}] {rel} loading ...', flush=True)
        t0 = time.time()
        data = load_patient(str(pdir), canonicalise=(args.orientation == 'canonical'))
        kdata_np = data['kdata']
        sens_np = data['coilmap']
        seg_np = data['segmask']
        params = data['params']
        t_load = time.time() - t0

        # Centred orthonormal IFFT along FE (last spatial axis).
        t1 = time.time()
        kdata_hybrid_np = _fe_ifft_numpy(kdata_np).astype(np.complex64)
        sens_np = sens_np.astype(np.complex64)
        seg_np = seg_np.astype(bool)
        t_ifft = time.time() - t1

        kdata_hybrid = torch.from_numpy(kdata_hybrid_np)
        sens = torch.from_numpy(sens_np)
        segmask = torch.from_numpy(seg_np)

        t2 = time.time()
        torch.save(
            {
                'kdata_hybrid': kdata_hybrid,
                'sens': sens,
                'segmask': segmask,
                'params': params,
                'orientation': args.orientation,
            },
            out_path,
        )
        t_save = time.time() - t2

        sz = out_path.stat().st_size
        total_bytes += sz
        n_done += 1
        print(
            f'  [{i + 1:4d}/{len(patient_dirs)}] {rel} '
            f'kdata={tuple(kdata_hybrid.shape)} {sz / 1e6:.1f} MB | '
            f'load {t_load:.1f}s ifft {t_ifft:.1f}s save {t_save:.1f}s', flush=True
        )

    t_total = time.time() - t_total_start
    print('[precompute] summary:', flush=True)
    print(f'  cached this run: {n_done}', flush=True)
    print(f'  skipped existing: {n_skipped}', flush=True)
    print(f'  total disk usage: {total_bytes / 1e9:.2f} GB', flush=True)
    print(f'  total time:       {t_total:.1f} s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
