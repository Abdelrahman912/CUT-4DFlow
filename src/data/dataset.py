"""CUT-4DFlow dataset.

Loads a patient volume, applies a kt-Gaussian undersampling at a chosen R, performs the SSDU
Theta/Lambda split (or the supervised split), builds coil-combined zero-filled images, and returns
per-slice samples. Supports per-scanner oversampling and optional per-scanner loss weighting.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import scipy.fft
import torch
from torch.utils.data import Dataset

from src.data.h5_reader import load_patient, read_params_csv
from src.data.ktgaussian import make_ktgaussian_mask
from src.data.ssdu_split import uniform_disjoint_selection


def _k2i_numpy(x: np.ndarray, ax: Sequence[int]) -> np.ndarray:
    """Centred orthonormal multi-dim inverse FFT."""
    return scipy.fft.fftshift(
        scipy.fft.ifftn(scipy.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho', workers=-1), axes=ax)


def _clone_for_ipc(x):
    """Detach numpy views from their (large) base buffer so DataLoader IPC stays cheap."""
    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x).copy()
    if torch.is_tensor(x):
        return x.contiguous().clone()
    return x


@lru_cache(maxsize=2)
def _load_patient_cached(patient_dir: str, canonicalise: bool = True) -> dict:
    """Small per-process LRU cache around ``load_patient``."""
    return load_patient(patient_dir, canonicalise=canonicalise)


@lru_cache(maxsize=2)
def _load_patient_cache_pt(cache_path: str) -> dict:
    """LRU cache around a precomputed .pt patient file (kdata already FE-iFFT'd)."""
    blob = torch.load(cache_path, map_location='cpu', weights_only=False)
    to_np = lambda a: a.numpy() if isinstance(a, torch.Tensor) else a  # noqa: E731
    return {
        'kdata_hybrid': to_np(blob['kdata_hybrid']),
        'coilmap': to_np(blob['sens']),
        'segmask': to_np(blob['segmask']),
        'params': blob['params'],
        'orientation': blob.get('orientation', 'canonical'),
    }


REQUIRED_FILES = ('kdata_full.mat', 'coilmap.mat', 'segmask.mat', 'params.csv')


def find_valid_patients(roots, required_files=REQUIRED_FILES, anchor='kdata_full.mat'):
    """Return every directory under ``roots`` that contains all ``required_files`` (sorted, unique)."""
    if anchor not in required_files:
        raise ValueError(f'anchor {anchor!r} must be in required_files')
    required = tuple(required_files)
    found: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        root = Path(r)
        if not root.exists():
            continue
        for kpath in root.rglob(anchor):
            case_dir = kpath.parent
            if all((case_dir / f).is_file() for f in required):
                key = str(case_dir)
                if key not in seen:
                    seen.add(key)
                    found.append(case_dir)
    return sorted(found)


def _seed_from_key(key: str) -> int:
    """Stable 32-bit seed from a string (reproducible across processes)."""
    return int.from_bytes(hashlib.md5(key.encode('utf-8')).digest()[:8], 'little', signed=False) % (2 ** 32)


class CMRx4DFlowDataset(Dataset):
    """4D-flow dataset. Train mode enumerates per-FE-slice entries; val/test process whole volumes."""

    def __init__(
        self,
        roots: Optional[Sequence[str | Path]] = None,
        mode: str = 'train',
        D_size: int = 1,
        T_size: int = 5,
        usrate_list: Sequence[int] = (10, 20, 30, 40, 50),
        rho: float = 0.2,
        ssdu_r2: int = 9,
        patient_dirs: Optional[Sequence[str | Path]] = None,
        seed_mode: str = 'random',
        cache_dir: Optional[str | Path] = None,
        orientation: str = 'canonical',
        supervised: bool = False,
        val_r_list: Optional[Sequence[int]] = None,
        masks_per_sample: int = 1,
        r_per_sample: int = 1,
        fixed_masks: bool = False,
        require_cache: bool = False,
        oversample_rule: Optional[Sequence[dict]] = None,
    ) -> None:
        super().__init__()
        if mode not in ('train', 'val', 'test'):
            raise ValueError(f'mode must be train/val/test; got {mode!r}')
        if D_size != -1 and D_size < 1:
            raise ValueError(f'D_size must be -1 or >= 1; got {D_size}')
        if T_size < 1:
            raise ValueError(f'T_size must be >= 1; got {T_size}')
        if seed_mode not in ('random', 'deterministic_by_path'):
            raise ValueError(f"seed_mode must be 'random' or 'deterministic_by_path'; got {seed_mode!r}")
        if orientation not in ('canonical', 'native'):
            raise ValueError(f"orientation must be 'canonical' or 'native'; got {orientation!r}")
        if roots is None and patient_dirs is None:
            raise ValueError('Either roots or patient_dirs must be provided.')

        self.roots = [Path(r) for r in (roots or [])]
        self.mode = mode
        self.orientation = orientation
        self.D_size = int(D_size)
        self.T_size = int(T_size)
        self.usrate_list = list(usrate_list)
        self.val_r_list = (list(val_r_list) if (val_r_list is not None and mode != 'train') else None)
        self.masks_per_sample = max(1, int(masks_per_sample))
        self.r_per_sample = max(1, int(r_per_sample))
        self.fixed_masks = bool(fixed_masks)
        self.rho = float(rho)
        self.ssdu_r2 = int(ssdu_r2)
        self.supervised = bool(supervised)
        self.seed_mode = seed_mode
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.require_cache = bool(require_cache)

        if patient_dirs is not None:
            self._patient_dirs_input: list[Path] = []
            required = set(REQUIRED_FILES)
            for p in patient_dirs:
                pp = Path(p)
                if pp.is_dir() and required.issubset({f.name for f in pp.iterdir()}):
                    self._patient_dirs_input.append(pp)
            patient_iter = sorted(self._patient_dirs_input)
        else:
            patient_iter = find_valid_patients(self.roots)

        # per-scanner oversampling (train only): first matching rule wins
        self.oversample_rule = [dict(r) for r in (oversample_rule or [])]
        _pat_per_scanner: Counter = Counter()
        for _p in patient_iter:
            _pat_per_scanner[(_p.parts[-3], _p.parts[-2])] += 1
        self._patients_per_scanner = dict(_pat_per_scanner)

        def _oversample_for(pdir: Path):
            if self.mode != 'train' or not self.oversample_rule:
                return 1, 'random'
            n = _pat_per_scanner[(pdir.parts[-3], pdir.parts[-2])]
            for rule in self.oversample_rule:
                if n <= int(rule['max_patients']):
                    return int(rule.get('reps', 1)), str(rule.get('mode', 'random'))
            return 1, 'random'

        # enumerate (patient_dir, fe_start, R) entries; R None (train) or fixed (val)
        self.filename: list[tuple[Path, int, Optional[int]]] = []
        for patient_dir in patient_iter:
            try:
                params = read_params_csv(patient_dir / 'params.csv')
                matrix_size = params.get('matrix_size')
                if matrix_size is None or len(matrix_size) != 6:
                    continue
                FE = int(matrix_size[5])
            except Exception:
                continue

            if self.mode == 'train':
                fe_starts = [0] if self.D_size == -1 else list(range(0, FE - self.D_size + 1))
            else:
                fe_starts = [0]

            _reps, _omode = _oversample_for(patient_dir)
            for s in fe_starts:
                if self.val_r_list is not None:
                    for R in self.val_r_list:
                        self.filename.append((patient_dir, int(s), int(R)))
                elif _omode == 'all_R':
                    for R in self.usrate_list:
                        self.filename.append((patient_dir, int(s), int(R)))
                else:
                    for _ in range(_reps):
                        self.filename.append((patient_dir, int(s), None))

        # per-scanner loss weights (mean weight 1.0; equal total mass per scanner)
        entries_per_scanner: Counter = Counter()
        for pdir, *_ in self.filename:
            entries_per_scanner[(pdir.parts[-3], pdir.parts[-2])] += 1
        self._entries_per_scanner = dict(entries_per_scanner)
        n_total = len(self.filename)
        n_scanners = len(entries_per_scanner) or 1
        self._scanner_loss_weights = {
            key: (n_total / n_scanners) / count for key, count in entries_per_scanner.items()
        }

        # patient -> dataset indices (for the block samplers)
        self.patient_to_entries: dict[str, list[int]] = {}
        for idx, (pdir, *_) in enumerate(self.filename):
            self.patient_to_entries.setdefault(str(pdir), []).append(idx)

    def __len__(self) -> int:
        return len(self.filename)

    def scanner_loss_weights(self):
        return dict(self._scanner_loss_weights)

    def entries_per_scanner(self):
        return dict(self._entries_per_scanner)

    def __getitem__(self, idx: int) -> dict:
        if self.mode != 'train' and self.seed_mode != 'deterministic_by_path':
            raise NotImplementedError(f'mode={self.mode!r} requires seed_mode=deterministic_by_path')

        patient_dir, fe_start, fixed_R = self.filename[idx]

        det_seed = None
        if self.seed_mode == 'deterministic_by_path':
            key = f'{patient_dir}|{fe_start}'
            if fixed_R is not None:
                key += f'|R{fixed_R}'
            det_seed = _seed_from_key(key)
            random.seed(det_seed)
            np.random.seed(det_seed)
        elif self.fixed_masks:
            _sseed = _seed_from_key(f'{patient_dir}|{fe_start}')
            random.seed(_sseed)
            np.random.seed(_sseed)

        cache_hit = False
        if self.cache_dir is not None:
            cache_path = self._resolve_cache_path(patient_dir)
            if cache_path is not None and cache_path.is_file():
                data = _load_patient_cache_pt(str(cache_path))
                if data.get('orientation', 'canonical') != self.orientation:
                    raise ValueError(f"cache orientation != dataset orientation ({cache_path})")
                kdata_hybrid = data['kdata_hybrid']
                coilmap = data['coilmap']
                segmask = data['segmask']
                params = data['params']
                cache_hit = True

        if not cache_hit:
            if self.cache_dir is not None and self.require_cache:
                raise FileNotFoundError(f'[require_cache] no .pt cache for {patient_dir}')
            data = _load_patient_cached(str(patient_dir), self.orientation == 'canonical')
            kdata = data['kdata']
            coilmap = data['coilmap']
            segmask = data['segmask']
            params = data['params']

        if cache_hit:
            Nv, Nt, Nc, SPE, PE, FE = kdata_hybrid.shape
        else:
            Nv, Nt, Nc, SPE, PE, FE = kdata.shape

        seg_idx = -1

        # random cardiac window with wrap-around
        if self.T_size >= Nt:
            cardiac_bins = np.arange(Nt)
            T_use = Nt
        else:
            first_bin = random.randint(-self.T_size + 1, Nt - self.T_size)
            cardiac_bins = np.mod(np.arange(first_bin, first_bin + self.T_size), Nt).astype(np.int64)
            T_use = self.T_size

        fe_sl = (slice(fe_start, fe_start + self.D_size) if self.D_size != -1 else slice(None))
        if cache_hit:
            f = np.take(kdata_hybrid[..., fe_sl], cardiac_bins, axis=1)
        else:
            f = np.take(kdata, cardiac_bins, axis=1)
            f = _k2i_numpy(f, ax=[-1])                 # readout iFFT: FE k-space -> image
            f = f[..., fe_sl]
        c = coilmap[..., fe_sl].astype(np.complex64)
        s = segmask[..., fe_sl]
        img_gt = np.sum(_k2i_numpy(f, ax=[-2, -3]) * np.conj(c), axis=-4)
        scanner_weight = self._scanner_loss_weights[(patient_dir.parts[-3], patient_dir.parts[-2])]
        B0 = np.float32(float(params.get('field_strength') or 3.0))
        VENC = np.asarray(params['VENC'], dtype=np.float32)

        def _realize(R: int, seed) -> dict:
            M_Omega = make_ktgaussian_mask(SPE=SPE, PE=PE, Nt=T_use, R=R, seed=seed)
            if self.supervised:
                M_Theta = M_Omega
                M_Lambda = (1.0 - M_Omega).astype(M_Omega.dtype)
            else:
                M_Theta, M_Lambda = uniform_disjoint_selection(M_Omega, rho=self.rho, r2=self.ssdu_r2, seed=seed)
            kdata_theta = (f * M_Theta).astype(np.complex64)
            kdata_lambda = (f * M_Lambda).astype(np.complex64)
            img_zf = np.sum(_k2i_numpy(kdata_theta, ax=[-2, -3]) * np.conj(c), axis=-4)
            denom = float(np.linalg.norm(np.abs(kdata_theta) != 0))
            norm = float(np.linalg.norm(kdata_theta)) / max(denom, 1.0) or 1.0
            out = {
                'kdata_theta': (kdata_theta / norm).astype(np.complex64),
                'kdata_lambda': (kdata_lambda / norm).astype(np.complex64),
                'mask_omega': M_Omega.astype(np.float32),
                'mask_theta': M_Theta.astype(np.float32),
                'mask_lambda': M_Lambda.astype(np.float32),
                'img_zf': (img_zf / norm).astype(np.complex64),
                'img_gt': (img_gt / norm).astype(np.complex64),
                'coilmap': c, 'segmask': s,
                'norm': np.float32(norm), 'R': np.int32(R), 'B0': B0,
                'seg_idx': np.int32(seg_idx),
                'cardiac_bins': cardiac_bins.astype(np.int64),
                'fe_start': np.int32(fe_start), 'patient_dir': str(patient_dir),
                'VENC': VENC, 'scanner_weight': np.float32(scanner_weight),
            }
            return {k: _clone_for_ipc(v) for k, v in out.items()}

        def _mseed(R, i):
            return _seed_from_key(f'{patient_dir}|{fe_start}|R{R}|m{i}') if self.fixed_masks else None

        if fixed_R is not None:
            return _realize(fixed_R, det_seed)
        if self.masks_per_sample * self.r_per_sample <= 1:
            R = random.choice(self.usrate_list)
            return _realize(R, _mseed(R, 0))
        m_use = min(self.r_per_sample, len(self.usrate_list))
        r_list = random.sample(self.usrate_list, m_use)
        return [_realize(R, _mseed(R, i)) for R in r_list for i in range(self.masks_per_sample)]

    def _resolve_cache_path(self, patient_dir: Path) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        parts = patient_dir.parts
        if len(parts) < 3:
            return None
        rel = Path(parts[-3]) / parts[-2] / f'{parts[-1]}.pt'
        return self.cache_dir / rel

    def patient_dirs(self) -> list[Path]:
        return sorted({pdir for pdir, *_ in self.filename})

    def patients_per_scanner(self):
        counts: dict[tuple[str, str], int] = {}
        for pdir in self.patient_dirs():
            counts[(pdir.parts[-3], pdir.parts[-2])] = counts.get((pdir.parts[-3], pdir.parts[-2]), 0) + 1
        return counts

    @classmethod
    def from_split_json(cls, split_path, root, key='train', **kwargs):
        """Build a dataset from a JSON split file mapping ``key`` to relative patient paths."""
        with open(split_path, 'r') as f:
            split = json.load(f)
        if key not in split:
            raise KeyError(f"split JSON {split_path!r} has no key {key!r}")
        if 'patient_dirs' in kwargs or 'roots' in kwargs:
            raise ValueError("from_split_json sets patient_dirs internally")
        patient_dirs = [Path(root) / rel for rel in split[key]]
        return cls(patient_dirs=patient_dirs, **kwargs)
