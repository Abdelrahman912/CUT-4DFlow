"""CMRx4DFlowDataset — PyTorch Dataset for the CMRx4DFlow2026 challenge.

Built in 4 sub-steps:

  A11 — ``__init__`` + ``__len__``
      Walk the training roots, find every patient with the 4 required files,
      enumerate (patient_dir, fe_slice_start) tuples.

  A12 — ``__getitem__`` skeleton (random R + kt-Gaussian mask generation)
  A13 — SSDU partitioning (theta, lambda) + random venc + cardiac window
        with wrap-around
  A14 — **inverse-frequency loss weighting per scanner.** Each sample in
        ``__getitem__`` carries a ``scanner_weight`` scalar (the training
        loop multiplies its loss by this). Scanners with fewer patients get
        higher weight per sample, equalising the per-epoch gradient
        contribution across scanners — without dropping any sample or
        oversampling minority patients.
"""

from __future__ import annotations

import hashlib
import json
import os
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


# ─── private helpers ──────────────────────────────────────────────────────
# scipy.fft thread pool per call. Default -1 (all cores) for single-process use
# (recon/tests); the training SLURM sets CMRX_FFT_WORKERS=1 so many dataloader
# workers on few cores don't oversubscribe (CPU-minute billing = cpus-per-task).
_FFT_WORKERS = int(os.environ.get('CMRX_FFT_WORKERS', '-1'))


def _k2i_numpy(x: np.ndarray, ax: Sequence[int]) -> np.ndarray:
    """Centred orthonormal multi-dim IFFT — same recipe as
    ``src.data.ktgaussian.create_gaussian_weight_matrix``'s ecosystem."""
    return scipy.fft.fftshift(
        scipy.fft.ifftn(
            scipy.fft.ifftshift(x, axes=ax),
            axes=ax, norm='ortho', workers=_FFT_WORKERS,
        ),
        axes=ax,
    )


def _clone_for_ipc(x):
    """Return a fresh, contiguous copy suitable for DataLoader IPC.

    Without this, every numpy view returned by ``__getitem__`` carries the
    full base buffer through pickle — for our 1.6 GB cached patient tensor
    that's catastrophic when ``num_workers > 0``. Cloning detaches the
    returned arrays from the cache so each sample is just tens of MB.

    Scalars (np.float32, int, str, etc.) and dicts pass through untouched.
    """
    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x).copy()
    if torch.is_tensor(x):
        return x.contiguous().clone()
    return x


@lru_cache(maxsize=2)
def _load_patient_cached(patient_dir: str, canonicalise: bool = True) -> dict:
    """Tiny in-process LRU cache around ``load_patient``.

    Each PyTorch DataLoader worker gets its own cache. Each cached
    patient is ~2 GB (full kdata + coilmap + segmask). With multiple
    workers and the precomputed-cache path also memory-resident, we keep
    ``maxsize`` small to avoid blowing RAM. ``canonicalise`` is part of the
    cache key so canonical and native loads never collide.
    """
    return load_patient(patient_dir, canonicalise=canonicalise)


@lru_cache(maxsize=2)
def _load_patient_cache_pt(cache_path: str) -> dict:
    """In-process LRU cache around the precomputed ``.pt`` patient file.

    Returns a dict mirroring ``load_patient`` (but with ``kdata_hybrid``
    already FE-IFFT'd) so the downstream ``__getitem__`` can branch on a
    single ``data`` dict.
    """
    blob = torch.load(cache_path, map_location='cpu', weights_only=False)
    return {
        'kdata_hybrid': blob['kdata_hybrid'].numpy()
            if isinstance(blob['kdata_hybrid'], torch.Tensor)
            else blob['kdata_hybrid'],
        'coilmap': blob['sens'].numpy()
            if isinstance(blob['sens'], torch.Tensor)
            else blob['sens'],
        'segmask': blob['segmask'].numpy()
            if isinstance(blob['segmask'], torch.Tensor)
            else blob['segmask'],
        'params': blob['params'],
        'orientation': blob.get('orientation', 'canonical'),  # tag; old caches default canonical
    }


# Files every patient directory must contain to be considered valid.
REQUIRED_FILES = ('kdata_full.mat', 'coilmap.mat', 'segmask.mat', 'params.csv')


def find_valid_patients(
    roots: Sequence[str | Path],
    required_files: Iterable[str] = REQUIRED_FILES,
    anchor: str = 'kdata_full.mat',
) -> list[Path]:
    """Walk ``roots`` recursively and return every patient directory that
    contains all of ``required_files``.

    Uses ``anchor`` as the rglob trigger (any required file works; using
    ``kdata_full.mat`` matches FlowVN). Returns a deduplicated, sorted list.

    Parameters
    ----------
    roots :
        One or more directories. ``ValidationSet/Aorta/`` or
        ``TrainSet/Aorta/`` typically.
    required_files :
        Filenames that must exist for the patient directory to be included.
    anchor :
        Filename used to find candidate patient dirs (must be in
        ``required_files``).

    Returns
    -------
    list[Path]  — patient directories, sorted, no duplicates.
    """
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
    """Stable 32-bit seed from a string key (md5 -> int).

    Reproducible across processes/machines/runs (unlike Python's salted hash()).
    Used to make masks/splits a deterministic FUNCTION of (slice identity + realization)
    so the training objective is stationary (``fixed_masks``).
    """
    return int.from_bytes(
        hashlib.md5(key.encode('utf-8')).digest()[:8], 'little', signed=False
    ) % (2 ** 32)


class CMRx4DFlowDataset(Dataset):
    """CMRx4DFlow PyTorch dataset.

    A11 covers only ``__init__`` and ``__len__``. The ``__getitem__`` lands
    in A12-A13. Calling ``__getitem__`` here raises ``NotImplementedError``.

    Parameters
    ----------
    roots :
        Sequence of root paths (e.g. ``[Path('data/.../TrainSet/Aorta')]``).
    mode :
        One of ``'train'``, ``'val'``, ``'test'``. Affects how many samples
        per patient are enumerated:
          - ``'train'`` — one entry per FE slice start (D_size positions).
          - ``'val'`` / ``'test'`` — one entry per patient (whole volume).
    D_size :
        FE-slice depth, in pixels. ``D_size=1`` (default) enumerates each
        FE position separately, FlowVN-style. ``D_size=-1`` collapses to
        one entry per patient (whole volume mode).
    T_size :
        Cardiac window length used by ``__getitem__`` (A12-A13). Not used
        by ``__init__`` / ``__len__`` directly — stored so the Dataset has
        the value for later.
    usrate_list :
        Pool of R values to sample from in train mode (A12). Stored here so
        all configuration lives on the class. Not used in __init__.
    """

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
            raise ValueError(
                f"seed_mode must be 'random' or 'deterministic_by_path'; "
                f"got {seed_mode!r}"
            )
        if orientation not in ('canonical', 'native'):
            raise ValueError(
                f"orientation must be 'canonical' or 'native'; got {orientation!r}"
            )
        if roots is None and patient_dirs is None:
            raise ValueError(
                'Either roots or patient_dirs must be provided.'
            )

        self.roots = [Path(r) for r in (roots or [])]
        self.mode = mode
        self.orientation = orientation
        self.D_size = int(D_size)
        self.T_size = int(T_size)
        self.usrate_list = list(usrate_list)
        # Val (patient x R): if set (val/test only), the entry list holds one
        # entry per (patient, R) with a FIXED R + a deterministic mask per R,
        # so every val patient is evaluated at every R. None -> R drawn per
        # sample from usrate_list (training behaviour).
        self.val_r_list: Optional[list[int]] = (
            list(val_r_list) if (val_r_list is not None and mode != 'train') else None
        )
        # Multi-realization per training sample (variance reduction): each drawn slice
        # yields masks_per_sample (N) fresh Omega-masks x r_per_sample (M) R-values =
        # N*M realizations (train only). 1x1 = today's behaviour.
        self.masks_per_sample = max(1, int(masks_per_sample))
        self.r_per_sample = max(1, int(r_per_sample))
        # fixed_masks: make the train realizations DETERMINISTIC per slice (cardiac window,
        # R selection, and each mask/split seeded by slice-id + realization index) so the
        # SAME N*M targets recur every epoch -> stationary objective. val is already fixed.
        self.fixed_masks = bool(fixed_masks)
        self.rho = float(rho)
        self.ssdu_r2 = int(ssdu_r2)
        self.supervised = bool(supervised)
        self.seed_mode = seed_mode
        self.cache_dir: Optional[Path] = (
            Path(cache_dir) if cache_dir is not None else None
        )
        # require_cache: with a cache_dir set, REFUSE the silent .mat fallback when a
        # patient's .pt is missing — raise instead, so an incomplete cache surfaces
        # immediately rather than as a ~30x per-slice slowdown.
        self.require_cache = bool(require_cache)

        # Either walk roots OR use the supplied patient list directly.
        if patient_dirs is not None:
            # Trust the caller: validate that each path is an existing
            # directory with the required files. Skip silently if not.
            self._patient_dirs_input: list[Path] = []
            required = set(REQUIRED_FILES)
            for p in patient_dirs:
                pp = Path(p)
                if pp.is_dir() and required.issubset({f.name for f in pp.iterdir()}):
                    self._patient_dirs_input.append(pp)
            patient_iter: list[Path] = sorted(self._patient_dirs_input)
        else:
            patient_iter = find_valid_patients(self.roots)

        # ── Per-scanner oversampling (TRAIN only) ────────────────────────
        # Rebalances scanners that are badly under-represented. ``oversample_rule``
        # is a list of {max_patients, reps, mode} evaluated in order; first match
        # wins. Counted per SCANNER (centre/vendor), not per centre — Center012
        # holds two scanners and its 3T Ingenia (4 patients) is the scarcest in
        # the whole set, which a centre-level rule would miss entirely.
        #   mode 'all_R'  -> one entry per R in usrate_list. Fixed R, but the mask
        #                    is still drawn fresh every epoch (seed_mode='random'
        #                    leaves det_seed=None), so each repeat is a genuinely
        #                    different undersampling, not a duplicate.
        #   mode 'random' -> ``reps`` entries with R=None: R *and* mask drawn per
        #                    __getitem__.
        self.oversample_rule: list[dict] = [dict(r) for r in (oversample_rule or [])]
        _pat_per_scanner: Counter = Counter()
        for _p in patient_iter:
            _pat_per_scanner[(_p.parts[-3], _p.parts[-2])] += 1
        self._patients_per_scanner: dict[tuple[str, str], int] = dict(_pat_per_scanner)

        def _oversample_for(pdir: Path) -> tuple[int, str]:
            if self.mode != 'train' or not self.oversample_rule:
                return 1, 'random'
            n = _pat_per_scanner[(pdir.parts[-3], pdir.parts[-2])]
            for rule in self.oversample_rule:
                if n <= int(rule['max_patients']):
                    return int(rule.get('reps', 1)), str(rule.get('mode', 'random'))
            return 1, 'random'

        # Walk patients and enumerate (patient_dir, fe_start, R) entries.
        # R is None (train: drawn per sample) or a fixed int (val: patient x R).
        self.filename: list[tuple[Path, int, Optional[int]]] = []
        for patient_dir in patient_iter:
            # Read FE from params.csv (cheap — no kdata load).
            try:
                params = read_params_csv(patient_dir / 'params.csv')
                matrix_size = params.get('matrix_size')
                if matrix_size is None or len(matrix_size) != 6:
                    continue
                FE = int(matrix_size[5])
            except Exception:
                continue

            if self.mode == 'train':
                if self.D_size == -1:
                    fe_starts = [0]
                else:
                    fe_starts = list(range(0, FE - self.D_size + 1))
            else:
                # val / test — process the whole volume (one entry per patient).
                fe_starts = [0]

            _reps, _omode = _oversample_for(patient_dir)
            for s in fe_starts:
                if self.val_r_list is not None:
                    for R in self.val_r_list:      # val: one entry per (patient, R)
                        self.filename.append((patient_dir, int(s), int(R)))
                elif _omode == 'all_R':            # scarce scanner: cover every R
                    for R in self.usrate_list:
                        self.filename.append((patient_dir, int(s), int(R)))
                else:                              # `reps` draws, R+mask random each
                    for _ in range(_reps):
                        self.filename.append((patient_dir, int(s), None))

        # ── A14 — inverse-frequency scanner loss weights ──
        # Count entries per (centre, vendor) and compute weights that:
        #   1) Average to 1.0 across the dataset (so loss magnitude stays
        #      comparable to unweighted training — keeps LR calibration easy).
        #   2) Equalise the EXPECTED total loss contribution per scanner per
        #      epoch (so scanner with fewer patients gets boosted gradient).
        entries_per_scanner: Counter = Counter()
        for pdir, *_ in self.filename:
            entries_per_scanner[(pdir.parts[-3], pdir.parts[-2])] += 1
        self._entries_per_scanner: dict[tuple[str, str], int] = dict(entries_per_scanner)

        # weight = (n_total_entries / n_scanners) / entries_for_that_scanner
        # -> sum over all samples = n_total_entries  -> average weight = 1.0
        # -> sum per scanner = n_total_entries / n_scanners (equal mass per scanner)
        n_total = len(self.filename)
        n_scanners = len(entries_per_scanner) or 1
        self._scanner_loss_weights: dict[tuple[str, str], float] = {
            key: (n_total / n_scanners) / count
            for key, count in entries_per_scanner.items()
        }

        # ── Map patient_dir -> list of dataset indices ────────────────────
        # Used by ``src.data.samplers.BlockShuffleSampler`` to yield indices
        # patient-by-patient, which turns the dataset's ``@lru_cache(maxsize=2)``
        # into a near-perfect cache (one ``torch.load`` per patient per epoch
        # instead of one per FE-slice entry).
        self.patient_to_entries: dict[str, list[int]] = {}
        for idx, (pdir, *_) in enumerate(self.filename):
            self.patient_to_entries.setdefault(str(pdir), []).append(idx)

    def __len__(self) -> int:
        return len(self.filename)

    def scanner_loss_weights(self) -> dict[tuple[str, str], float]:
        """Per-scanner loss weight, computed once in ``__init__``.

        The training loop should multiply each sample's loss by the weight
        looked up from this dict for that sample's scanner. The convention:

        - average weight across the dataset = ``1.0`` (so loss magnitude
          and learning-rate calibration stay comparable to unweighted training);
        - **each scanner's TOTAL loss mass per epoch is equal** (so a 5-patient
          minority scanner gets the same gradient contribution as a 75-patient
          majority scanner, without dropping any majority sample or
          oversampling any minority sample).

        Returns
        -------
        dict mapping ``(centre, vendor)`` -> float weight.
        """
        return dict(self._scanner_loss_weights)

    def entries_per_scanner(self) -> dict[tuple[str, str], int]:
        """Count of dataset entries per ``(centre, vendor)`` tuple.

        Differs from ``patients_per_scanner`` when ``D_size != -1`` because
        each patient contributes multiple entries (one per FE slice).
        """
        return dict(self._entries_per_scanner)

    def __getitem__(self, idx: int) -> dict:
        """Return one training/val/test sample.

        Implements A12 (random R + kt-Gaussian mask) and A13 (SSDU split +
        random venc + cardiac window with wrap). Currently train-mode only;
        val/test code paths land later when we wire up the training loop.
        """
        # val/test modes share the train-mode item layout when seeded
        # deterministically. The kdata_full path requires a real volume; if
        # the dataset was built off val patients with only ktGaussian files,
        # find_valid_patients will have skipped them and ``filename`` is empty.
        if (
            self.mode != 'train'
            and self.seed_mode != 'deterministic_by_path'
        ):
            raise NotImplementedError(
                f'mode={self.mode!r} __getitem__ TBD (train mode only for now)'
            )

        patient_dir, fe_start, fixed_R = self.filename[idx]

        # ── Deterministic seed by patient path (val) — same R, mask, venc,
        #    cardiac window, SSDU partition every time the same patient is
        #    drawn. Keeps validation reproducible across runs/workers.
        det_seed: int | None = None
        if self.seed_mode == 'deterministic_by_path':
            key = f'{patient_dir}|{fe_start}'
            if fixed_R is not None:          # val (patient x R): mask varies per R
                key += f'|R{fixed_R}'
            det_seed = _seed_from_key(key)
            random.seed(det_seed)
            np.random.seed(det_seed)
        elif self.fixed_masks:               # train + fixed masks: deterministic cardiac
            # window + R selection per slice (each mask is seeded per-realization below)
            _sseed = _seed_from_key(f'{patient_dir}|{fe_start}')
            random.seed(_sseed)
            np.random.seed(_sseed)

        # ── Load patient (canonical OR native per self.orientation, cached) ──
        # If a pre-computed cache file exists, load the hybrid-space tensor
        # directly (skips the .mat load + canonicalise + FE-IFFT).
        cache_hit = False
        if self.cache_dir is not None:
            cache_path = self._resolve_cache_path(patient_dir)
            if cache_path is not None and cache_path.is_file():
                data = _load_patient_cache_pt(str(cache_path))
                # Guard: a canonical cache must never be reused for a native run (or vice-versa).
                cache_or = data.get('orientation', 'canonical')
                if cache_or != self.orientation:
                    raise ValueError(
                        f"patient-cache orientation '{cache_or}' != dataset orientation "
                        f"'{self.orientation}' ({cache_path}). Regenerate with "
                        f"src/data/precompute_patient_cache.py --orientation {self.orientation}."
                    )
                kdata_hybrid = data['kdata_hybrid']   # (Nv, Nt, Nc, SPE, PE, FE)
                coilmap = data['coilmap']             # (Nc, SPE, PE, FE)
                segmask = data['segmask']             # (SPE, PE, FE) bool
                params = data['params']
                cache_hit = True

        if not cache_hit:
            if self.cache_dir is not None and self.require_cache:
                raise FileNotFoundError(
                    f'[require_cache] no .pt cache for {patient_dir} '
                    f'(expected {self._resolve_cache_path(patient_dir)}). The cache is '
                    f'incomplete — rebuild it before training instead of falling back to '
                    f'slow .mat reads.'
                )
            data = _load_patient_cached(str(patient_dir), self.orientation == 'canonical')
            kdata   = data['kdata']      # (Nv, Nt, Nc, SPE, PE, FE) complex64
            coilmap = data['coilmap']    # (Nc, SPE, PE, FE)          complex64
            segmask = data['segmask']    # (SPE, PE, FE)              bool
            params  = data['params']

        if cache_hit:
            Nv, Nt, Nc, SPE, PE, FE = kdata_hybrid.shape
        else:
            Nv, Nt, Nc, SPE, PE, FE = kdata.shape

        # ── A13.1 — keep ALL velocity encodings ──
        # Arch-A / flowmri tradition: the model jointly embeds all Nv=4
        # velocity components in its input layer (in_channels=Nv). We
        # therefore keep V intact instead of randomly picking one per
        # __getitem__. The ``seg_idx`` field is kept for backward
        # compatibility with downstream code that still references it
        # (set to -1 to indicate "all velocities").
        seg_idx = -1

        # ── A13.2 — random cardiac window with wrap-around ──
        if self.T_size >= Nt:
            cardiac_bins = np.arange(Nt)            # full cycle, no wrap needed
            T_use = Nt
        else:
            first_bin = random.randint(-self.T_size + 1, Nt - self.T_size)
            cardiac_bins = np.mod(
                np.arange(first_bin, first_bin + self.T_size), Nt
            ).astype(np.int64)
            T_use = self.T_size

        # ── Slice k-space to (Nv, T=cardiac_bins, Nc, SPE, PE, FE), FE-slice, and the
        #    fully-sampled reference — SHARED across all mask/R realizations of this slice ──
        fe_sl = (slice(fe_start, fe_start + self.D_size)
                 if self.D_size != -1 else slice(None))
        if cache_hit:
            # FE-slice BEFORE the cardiac gather. The slice is a basic-index view,
            # so np.take then copies only the FE positions we keep instead of the
            # whole readout extent — for D_size=1 that is ~100x less memory traffic
            # (a 580 MB temporary vs 5 MB) for a bit-identical result. Valid only
            # here: the cached tensor is already FE-IFFT'd, so FE-slicing commutes.
            f = np.take(kdata_hybrid[..., fe_sl], cardiac_bins, axis=1)
        else:
            # .mat path: the readout IFFT below needs the full FE axis, so the
            # slice has to wait until after it.
            f = np.take(kdata, cardiac_bins, axis=1)          # (Nv,T,Nc,SPE,PE,FE) k-space
            f = _k2i_numpy(f, ax=[-1])                        # IFFT along FE (readout -> image)
            f = f[..., fe_sl]
        c = coilmap[..., fe_sl].astype(np.complex64)
        s = segmask[..., fe_sl]
        img_gt = np.sum(_k2i_numpy(f, ax=[-2, -3]) * np.conj(c), axis=-4)   # (Nv,T,SPE,PE,D)
        scanner_weight = self._scanner_loss_weights[(patient_dir.parts[-3], patient_dir.parts[-2])]
        B0 = np.float32(float(params.get('field_strength') or 3.0))
        VENC = np.asarray(params['VENC'], dtype=np.float32)

        def _realize(R: int, seed) -> dict:
            """One (R, fresh Omega-mask) realization -> IPC-cloned sample dict.

            det_seed is None in train mode -> global RNG -> a DIFFERENT mask/split each
            call; set (val/test) -> deterministic. Reuses the shared f/c/s/img_gt.
            """
            M_Omega = make_ktgaussian_mask(SPE=SPE, PE=PE, Nt=T_use, R=R, seed=seed)
            if self.supervised:
                # input = whole acquired Omega; loss on the UN-acquired complement vs GT.
                M_Theta = M_Omega
                M_Lambda = (1.0 - M_Omega).astype(M_Omega.dtype)
            else:
                # SSDU: split acquired Omega into disjoint Theta (input) + Lambda (held-out).
                M_Theta, M_Lambda = uniform_disjoint_selection(
                    M_Omega, rho=self.rho, r2=self.ssdu_r2, seed=seed,
                )
            kdata_theta = (f * M_Theta).astype(np.complex64)
            kdata_lambda = (f * M_Lambda).astype(np.complex64)
            img_zf = np.sum(_k2i_numpy(kdata_theta, ax=[-2, -3]) * np.conj(c), axis=-4)
            denom = float(np.linalg.norm(np.abs(kdata_theta) != 0))
            norm = float(np.linalg.norm(kdata_theta)) / max(denom, 1.0) or 1.0
            out = {
                'kdata_theta':  (kdata_theta / norm).astype(np.complex64),
                'kdata_lambda': (kdata_lambda / norm).astype(np.complex64),
                'mask_omega':   M_Omega.astype(np.float32),
                'mask_theta':   M_Theta.astype(np.float32),
                'mask_lambda':  M_Lambda.astype(np.float32),
                'img_zf':       (img_zf / norm).astype(np.complex64),
                'img_gt':       (img_gt / norm).astype(np.complex64),
                'coilmap':      c,
                'segmask':      s,
                'norm':         np.float32(norm),
                'R':            np.int32(R),
                'B0':           B0,
                'seg_idx':      np.int32(seg_idx),
                'cardiac_bins': cardiac_bins.astype(np.int64),
                'fe_start':     np.int32(fe_start),
                'patient_dir':  str(patient_dir),
                'VENC':         VENC,
                'scanner_weight': np.float32(scanner_weight),
            }
            # IPC clone: detach from the ~1.6 GB LRU-cached patient blob.
            return {k: _clone_for_ipc(v) for k, v in out.items()}

        # ── Realizations: val/test = one deterministic; train = N masks x M R-values ──
        def _mseed(R, i):   # per-realization mask seed: fixed (deterministic) or None (fresh)
            return _seed_from_key(f'{patient_dir}|{fe_start}|R{R}|m{i}') if self.fixed_masks else None
        if fixed_R is not None:                       # val/test entry (patient x R)
            return _realize(fixed_R, det_seed)
        if self.masks_per_sample * self.r_per_sample <= 1:
            R = random.choice(self.usrate_list)
            return _realize(R, _mseed(R, 0))          # default: 1 realization
        m_use = min(self.r_per_sample, len(self.usrate_list))
        r_list = random.sample(self.usrate_list, m_use)        # M distinct R's (fixed per slice if fixed_masks)
        return [_realize(R, _mseed(R, i))
                for R in r_list for i in range(self.masks_per_sample)]

    # ─── Cache helpers ──────────────────────────────────────────────────
    def _resolve_cache_path(self, patient_dir: Path) -> Optional[Path]:
        """Map ``.../centre/vendor/PXXX`` -> ``<cache_dir>/centre/vendor/PXXX.pt``.

        Uses the last three path components as the relative key, which matches
        the layout produced by ``src/data/precompute_patient_cache.py``.
        """
        if self.cache_dir is None:
            return None
        parts = patient_dir.parts
        if len(parts) < 3:
            return None
        rel = Path(parts[-3]) / parts[-2] / f'{parts[-1]}.pt'
        return self.cache_dir / rel

    # ─── Convenience helpers for tests and introspection ────────────────
    def patient_dirs(self) -> list[Path]:
        """Unique sorted list of patient directories in the dataset."""
        return sorted({pdir for pdir, *_ in self.filename})

    def patients_per_scanner(self) -> dict[tuple[str, str], int]:
        """Count of unique patients per (centre, vendor) tuple."""
        counts: dict[tuple[str, str], int] = {}
        for pdir in self.patient_dirs():
            centre = pdir.parts[-3]
            vendor = pdir.parts[-2]
            counts[(centre, vendor)] = counts.get((centre, vendor), 0) + 1
        return counts

    # ─── Alternative constructor: from JSON patient-list split ─────────
    @classmethod
    def from_split_json(
        cls,
        split_path: str | Path,
        root: str | Path,
        key: str = 'train',
        **kwargs,
    ) -> 'CMRx4DFlowDataset':
        """Build a dataset from a FlowMRI-Net-style patient-list JSON.

        Parameters
        ----------
        split_path :
            Path to a JSON produced by ``scripts/build_train_val_split.py``.
            Must contain a ``key`` field mapping to a list of relative
            patient paths.
        root :
            Directory under which the relative paths in
            ``split[key]`` resolve (e.g. ``TrainSet/Aorta``).
        key :
            Which split list to materialise — typically ``'train'`` or
            ``'val'``.
        **kwargs :
            Forwarded verbatim to ``CMRx4DFlowDataset.__init__``.
            ``patient_dirs`` is set automatically from the split file and
            cannot be overridden.
        """
        with open(split_path, 'r') as f:
            split = json.load(f)
        if key not in split:
            raise KeyError(
                f"split JSON {split_path!r} has no key {key!r}; "
                f"available keys: {sorted(split.keys())}"
            )
        if 'patient_dirs' in kwargs:
            raise ValueError(
                "from_split_json sets patient_dirs internally; "
                "passing it via kwargs is not allowed."
            )
        if 'roots' in kwargs:
            raise ValueError(
                "from_split_json sets the patient list directly; "
                "passing roots via kwargs is not allowed."
            )
        root_path = Path(root)
        patient_dirs = [root_path / rel for rel in split[key]]
        return cls(patient_dirs=patient_dirs, **kwargs)

