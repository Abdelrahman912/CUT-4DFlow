"""Prospectively-undersampled SSDU dataset (TaskS1 / TaskS2 fine-tuning).

Validation-set patients ship WITHOUT ground truth:
    kdata_ktGaussian{R}.mat   undersampled k-space  (Nv, Nt, Nc, SPE, PE, FE)
    usmask_ktGaussian{R}.mat  acquired Omega mask   (1, Nt, 1, SPE, PE, 1)
    coilmap.mat, segmask.mat, params.csv

SSDU is the only training signal: the SHIPPED Omega is split into disjoint
Theta (input) / Lambda (loss target) with the same ``uniform_disjoint_selection``
used in retrospective training. Sample dicts mirror ``CMRx4DFlowDataset`` exactly
(same keys/shapes), so ``_slice_to_tensors``/``training_step``/``validate`` work
unchanged. ``img_gt`` is zeros — this dataset supports supervised=False ONLY.

orientation='canonical' flips to the training frame (S1 Aorta); 'native' skips
flipping (S2 organs, whose spatial_order permutes — flips can't canonicalise it).
Array axis ROLES (SPE, PE, FE-last, FE fully sampled) are identical either way.
"""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from torch.utils.data import Dataset

from src.data.dataset import _k2i_numpy, _seed_from_key, _clone_for_ipc
from src.data.h5_reader import h5_complex, read_params_csv
from src.data.orientation import canonicalise_spatial
from src.data.ssdu_split import uniform_disjoint_selection


@lru_cache(maxsize=2)          # matches the retrospective loader; per DataLoader worker
def _load_prospective_patient(pdir: str, canonicalise: bool) -> dict:
    """Load + FE-iFFT one prospective patient. LRU-cached per worker (arrays can be GB)."""
    pdir = Path(pdir)
    kfile = next(pdir.glob("kdata_ktGaussian*.mat"))
    R = int(kfile.stem.replace("kdata_ktGaussian", ""))
    kdata = h5_complex(kfile, "kdata_ktGaussian")                    # (Nv,Nt,Nc,SPE,PE,FE)
    coil = h5_complex(pdir / "coilmap.mat", "coilmap")               # (Nc,SPE,PE,FE)
    with h5py.File(pdir / f"usmask_ktGaussian{R}.mat", "r") as f:
        usmask = np.array(f[f"usmask_ktGaussian"]).astype(np.float32)  # (1,Nt,1,SPE,PE,1)
    with h5py.File(pdir / "segmask.mat", "r") as f:
        segmask = np.array(f["segmask"]).astype(bool)                # (SPE,PE,FE)
    params = read_params_csv(pdir / "params.csv")
    so = params["spatial_order"]
    if canonicalise:
        kdata = canonicalise_spatial(kdata, so)
        coil = canonicalise_spatial(coil, so)
        segmask = canonicalise_spatial(segmask, so)
        usmask = canonicalise_spatial(usmask, so)                    # FE axis len 1 -> no-op flip
    kdh = _k2i_numpy(kdata, ax=[-1]).astype(np.complex64)            # FE readout -> image (hybrid)
    return dict(
        kdh=kdh, coil=coil.astype(np.complex64), usmask=usmask, segmask=segmask,
        R=R, B0=np.float32(float(params.get("field_strength") or 3.0)),
        VENC=np.asarray(params["VENC"], np.float32),
    )


class ProspectiveSSDUDataset(Dataset):
    """SSDU fine-tuning entries over prospective validation patients.

    mode='train': one entry per (patient, FE position); random cardiac window +
    fresh random Theta/Lambda split each draw (like retrospective training).
    mode='val'  : one entry per patient; D_size=-1 (full volume), full cardiac
    cycle, deterministic split — for ``validate()``'s middle-FE slice.
    """

    def __init__(self, patient_dirs: Sequence[Path], mode: str = "train",
                 T_size: int = 5, D_size: int = 1, rho: float = 0.2, ssdu_r2: int = 9,
                 orientation: str = "canonical"):
        assert mode in ("train", "val")
        assert orientation in ("canonical", "native")
        self.mode = mode
        self.T_size = int(T_size)
        self.D_size = -1 if mode == "val" else int(D_size)
        self.rho = float(rho)
        self.ssdu_r2 = int(ssdu_r2)
        self.canon = orientation == "canonical"
        self.entries: list[tuple[Path, int]] = []
        for pdir in sorted(Path(p) for p in patient_dirs):
            if self.mode == "val" or self.D_size == -1:
                self.entries.append((pdir, 0))
            else:
                with h5py.File(pdir / "coilmap.mat", "r") as f:
                    FE = f["coilmap"].shape[-1]                       # (Nc,SPE,PE,FE) on disk
                for fe in range(0, FE, self.D_size):
                    self.entries.append((pdir, fe))

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def filename(self) -> list:
        """[(patient_dir, fe_start, R)] — same interface the trainer's scanner-balance
        counter expects from CMRx4DFlowDataset. R is per-patient here (shipped mask)."""
        return [(str(p), fe, -1) for p, fe in self.entries]

    @property
    def patient_to_entries(self) -> dict:
        """{patient_dir: [entry indices]} — consumed by the Block/Interleave samplers."""
        d: dict = {}
        for i, (pdir, _fe) in enumerate(self.entries):
            d.setdefault(str(pdir), []).append(i)
        return d

    def __getitem__(self, idx: int) -> dict:
        pdir, fe_start = self.entries[idx]
        blob = _load_prospective_patient(str(pdir), self.canon)
        kdh, coil, usmask = blob["kdh"], blob["coil"], blob["usmask"]
        Nv, Nt, Nc, SPE, PE, FE = kdh.shape

        # cardiac window (train: random with wrap; val: full cycle)
        if self.mode == "val" or self.T_size >= Nt:
            cardiac_bins = np.arange(Nt)
        else:
            first = random.randint(-self.T_size + 1, Nt - self.T_size)
            cardiac_bins = np.mod(np.arange(first, first + self.T_size), Nt).astype(np.int64)

        fe_sl = slice(None) if self.D_size == -1 else slice(fe_start, fe_start + self.D_size)
        f = np.take(kdh[..., fe_sl], cardiac_bins, axis=1)            # (Nv,T,Nc,SPE,PE,D) hybrid
        c = coil[..., fe_sl]
        s = blob["segmask"][..., fe_sl]
        M_Omega = np.take(usmask, cardiac_bins, axis=1)               # (1,T,1,SPE,PE,1) SHIPPED mask

        seed = _seed_from_key(f"{pdir}|prospective") if self.mode == "val" else None
        M_Theta, M_Lambda = uniform_disjoint_selection(
            M_Omega, rho=self.rho, r2=self.ssdu_r2, seed=seed,
        )
        kdata_theta = (f * M_Theta).astype(np.complex64)
        kdata_lambda = (f * M_Lambda).astype(np.complex64)
        img_zf = np.sum(_k2i_numpy(kdata_theta, ax=[-2, -3]) * np.conj(c), axis=-4)
        denom = float(np.linalg.norm(np.abs(kdata_theta) != 0))
        norm = float(np.linalg.norm(kdata_theta)) / max(denom, 1.0) or 1.0


        out = {
            "kdata_theta":  (kdata_theta / norm).astype(np.complex64),
            "kdata_lambda": (kdata_lambda / norm).astype(np.complex64),
            "mask_omega":   M_Omega.astype(np.float32),
            "mask_theta":   M_Theta.astype(np.float32),
            "mask_lambda":  M_Lambda.astype(np.float32),
            "img_zf":       (img_zf / norm).astype(np.complex64),
            "img_gt":       np.zeros_like(img_zf),                    # NO GT (SSDU only)
            "coilmap":      c,
            "segmask":      s,
            "norm":         np.float32(norm),
            "R":            np.int32(blob["R"]),
            "B0":           blob["B0"],
            "seg_idx":      np.int32(-1),
            "cardiac_bins": cardiac_bins.astype(np.int64),
            "fe_start":     np.int32(fe_start),
            "patient_dir":  str(pdir),
            "VENC":         blob["VENC"],
            "scanner_weight": np.float32(1.0),
        }
        return {k: _clone_for_ipc(v) for k, v in out.items()}
