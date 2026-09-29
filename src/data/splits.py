"""Deterministic train/val patient split for the CMRx 4D-flow pipeline.

The split only decides WHICH patients are train vs val — it does not load or
serve data (that is ``CMRx4DFlowDataset``'s job). Patients are held out
per scanner so every scanner is represented in validation, and the choice is
deterministic given ``seed`` so it is reproducible across runs and across GA
hyper-parameter candidates.

This replaces the old static ``configs/train_val_split.json`` (+ its missing
generator): the split is now (re)derived from whatever patients exist under
``root`` at run time.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from src.data.dataset import find_valid_patients


def scanner_of(patient_dir: Path) -> str:
    """Scanner key ``centre/vendor`` (e.g. ``Center007/GE_30T_Architect``)."""
    return f'{patient_dir.parts[-3]}/{patient_dir.parts[-2]}'


def make_split(
    root: str | Path,
    val_per_scanner: int = 1,
    seed: int = 1337,
) -> tuple[list[Path], list[Path]]:
    """Per-scanner train/val split of the patients under ``root``.

    For each scanner (centre/vendor), ``val_per_scanner`` patients are held out
    for validation; the rest are training. Deterministic given ``seed``.

    Returns ``(train_dirs, val_dirs)`` — sorted lists of patient directories.
    With the default ``val_per_scanner=1`` over the 6 aortic scanners this
    yields 6 val patients (one per scanner) and the remaining 132 for training.
    """
    patients = find_valid_patients([root])
    by_scanner: dict[str, list[Path]] = defaultdict(list)
    for p in patients:
        by_scanner[scanner_of(p)].append(p)

    train: list[Path] = []
    val: list[Path] = []
    for scanner in sorted(by_scanner):
        pts = sorted(by_scanner[scanner])
        k = min(val_per_scanner, len(pts))
        # str seed -> deterministic across processes (unlike hash()), and
        # per-scanner so adding/removing one scanner doesn't reshuffle others.
        rng = random.Random(f'{seed}:{scanner}')
        val_pick = set(rng.sample(pts, k))
        for p in pts:
            (val if p in val_pick else train).append(p)
    return sorted(train), sorted(val)
