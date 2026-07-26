"""Deterministic per-scanner train/val patient split."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from src.data.dataset import find_valid_patients


def scanner_of(patient_dir: Path) -> str:
    """Scanner key ``centre/vendor``."""
    return f'{patient_dir.parts[-3]}/{patient_dir.parts[-2]}'


def make_split(root, val_per_scanner: int = 1, seed: int = 1337):
    """Hold out ``val_per_scanner`` patients per scanner. Deterministic given ``seed``.

    Returns ``(train_dirs, val_dirs)`` (sorted lists of patient directories).
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
        rng = random.Random(f'{seed}:{scanner}')
        val_pick = set(rng.sample(pts, k))
        for p in pts:
            (val if p in val_pick else train).append(p)
    return sorted(train), sorted(val)
