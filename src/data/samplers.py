"""Patient-block samplers.

Both yield each patient's slices together so a patient's cache is loaded once per epoch.
``BlockShuffleSampler`` emits one patient at a time; ``PatientInterleaveSampler`` keeps
``num_workers`` patients in flight so a gradient-accumulation window spans several subjects.
Within a block, R / mask / SSDU split still randomise per item. Call ``set_epoch`` each epoch.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Sequence

import torch
from torch.utils.data import Sampler


class BlockShuffleSampler(Sampler[int]):
    """Yield dataset indices patient-block by patient-block (shuffled per epoch)."""

    def __init__(self, patient_to_entries: Dict[str, Sequence[int]], seed: int = 0):
        self.patient_to_entries = {str(k): list(v) for k, v in patient_to_entries.items()}
        if not self.patient_to_entries:
            raise ValueError("BlockShuffleSampler requires at least one patient")
        for k, v in self.patient_to_entries.items():
            if len(v) == 0:
                raise ValueError(f"patient {k!r} has 0 entries")
        self.seed = int(seed)
        self.epoch = 0
        self._total = sum(len(v) for v in self.patient_to_entries.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        patients: List[str] = list(self.patient_to_entries.keys())
        patient_perm = torch.randperm(len(patients), generator=g).tolist()
        for pi in patient_perm:
            entries = self.patient_to_entries[patients[pi]]
            fe_perm = torch.randperm(len(entries), generator=g).tolist()
            for fi in fe_perm:
                yield entries[fi]

    def __len__(self) -> int:
        return self._total


class PatientInterleaveSampler(Sampler[int]):
    """Block-shuffle with ``num_workers`` patients interleaved, so each patient is read once per epoch.

    Patients are assigned to workers least-loaded-first (after a per-epoch shuffle) and the streams
    are round-robined. ``num_workers=0`` reduces to a single stream.
    """

    def __init__(self, patient_to_entries: Dict[str, Sequence[int]],
                 num_workers: int = 1, seed: int = 0):
        self.patient_to_entries = {str(k): list(v) for k, v in patient_to_entries.items()}
        if not self.patient_to_entries:
            raise ValueError("PatientInterleaveSampler requires at least one patient")
        for k, v in self.patient_to_entries.items():
            if len(v) == 0:
                raise ValueError(f"patient {k!r} has 0 entries")
        self.num_streams = max(1, int(num_workers))
        self.seed = int(seed)
        self.epoch = 0
        self._total = sum(len(v) for v in self.patient_to_entries.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _streams(self, g) -> List[List[int]]:
        patients = list(self.patient_to_entries.keys())
        order = torch.randperm(len(patients), generator=g).tolist()
        groups: List[List[str]] = [[] for _ in range(self.num_streams)]
        loads = [0] * self.num_streams
        for pi in order:
            p = patients[pi]
            w = min(range(self.num_streams), key=lambda i: loads[i])
            groups[w].append(p)
            loads[w] += len(self.patient_to_entries[p])

        streams: List[List[int]] = []
        for grp in groups:
            stream: List[int] = []
            for p in grp:
                entries = self.patient_to_entries[p]
                perm = torch.randperm(len(entries), generator=g).tolist()
                stream.extend(entries[i] for i in perm)
            streams.append(stream)
        return streams

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        streams = self._streams(g)
        pos = [0] * len(streams)
        remaining = self._total
        while remaining:
            for w, stream in enumerate(streams):
                if pos[w] < len(stream):
                    yield stream[pos[w]]
                    pos[w] += 1
                    remaining -= 1

    def __len__(self) -> int:
        return self._total
