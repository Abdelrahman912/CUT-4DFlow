"""Custom samplers for the CMRx dataset.

BlockShuffleSampler — yields indices in "patient blocks":
  1. Shuffle the patient order each epoch
  2. Within each patient, shuffle the FE-slice indices
  3. Yield all entries of patient_i, then all of patient_{i+1}, ...

Effect: src.data.dataset.CMRx4DFlowDataset's @lru_cache(maxsize=2) becomes
near-perfect — each patient's .pt is loaded ONCE per epoch instead of ~112
times. Drops disk I/O from 14,280 torch.load()s per epoch to ~122.

Within a block, R, kt-Gaussian mask, SSDU split, velocity, and cardiac window
still randomize per __getitem__, so the optimizer still sees varied gradients.
Only the underlying patient anatomy is shared for 112 consecutive steps.

Wire into the DataLoader:

    sampler = BlockShuffleSampler(train_dataset.patient_to_entries, seed=0)
    DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=2)

For multi-epoch training, call sampler.set_epoch(epoch) before each
training epoch — this re-seeds the shuffle so the order changes per epoch
but stays deterministic given (seed, epoch).
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Sequence

import torch
from torch.utils.data import Sampler


class BlockShuffleSampler(Sampler[int]):
    """Sampler that yields dataset indices in patient-block order.

    Parameters
    ----------
    patient_to_entries :
        Mapping of patient identifier (string or Path) to the list of
        dataset indices that belong to that patient. Typically built once
        from the dataset:
            patient_to_entries = dataset.patient_to_entries
    seed :
        Base seed. The actual shuffle uses (seed + epoch) so calling
        set_epoch(epoch) gives a different permutation per epoch.
    """

    def __init__(self, patient_to_entries: Dict[str, Sequence[int]], seed: int = 0):
        self.patient_to_entries = {str(k): list(v) for k, v in patient_to_entries.items()}
        if not self.patient_to_entries:
            raise ValueError("BlockShuffleSampler requires at least one patient")
        for k, v in self.patient_to_entries.items():
            if len(v) == 0:
                raise ValueError(f"patient {k!r} has 0 entries — cannot block-shuffle")
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
            patient = patients[pi]
            entries = self.patient_to_entries[patient]
            fe_perm = torch.randperm(len(entries), generator=g).tolist()
            for fi in fe_perm:
                yield entries[fi]

    def __len__(self) -> int:
        return self._total


class PatientInterleaveSampler(Sampler[int]):
    """Block-shuffle, but with ``num_workers`` patients in flight at once.

    Motivation
    ----------
    :class:`BlockShuffleSampler` hands the *same* patient to every DataLoader
    worker at once, because DataLoader assigns sampler position ``i`` to worker
    ``i % num_workers`` and a patient's slices occupy consecutive positions.
    Two consequences:

      * every worker independently loads the same ``.pt`` (so it is read
        ``num_workers`` times per epoch, not once), and
      * ~112 consecutive optimiser steps see one patient's anatomy, so a
        gradient-accumulation window is entirely one subject.

    This sampler instead partitions patients across workers and round-robins
    between them, so position ``i`` and ``i+1`` come from *different* patients.
    Each patient is then touched by exactly one worker -> one ``.pt`` read per
    patient per epoch, and an accumulation window spans ``num_workers`` subjects.

    Ordering is deterministic given ``(seed, epoch)``; call :meth:`set_epoch`.

    Notes
    -----
    Patients are assigned to workers least-loaded-first (after a per-epoch
    shuffle) so the per-worker streams end up nearly equal in length. Whatever
    imbalance remains shows up only as a short tail at the end of the epoch,
    where the interleaving degrades back to block-like behaviour. Every index is
    still yielded exactly once.

    ``num_workers=0`` (single-process loading) reduces to one stream, i.e. the
    same ordering guarantees as :class:`BlockShuffleSampler`.
    """

    def __init__(self, patient_to_entries: Dict[str, Sequence[int]],
                 num_workers: int = 1, seed: int = 0):
        self.patient_to_entries = {str(k): list(v) for k, v in patient_to_entries.items()}
        if not self.patient_to_entries:
            raise ValueError("PatientInterleaveSampler requires at least one patient")
        for k, v in self.patient_to_entries.items():
            if len(v) == 0:
                raise ValueError(f"patient {k!r} has 0 entries — cannot interleave")
        self.num_streams = max(1, int(num_workers))
        self.seed = int(seed)
        self.epoch = 0
        self._total = sum(len(v) for v in self.patient_to_entries.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _streams(self, g) -> List[List[int]]:
        """One index stream per worker: its patients, each internally shuffled."""
        patients = list(self.patient_to_entries.keys())
        order = torch.randperm(len(patients), generator=g).tolist()

        # Least-loaded-first assignment keeps the streams close to equal length
        # without sorting by size (which would make the split identical every epoch).
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
