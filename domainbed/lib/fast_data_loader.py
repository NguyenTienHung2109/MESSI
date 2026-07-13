# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random
from collections import defaultdict

import torch

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class LocationGroupedBatchSampler(torch.utils.data.Sampler):
    """Yield batches of (K * bs_per_group) indices, group-ordered by location.

    Each yielded batch:
      - picks K random source-camera locations (without replacement among
        qualified locations) and
      - draws bs_per_group indices per location WITH replacement (so locations
        smaller than bs_per_group don't cause hangs).
      - Returns a flat list of K * bs_per_group indices, ordered:
        [<bs_per_group indices from loc A>, <bs_per_group from loc B>, ...].

    Train.py reshapes the resulting flat batch into K group-batches of size
    bs_per_group before passing to the algorithm — algorithms with a
    domain-invariance loss (e.g. GMOE_InvMMD) then see num_domains=K source
    domains per step with non-trivial pairwise alignment terms.
    """

    def __init__(self, location_per_idx, K, bs_per_group, seed=0):
        self.K = int(K)
        self.bs_per_group = int(bs_per_group)
        self.batch_size = self.K * self.bs_per_group
        idx_by_location = defaultdict(list)
        for i, loc in enumerate(location_per_idx):
            idx_by_location[int(loc)].append(int(i))
        self.idx_by_location = dict(idx_by_location)
        self.qualified_locations = list(self.idx_by_location.keys())
        if len(self.qualified_locations) < self.K:
            raise ValueError(
                f"LocationGroupedBatchSampler: only {len(self.qualified_locations)} "
                f"locations available, need at least K={self.K}.")
        self.rng = random.Random(seed)

    def __iter__(self):
        while True:
            picks = self.rng.sample(self.qualified_locations, self.K)
            batch = []
            for loc in picks:
                pool = self.idx_by_location[loc]
                batch.extend(self.rng.choices(pool, k=self.bs_per_group))
            yield batch

    def __len__(self):
        # `_InfiniteSampler` ignores this; provide a large arbitrary length so
        # any caller using `len()` (e.g. tqdm) still works.
        return 10 ** 9


class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers,
                 batch_sampler=None):
        super().__init__()

        if batch_sampler is None:
            if weights is not None:
                sampler = torch.utils.data.WeightedRandomSampler(weights,
                    replacement=True,
                    num_samples=batch_size)
            else:
                sampler = torch.utils.data.RandomSampler(dataset,
                    replacement=True,
                    num_samples=batch_size)

            batch_sampler = torch.utils.data.BatchSampler(
                sampler,
                batch_size=batch_size,
                drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, replacement=False),
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length
