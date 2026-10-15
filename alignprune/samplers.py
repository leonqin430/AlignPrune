"""Samplers that re-draw the training subset of a pruning wrapper at the beginning of every epoch.

Adapted from InfoBatch (https://github.com/NUS-HPC-AI-Lab/InfoBatch).
"""
import math

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler


class IBSampler:
    """Iterates over `dataset.prune()` during the first `delta * num_epochs` epochs and over
    `dataset.no_prune()` afterwards (annealing). The number of samples changes every epoch."""

    def __init__(self, dataset, seed=0):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.annealing = False
        self.sample_indices = list(range(len(dataset)))

    def reset(self):
        # the subset is drawn with numpy, re-seeded every epoch so that all processes draw the same subset
        np.random.seed(self.seed + self.epoch)
        if self.epoch + 1 > self.dataset.stop_prune:
            if not self.annealing:
                self.dataset.reset_weights()
                self.annealing = True
            indices = self.dataset.no_prune()
        else:
            indices = self.dataset.prune()
        self.sample_indices = np.asarray(indices).tolist()
        self.epoch += 1

    def __iter__(self):
        self.reset()
        return iter(self.sample_indices)

    def __len__(self):
        return len(self.sample_indices)


class DistributedIBSampler(DistributedSampler):
    """Shards the indices drawn by an `IBSampler` across processes."""

    def __init__(self, sampler, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=True):
        super().__init__(sampler.dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.ib_sampler = sampler

    def __iter__(self):
        self.ib_sampler.reset()
        indices = self.ib_sampler.sample_indices
        if self.drop_last:
            self.num_samples = len(indices) // self.num_replicas
        else:
            self.num_samples = math.ceil(len(indices) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(indices), generator=g).tolist()
        else:
            order = list(range(len(indices)))
        if self.total_size > len(order):  # drop_last=False: pad to make the shards equally long
            order += (order * math.ceil(self.total_size / len(order)))[:self.total_size - len(order)]
        order = order[self.rank:self.total_size:self.num_replicas]
        return iter([indices[i] for i in order])

    def __len__(self):
        return self.num_samples
