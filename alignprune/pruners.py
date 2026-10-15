"""Dynamic data pruning wrappers: InfoBatch, SeTa, their AlignPrune variants, and random baselines.

Every wrapper is a Dataset around the training set whose sampler re-draws the training subset at the
beginning of each epoch from the current per-sample scores:

    trainset = AlignPrune(trainset, num_epochs, prune_ratio=0.5, window_size=25)
    loader = DataLoader(trainset, sampler=trainset.sampler, batch_size=128)
    for epoch in range(num_epochs):
        for images, targets in loader:
            loss = criterion(model(images), targets)    # per-sample losses (reduction='none')
            loss = trainset.update(loss, epoch)         # records the losses, returns the rescaled mean
            ...
        trainset.update_reference(ref_loss, epoch)      # mean loss on the clean reference set

InfoBatch, SeTa and the sampler are adapted from https://github.com/NUS-HPC-AI-Lab/InfoBatch and
https://github.com/mrazhou/SeTa.
"""
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataloader import _BaseDataLoaderIter

from .samplers import DistributedIBSampler, IBSampler
from .scores import SCORES


def _all_gather(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensors = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors, tensor)
    return torch.cat(tensors)


class InfoBatch(Dataset):
    """InfoBatch (Qin et al., ICLR 2024). The score of a sample is its latest training loss. At every
    epoch, the samples scored below the mean are pruned with probability `prune_ratio`, and the loss of
    the kept ones is scaled by 1 / (1 - prune_ratio) to keep the gradient unbiased. The full dataset is
    used during the last (1 - delta) fraction of the epochs (annealing)."""

    uses_reference = False

    def __init__(self, dataset, num_epochs, prune_ratio=0.5, delta=0.875, seed=0):
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.keep_ratio = min(1.0, max(0.1, 1.0 - prune_ratio))
        self.delta = delta
        self.seed = seed
        self.scores = torch.full((len(dataset),), 3.0)
        self.weights = torch.ones(len(dataset))
        self.cur_batch_index = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]

    def __getattr__(self, name):
        # delegate unknown attributes to the wrapped dataset, except `dataset` itself (e.g. while
        # unpickling) and special methods such as `__getitems__`, which would bypass __getitem__
        if name == 'dataset' or name.startswith('__'):
            raise AttributeError(name)
        return getattr(self.dataset, name)

    @property
    def sampler(self):
        sampler = IBSampler(self, seed=self.seed)
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedIBSampler(sampler, seed=self.seed)
        return sampler

    @property
    def stop_prune(self):
        return self.num_epochs * self.delta

    def set_active_indices(self, indices):
        self.cur_batch_index = indices

    def update(self, values, epoch=None):
        """Records the per-sample losses `values` of the current batch and returns their rescaled mean."""
        assert self.cur_batch_index is not None and len(self.cur_batch_index) == len(values), \
            'update() must be called once per batch with the per-sample losses of that batch'
        indices = self.cur_batch_index.to(values.device)
        weights = self.weights[self.cur_batch_index].to(values.device)
        self.cur_batch_index = None
        indices, losses = _all_gather(indices), _all_gather(values.detach().float())
        self.record(indices.cpu(), losses.cpu(), epoch)
        return (values * weights).mean()

    def record(self, indices, losses, epoch):
        self.scores[indices] = losses

    def update_reference(self, ref_loss, epoch):
        pass

    def reset_weights(self):
        self.weights[:] = 1

    def prune(self):
        """Indices of the next epoch: the samples scored above the mean plus a random `keep_ratio`
        fraction of the others, whose loss is upweighted."""
        low = (self.scores < self.scores.mean()).numpy()
        low_indices = np.where(low)[0]
        kept = np.random.choice(low_indices, int(self.keep_ratio * len(low_indices)), replace=False)
        self.reset_weights()
        self.weights[torch.from_numpy(kept)] = 1 / self.keep_ratio
        indices = np.concatenate([np.where(~low)[0], kept])
        np.random.shuffle(indices)
        return indices

    def no_prune(self):
        return np.random.permutation(len(self))


def kmeans_group(scores, indices, num_group):
    """Splits `indices` into `num_group` groups by 1-D k-means on `scores`, ordered by increasing score."""
    if num_group == 1:
        return [indices]
    if scores.max() == scores.min():
        return np.array_split(indices, num_group)
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=num_group, random_state=0, n_init=10).fit(scores.reshape(-1, 1)).labels_
    centers = [scores[labels == k].mean() if np.any(labels == k) else np.inf for k in range(num_group)]
    return [indices[labels == k] for k in np.argsort(centers)]


def slide_window(groups, iteration, window_scale):
    """The window of `window_scale * len(groups)` consecutive groups used at `iteration`."""
    if iteration == 0:
        return groups
    size = round(len(groups) * window_scale)
    assert size > 0
    start = iteration % (len(groups) - size + 1)
    return groups[start:start + size]


class SeTa(InfoBatch):
    """SeTa (Zhou et al., CVPR 2025). At every epoch, a random `prune_ratio` fraction of the samples is
    dropped, the others are clustered into `num_group` groups by score (k-means), and the training set is
    a window of `window_scale * num_group` consecutive groups that slides over the groups sorted by score.
    During annealing, a random `1 - prune_ratio` fraction of the samples is used."""

    def __init__(self, dataset, num_epochs, prune_ratio=0.1, num_group=5, window_scale=0.9, delta=0.875, seed=0):
        super().__init__(dataset, num_epochs, prune_ratio, delta, seed)
        self.num_group = num_group
        self.window_scale = window_scale
        self.iterations = 0

    def prune(self):
        indices = np.random.permutation(len(self))[:int(len(self) * self.keep_ratio)]
        groups = kmeans_group(self.scores.numpy()[indices], indices, self.num_group)
        groups = slide_window(groups, self.iterations, self.window_scale)
        self.iterations += 1
        indices = np.concatenate(groups)
        if len(indices) == 0:
            indices = np.random.choice(len(self), 1)
        np.random.shuffle(indices)
        return indices

    def no_prune(self):
        return np.random.permutation(len(self))[:int(len(self) * self.keep_ratio)]


class StaticRandom(InfoBatch):
    """Static random pruning: a fixed random subset of `1 - prune_ratio` of the samples."""

    def __init__(self, dataset, num_epochs, prune_ratio=0.5, seed=0):
        keep = np.random.RandomState(seed).permutation(len(dataset))[:int((1 - prune_ratio) * len(dataset))]
        super().__init__(Subset(dataset, np.sort(keep).tolist()), num_epochs, prune_ratio=0.0, delta=1.0, seed=seed)

    def prune(self):
        return self.no_prune()


class _DynamicAlignmentScore:
    """Replaces the loss-based score of a dynamic pruning method by the Dynamic Alignment Score (DAS):
    the correlation `rho` between the training losses of a sample over the last `window_size` epochs
    and the mean losses of a clean reference set over the same epochs. The DAS of the samples trained
    in an epoch is computed in `update_reference()`, once the reference loss of that epoch is known."""

    uses_reference = True

    def _init_alignment(self, window_size, score, init_loss=6.0):
        # ring buffers indexed by epoch % window_size; slots that are not filled yet hold `init_loss`,
        # which should be larger than the loss at initialization (6.0 for CIFAR-10/100)
        self.window_size = window_size
        self.score_fn = SCORES[score]
        self.train_history = torch.full((window_size, len(self.dataset)), init_loss)
        self.ref_history = torch.full((window_size,), init_loss)
        self.visited = torch.zeros(len(self.dataset), dtype=torch.bool)
        self.scores = torch.ones(len(self.dataset))

    def record(self, indices, losses, epoch):
        assert epoch is not None, 'AlignPrune needs the current epoch: update(loss, epoch)'
        self.train_history[epoch % self.window_size, indices] = losses
        self.visited[indices] = True

    def update_reference(self, ref_loss, epoch):
        """Stores the mean loss on the reference set after `epoch` and updates the DAS of the samples
        trained in this epoch. Must be called once per epoch, after the training epoch."""
        slot = epoch % self.window_size
        self.ref_history[slot] = float(ref_loss)
        order = (torch.arange(self.window_size) + slot + 1) % self.window_size  # oldest to newest
        indices = self.visited.nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            traj = self.train_history[order][:, indices]
            self.scores[indices] = self.score_fn(traj, self.ref_history[order])
        self.visited[:] = False


class AlignPrune(_DynamicAlignmentScore, InfoBatch):
    """AlignPrune on top of InfoBatch: the samples with a DAS below the mean are pruned with probability
    `prune_ratio`, and the loss of the kept ones is rescaled as in InfoBatch."""

    def __init__(self, dataset, num_epochs, prune_ratio=0.5, delta=0.875, window_size=25, score='pearson', seed=0):
        InfoBatch.__init__(self, dataset, num_epochs, prune_ratio, delta, seed)
        self._init_alignment(window_size, score)


class AlignPruneSeTa(_DynamicAlignmentScore, SeTa):
    """AlignPrune on top of SeTa: SeTa's grouping and sliding window operate on the DAS."""

    def __init__(self, dataset, num_epochs, prune_ratio=0.1, num_group=5, window_scale=0.9, delta=0.875,
                 window_size=25, score='pearson', seed=0):
        SeTa.__init__(self, dataset, num_epochs, prune_ratio, num_group, window_scale, delta, seed)
        self._init_alignment(window_size, score)


# The wrappers return (index, sample); this DataLoader hook strips the indices from every batch and
# hands them to the wrapper, so that update() knows which samples the losses belong to.
_dataloader_next = _BaseDataLoaderIter.__next__


def _next_with_indices(self):
    batch = _dataloader_next(self)
    if isinstance(self._dataset, InfoBatch):
        indices, batch = batch
        self._dataset.set_active_indices(indices)
    return batch


_BaseDataLoaderIter.__next__ = _next_with_indices
