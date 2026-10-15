"""CIFAR-10N / CIFAR-100N with real (human) or synthetic label noise, and the clean reference set."""
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

# label sets of CIFAR-10N (Wei et al., ICLR 2022), https://github.com/UCSC-REAL/cifar-10-100n
CIFAR10N_LABELS = {'aggre': 'aggre_label', 'worst': 'worse_label',
                   'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3'}
# asymmetric noise on CIFAR-10 swaps the two classes of each pair:
# airplane-bird, automobile-truck, cat-dog, deer-horse, frog-ship
CIFAR10_ASYM = {0: 2, 2: 0, 1: 9, 9: 1, 3: 5, 5: 3, 4: 7, 7: 4, 6: 8, 8: 6}

MEAN, STD = (0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)


class CIFARSubset(Dataset):
    """A subset of the CIFAR training images with the given labels and transform."""

    def __init__(self, images, targets, indices, transform):
        self.images = images[indices]
        self.targets = np.asarray(targets)[indices]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.images[index])), int(self.targets[index])


def cifar100_superclasses(root):
    """The 20 groups of 5 fine classes that form the CIFAR-100 superclasses."""
    with open(os.path.join(root, 'cifar-100-python', 'train'), 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
    groups = [set() for _ in range(20)]
    for fine, coarse in zip(entry['fine_labels'], entry['coarse_labels']):
        groups[coarse].add(fine)
    return [sorted(g) for g in groups]


def noisy_labels(args, clean, rng):
    """Training labels for `args.noise_type`."""
    if args.noise_type == 'clean':
        return clean.copy()

    if args.noise_type in ('sym', 'asym'):
        noisy = clean.copy()
        idx = rng.permutation(len(clean))[:int(args.noise_rate * len(clean))]
        num_classes = clean.max() + 1
        if args.noise_type == 'sym':  # to one of the other classes, uniformly
            noisy[idx] = (clean[idx] + rng.randint(1, num_classes, size=len(idx))) % num_classes
        elif args.dataset == 'cifar10':
            noisy[idx] = [CIFAR10_ASYM[c] for c in clean[idx]]
        else:  # to one of the other classes of the same superclass, uniformly
            group = {c: g for g in cifar100_superclasses(args.data_path) for c in g}
            for i in idx:
                others = [c for c in group[clean[i]] if c != clean[i]]
                noisy[i] = others[rng.randint(len(others))]
        return noisy

    # real noise: human annotations of CIFAR-N
    if args.dataset == 'cifar10' and args.noise_type in CIFAR10N_LABELS:
        file, key = 'CIFAR-10_human.pt', CIFAR10N_LABELS[args.noise_type]
    elif args.dataset == 'cifar100' and args.noise_type == 'real':
        file, key = 'CIFAR-100_human.pt', 'noisy_label'
    else:
        raise ValueError(f'unknown noise type {args.noise_type} for {args.dataset}')
    labels = torch.load(os.path.join(args.data_path, file), weights_only=False)
    assert np.array_equal(np.asarray(labels['clean_label']), clean), 'CIFAR-N labels do not match the CIFAR training set'
    return np.asarray(labels[key]).astype(np.int64)


def build_cifar_n(args):
    """Returns the training set with noisy labels, the clean reference set and the test set.

    The reference set is a random subset of `args.ref_size` training images with their clean labels.
    It is removed from the training set of every method, so that all methods train on the same images.
    """
    cifar = CIFAR10 if args.dataset == 'cifar10' else CIFAR100
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train = cifar(args.data_path, train=True, download=args.download)
    dataset_test = cifar(args.data_path, train=False, transform=test_transform, download=args.download)

    rng = np.random.RandomState(args.seed)
    clean = np.asarray(train.targets, dtype=np.int64)
    noisy = noisy_labels(args, clean, rng)
    perm = rng.permutation(len(clean))
    ref_idx, train_idx = np.sort(perm[:args.ref_size]), np.sort(perm[args.ref_size:])

    dataset_train = CIFARSubset(train.data, noisy, train_idx, train_transform)
    dataset_ref = CIFARSubset(train.data, clean, ref_idx, test_transform)
    noise = (noisy[train_idx] != clean[train_idx]).mean()
    print(f'{args.dataset}, noise {args.noise_type}: {len(dataset_train)} training images ({100 * noise:.1f}% '
          f'noisy labels), {len(dataset_ref)} clean reference images, {len(dataset_test)} test images')
    return dataset_train, dataset_ref, dataset_test
