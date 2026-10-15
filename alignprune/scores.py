"""Correlation functions rho of the Dynamic Alignment Score (DAS).

Each function compares the loss trajectories of B training samples, `traj` of shape [N, B] (oldest
epoch first), with the reference trajectory `ref` of shape [N], and returns one score per sample.
A higher score means that the sample follows the clean reference more closely.
"""
import torch

EPS = 1e-8


def pearson(traj, ref):
    """Pearson correlation (the default rho)."""
    traj = traj - traj.mean(dim=0, keepdim=True)
    ref = (ref - ref.mean()).view(-1, 1)
    num = (traj * ref).sum(dim=0)
    den = torch.sqrt((traj ** 2).sum(dim=0) * (ref ** 2).sum()) + EPS
    return num / den


def cosine(traj, ref):
    """Cosine similarity."""
    ref = ref.view(-1, 1)
    return (traj * ref).sum(dim=0) / (traj.norm(dim=0) * (ref.norm() + EPS) + EPS)


def dtw(traj, ref):
    """Dynamic time warping distance (absolute-difference cost), mapped to a similarity in [0, 1]."""
    x = traj.t()                                          # [B, N]
    cost = (x.unsqueeze(2) - ref.view(1, 1, -1)).abs()    # [B, N, N]
    B, N = x.shape
    acc = torch.full((B, N + 1, N + 1), float('inf'), dtype=x.dtype, device=x.device)
    acc[:, 0, 0] = 0
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            prev = torch.minimum(torch.minimum(acc[:, i - 1, j], acc[:, i, j - 1]), acc[:, i - 1, j - 1])
            acc[:, i, j] = cost[:, i - 1, j - 1] + prev
    dist = acc[:, N, N]
    return 1 - dist / (dist.max() + EPS)


SCORES = {'pearson': pearson, 'cosine': cosine, 'dtw': dtw}
