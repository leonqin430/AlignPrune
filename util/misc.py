import builtins
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def init_distributed_mode(args):
    """Initializes torch.distributed when launched with torchrun; single-GPU training otherwise."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.distributed = True
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
    else:
        args.rank, args.world_size, args.gpu, args.distributed = 0, 1, 0, False
        torch.cuda.set_device(args.gpu)
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """Disables printing in the non-master processes (print(..., force=True) still prints)."""
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return not is_dist_avail_and_initialized() or dist.get_rank() == 0


def all_reduce_sum(values):
    """Sums a list of numbers over all processes."""
    if not is_dist_avail_and_initialized():
        return values
    t = torch.tensor(values, dtype=torch.float64, device='cuda')
    dist.all_reduce(t)
    return t.tolist()


def all_reduce_mean(value):
    """Averages a number over all processes."""
    if not is_dist_avail_and_initialized():
        return value
    return all_reduce_sum([value])[0] / dist.get_world_size()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
