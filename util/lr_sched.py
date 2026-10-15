import math


def adjust_learning_rate(optimizer, epoch, args):
    """One-cycle cosine schedule (as torch.optim.lr_scheduler.OneCycleLR) at a fractional epoch.

    The learning rate depends on the training progress only, so that the schedule is not affected by
    the number of iterations of an epoch, which changes from epoch to epoch under dynamic pruning.
    """
    progress = epoch / args.epochs
    initial_lr = args.max_lr / args.div_factor
    min_lr = initial_lr / args.final_div
    if progress < args.pct_start:
        lr = _cosine(initial_lr, args.max_lr, progress / args.pct_start)
    else:
        lr = _cosine(args.max_lr, min_lr, (progress - args.pct_start) / (1 - args.pct_start))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def _cosine(start, end, pct):
    return end + (start - end) / 2.0 * (math.cos(math.pi * pct) + 1)
