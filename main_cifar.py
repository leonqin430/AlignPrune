# Training on CIFAR-10N / CIFAR-100N with dynamic data pruning (AlignPrune, InfoBatch, SeTa, random).
#   single GPU: python main_cifar.py --prune_type alignprune ...
#   multi-GPU:  torchrun --nproc_per_node=2 main_cifar.py --prune_type alignprune ...
import argparse
import datetime
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

import util.lr_sched as lr_sched
import util.misc as misc
from alignprune import AlignPrune, AlignPruneSeTa, InfoBatch, SeTa, StaticRandom
from util.cifar_noisy import build_cifar_n
from util.lars import Lars
from util.resnet import resnet18, resnet50


def get_args_parser():
    parser = argparse.ArgumentParser('Dynamic data pruning on CIFAR-N')
    # data
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--download', action='store_true', help='download CIFAR with torchvision')
    parser.add_argument('--noise_type', default='clean', type=str,
                        help='clean | sym | asym | real (CIFAR-100N) | aggre | worst | rand1 | rand2 | rand3 (CIFAR-10N)')
    parser.add_argument('--noise_rate', default=0.0, type=float, help='rate of the synthetic sym / asym noise')
    parser.add_argument('--ref_size', default=1000, type=int,
                        help='number of clean training images held out as the reference set')

    # pruning
    parser.add_argument('--prune_type', default='alignprune',
                        choices=['none', 'static', 'random', 'infobatch', 'seta', 'alignprune', 'alignprune_seta'])
    parser.add_argument('--prune_ratio', default=0.5, type=float, help='pruning probability r')
    parser.add_argument('--delta', default=0.875, type=float, help='pruning stops after delta * epochs (annealing)')
    parser.add_argument('--num_group', default=5, type=int, help='SeTa: number of groups k')
    parser.add_argument('--window_scale', default=0.9, type=float, help='SeTa: window scale alpha')
    parser.add_argument('--window_size', default=25, type=int, help='AlignPrune: trajectory window size N')
    parser.add_argument('--score', default='pearson', choices=['pearson', 'cosine', 'dtw'],
                        help='AlignPrune: correlation function rho')

    # training
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--full_epochs', default=200, type=int,
                        help='epochs of full-data training, the budget against which the pruning ratio is reported')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size per GPU')
    parser.add_argument('--optimizer', default='lars', choices=['lars', 'sgd'])
    parser.add_argument('--max_lr', default=5.2, type=float, help='peak learning rate of the one-cycle schedule')
    parser.add_argument('--div_factor', default=25, type=float, help='initial lr = max_lr / div_factor')
    parser.add_argument('--final_div', default=1e4, type=float, help='final lr = initial lr / final_div')
    parser.add_argument('--pct_start', default=0.3, type=float, help='fraction of the epochs with increasing lr')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--label_smoothing', default=0.1, type=float)
    parser.add_argument('--amp', action='store_true', help='mixed-precision training')

    # misc
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output_dir', default='', help='where to write log.txt, nothing is written if empty')
    return parser


def build_pruner(dataset, args):
    kwargs = dict(num_epochs=args.epochs, prune_ratio=args.prune_ratio, seed=args.seed)
    seta = dict(num_group=args.num_group, window_scale=args.window_scale, delta=args.delta)
    das = dict(window_size=args.window_size, score=args.score)
    if args.prune_type == 'static':
        return StaticRandom(dataset, **kwargs)
    if args.prune_type == 'random':  # dynamic random pruning: SeTa with a single group
        return SeTa(dataset, num_group=1, window_scale=1.0, delta=args.delta, **kwargs)
    if args.prune_type == 'infobatch':
        return InfoBatch(dataset, delta=args.delta, **kwargs)
    if args.prune_type == 'seta':
        return SeTa(dataset, **seta, **kwargs)
    if args.prune_type == 'alignprune':
        return AlignPrune(dataset, delta=args.delta, **das, **kwargs)
    if args.prune_type == 'alignprune_seta':
        return AlignPruneSeTa(dataset, **seta, **das, **kwargs)
    raise ValueError(args.prune_type)


def train_one_epoch(model, criterion, data_loader, optimizer, scaler, epoch, args, pruner=None):
    model.train()
    loss_sum, correct, seen = 0.0, 0, 0
    for it, (images, targets) in enumerate(data_loader):
        if it == 0:
            num_steps = len(data_loader)  # the subset of this epoch is drawn when its first batch is loaded
        lr = lr_sched.adjust_learning_rate(optimizer, epoch + it / num_steps, args)
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        with torch.autocast('cuda', enabled=args.amp):
            outputs = model(images)
            loss = criterion(outputs, targets)  # per-sample losses
        loss = pruner.update(loss, epoch) if pruner is not None else loss.mean()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * len(targets)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        seen += len(targets)

    loss_sum, correct, seen = misc.all_reduce_sum([loss_sum, correct, seen])
    return {'lr': lr, 'train_loss': loss_sum / seen, 'train_acc': 100.0 * correct / seen, 'num_samples': int(seen)}


@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for images, targets in data_loader:
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        outputs = model(images)
        loss_sum += F.cross_entropy(outputs, targets, reduction='sum').item()
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += len(targets)
    return loss_sum / total, 100.0 * correct / total


def main(args):
    misc.init_distributed_mode(args)
    misc.set_seed(args.seed)
    print('{}'.format(args).replace(', ', ',\n'))

    dataset_train, dataset_ref, dataset_test = build_cifar_n(args)
    num_train = len(dataset_train)

    pruner = None
    if args.prune_type != 'none':
        pruner = build_pruner(dataset_train, args)
        assert not pruner.uses_reference or len(dataset_ref) > 0, f'{args.prune_type} needs --ref_size > 0'
        dataset_train, sampler_train = pruner, pruner.sampler
    elif args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True, seed=args.seed)
    else:
        sampler_train = RandomSampler(dataset_train)
    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size,
                                   num_workers=args.num_workers, pin_memory=True)
    data_loader_ref = DataLoader(dataset_ref, batch_size=500, num_workers=args.num_workers, pin_memory=True)
    data_loader_test = DataLoader(dataset_test, batch_size=500, num_workers=args.num_workers, pin_memory=True)

    num_classes = 10 if args.dataset == 'cifar10' else 100
    model = {'resnet18': resnet18, 'resnet50': resnet50}[args.model](num_classes).cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='none')
    optimizer_cls = Lars if args.optimizer == 'lars' else torch.optim.SGD
    optimizer = optimizer_cls(model.parameters(), lr=args.max_lr / args.div_factor, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    if args.output_dir and misc.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    num_seen = 0
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, scaler, epoch, args, pruner)
        num_seen += train_stats['num_samples']

        if pruner is not None and pruner.uses_reference:
            ref_loss, _ = evaluate(model, data_loader_ref)
            pruner.update_reference(misc.all_reduce_mean(ref_loss), epoch)
        test_loss, test_acc = evaluate(model, data_loader_test)

        elapsed = datetime.timedelta(seconds=int(time.time() - start_time))
        print(f"Epoch {epoch + 1}/{args.epochs}  lr {train_stats['lr']:.4f}  samples {train_stats['num_samples']}  "
              f"train loss {train_stats['train_loss']:.4f}  train acc {train_stats['train_acc']:.2f}  "
              f"test loss {test_loss:.4f}  test acc {test_acc:.2f}  [{elapsed}]")
        if args.output_dir and misc.is_main_process():
            log_stats = {'epoch': epoch, **train_stats, 'test_loss': test_loss, 'test_acc': test_acc}
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    prune_ratio = 1 - num_seen / (args.full_epochs * num_train)
    print(f'Final test accuracy: {test_acc:.2f}%')
    print(f'Pruning ratio: {100 * prune_ratio:.1f}% ({num_seen} trained samples vs. {args.full_epochs} epochs x '
          f'{num_train} samples of full-data training)')
    print(f'Training time: {datetime.timedelta(seconds=int(time.time() - start_time))}')
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main(get_args_parser().parse_args())
