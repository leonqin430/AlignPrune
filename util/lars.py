"""LARS optimizer (You et al., "Large batch training of convolutional networks", 2017), as in the
CIFAR example of InfoBatch (https://github.com/NUS-HPC-AI-Lab/InfoBatch)."""
import torch
from torch.optim.optimizer import Optimizer


class Lars(Optimizer):
    """SGD with momentum whose learning rate is scaled, per parameter tensor, by the trust ratio
    eeta * ||w|| / (||g|| + weight_decay * ||w|| + epsilon), clipped to [0, 50]."""

    def __init__(self, params, lr=1e-3, momentum=0, eeta=1e-3, weight_decay=0, epsilon=0.0):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if eeta <= 0:
            raise ValueError(f'Invalid eeta value: {eeta}')
        if epsilon < 0:
            raise ValueError(f'Invalid epsilon value: {epsilon}')
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eeta=eeta, epsilon=epsilon, lars=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay, momentum = group['weight_decay'], group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                lr = group['lr']
                if group['lars']:
                    w_norm, g_norm = torch.norm(p), torch.norm(grad)
                    trust_ratio = torch.where(
                        (w_norm > 0) & (g_norm > 0),
                        group['eeta'] * w_norm / (g_norm + weight_decay * w_norm + group['epsilon']),
                        torch.ones_like(w_norm)).clamp(0.0, 50)
                    lr = lr * trust_ratio.item()
                    if weight_decay != 0:
                        grad = grad.add(p, alpha=weight_decay)
                grad = torch.clamp(grad, -10.0, 10.0)

                if momentum != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)
                    grad = buf

                p.add_(grad, alpha=-lr)
        return loss
