import torch
from torch.optim.optimizer import Optimizer

class LARS(Optimizer):
    r"""Layer-wise Adaptive Rate Scaling (LARS) for large-batch training."""
    def __init__(self, params, lr, momentum=0.9, weight_decay=1e-6,
                 eta=0.001, eps=1e-9, nesterov=False,
                 exclude_from_adaptation=('bias', 'bn')):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        eta=eta, eps=eps, nesterov=nesterov)
        super().__init__(params, defaults)
        self.exclude = exclude_from_adaptation

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            wd = group['weight_decay']
            eta = group['eta']
            eps = group['eps']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad

                # decoupled weight decay (already excluded for BN/bias via param_groups)
                if wd != 0:
                    g = g.add(p, alpha=wd)

                # trust ratio: only for tensors with ndim > 1
                # (skip bias/BN/LayerNorm which are vectors)
                trust = 1.0
                if p.ndim > 1:
                    w_norm = p.norm()
                    g_norm = g.norm()
                    if w_norm > 0 and g_norm > 0:
                        trust = eta * w_norm / (g_norm + eps)

                # momentum
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.zeros_like(p)
                else:
                    buf = state['momentum_buffer']

                buf.mul_(mu).add_(g, alpha=trust * lr)
                if nesterov:
                    update = g.mul(trust * lr).add(buf, alpha=mu)
                else:
                    update = buf

                p.add_(-update)
