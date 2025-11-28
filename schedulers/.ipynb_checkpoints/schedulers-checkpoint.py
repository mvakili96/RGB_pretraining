from torch.optim.lr_scheduler import _LRScheduler

class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter=None, decay_iter=1, gamma=0.9, last_epoch=-1, T_max=None, **kwargs):
        # accept cosine-style alias T_max for compatibility
        if max_iter is None and T_max is not None:
            max_iter = T_max
        if max_iter is None:
            raise ValueError("PolynomialLR requires max_iter (or T_max as alias).")

        self.decay_iter = decay_iter
        self.max_iter = int(max_iter)
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 - (self.last_epoch) / float(self.max_iter)) ** self.gamma
        return [base_lr * factor for base_lr in self.base_lrs]


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, after_scheduler,
                 mode="linear", warmup_iters=1000, gamma=0.0, last_epoch=-1):
        """
        mode: "linear" or "constant"
        warmup_iters: number of *iterations* (optimizer steps) to warm up
        gamma: starting factor (0.0 starts from 0; 0.1 starts from 10% of base LR)
        """
        self.mode = mode
        self.after_scheduler = after_scheduler
        self.warmup_iters = int(warmup_iters)
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch  
        # DURING WARMUP: scale base_lrs directly; do NOT query inner scheduler
        if t <= self.warmup_iters:
            W = float(self.warmup_iters) if self.warmup_iters > 0 else 1.0
            if self.mode == "linear":
                fac = self.gamma + (1.0 - self.gamma) * (t / W)
            elif self.mode == "constant":
                fac = self.gamma
            else:
                raise KeyError(f"WarmUp mode {self.mode} not implemented")
            return [b * fac for b in self.base_lrs]

        # AFTER WARMUP: hand off to base scheduler with shifted step
        inner_t = t - self.warmup_iters
        self.after_scheduler.base_lrs = self.base_lrs
        self.after_scheduler.last_epoch = inner_t
        return self.after_scheduler.get_lr()
