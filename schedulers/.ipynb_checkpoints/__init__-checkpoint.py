# schedulers/__init__.py
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR
from .schedulers import WarmUpLR, ConstantLR, PolynomialLR

key2scheduler = {
    "constant": ConstantLR,
    "poly": PolynomialLR,
    "multi_step": MultiStepLR,
    "cosine_annealing": CosineAnnealingLR,
    "exp": ExponentialLR,
}

def get_scheduler(optimizer, scheduler_dict):
    if scheduler_dict is None:
        return ConstantLR(optimizer)

    s_type = scheduler_dict.pop("name")
    warmup_iters = int(scheduler_dict.pop("warmup_iters", 0))
    warmup_mode  = scheduler_dict.pop("warmup_mode", "linear")
    warmup_factor = float(scheduler_dict.pop("warmup_factor", 0.0))  # start at 0

    # If cosine, reduce horizon by warmup
    if s_type == "cosine_annealing" and "T_max" in scheduler_dict and warmup_iters > 0:
        scheduler_dict["T_max"] = max(1, int(scheduler_dict["T_max"]) - warmup_iters)

    base = key2scheduler[s_type](optimizer, **scheduler_dict)

    if warmup_iters > 0:
        return WarmUpLR(optimizer, base, mode=warmup_mode, warmup_iters=warmup_iters, gamma=warmup_factor)
    return base
