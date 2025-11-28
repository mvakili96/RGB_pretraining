import argparse
import numpy as np
import torch
import random
import inspect
from pprint import pformat
from torch.utils.data import DataLoader
import os, time
from torch.cuda.amp import autocast, GradScaler

from models import get_model
from optimizers import get_optimizer
from schedulers import get_scheduler
from data.augmentation import image_transform
from data.MIM_loader import MIMImages, collate_mim
from losses.MIM_loss import masked_mse_loss
from utils import get_logger,save_checkpoint, tensor_to_image_uint8, make_mim_sample_from_tensor


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Rail Scene Network Pretraining using Masked Image Modeling")
    parser.add_argument("--arch", default="TRIT-Net", type=str)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--dir_dataset", default="/run/determined/workdir/nas2/Mohammadjavad/jpgs", type=str)
    parser.add_argument("--dir_log", default="./", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--max_iters", default=90000, type=int)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--optimizer", default="adamw", type=str)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0.0005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--scheduler", default="poly", type=str)
    parser.add_argument("--warmup_iters", default=1000, type=int)

    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--amp", action="store_true", help="mixed precision")

    parser.add_argument(
        "--RGB-mean", nargs=3, type=float,
        default=[113.95/255.0, 118.05/255.0, 110.18/255.0],
        metavar=("R","G","B"),
        help="Per-channel mean"
    )
    parser.add_argument(
        "--RGB-std", nargs=3, type=float,
        default=[78.37/255.0, 68.79/255.0, 65.80/255.0],
        metavar=("R","G","B"),
        help="Per-channel std"
    )

    parser.add_argument("--image_size",  default=448, type=int)
    parser.add_argument("--patch_size",  default=16, type=int)
    parser.add_argument("--mask_ratio",  default=0.75, type=float)

    parser.add_argument("--num_workers", default=4,   type=int)
    parser.add_argument("--log_interval", default=100, type=int)
    
    parser.add_argument("--save_every", default=1000, type=int, help="save every N iterations")
    parser.add_argument("--ckpt_dir",   default="./", type=str, help="where to write .pth files")
    parser.add_argument("--keep_last_k", default=3, type=int, help="how many rolling checkpoints to keep (0 = keep all)")

    return parser.parse_args(args)

 
def train(args,logger):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.arch)          
    model.to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}")
        
  

    # --- param groups: decay / no-decay ---
    enc_decay, enc_nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            enc_nodecay.append(p)   # BN/LayerNorm/bias -> no weight decay
        else:
            enc_decay.append(p)

    param_groups = [
        {"params": enc_decay,   "lr": args.learning_rate, "weight_decay": getattr(args, "weight_decay", 0.0)},
        {"params": enc_nodecay, "lr": args.learning_rate, "weight_decay": 0.0},
    ]

    # --- pick optimizer class ---
    opt_cls = get_optimizer(args.optimizer.lower())

    # Common knobs (filtered by signature below)
    common_kwargs = {
        "lr": args.learning_rate,
        "weight_decay": getattr(args, "weight_decay", 0.0),
        "momentum": getattr(args, "momentum", 0.9),  # used by SGD/RMSprop/LARS; filtered out for Adam/AdamW/...
    }

    # Optional knobs (only passed if supported by the chosen optimizer)
    maybe_kwargs = {
        # Adam / Adamax / AdamW
        "betas":     getattr(args, "betas", (0.9, 0.999)),
        "eps":       getattr(args, "eps", 1e-8),
        "amsgrad":   getattr(args, "amsgrad", False),
        "maximize":  getattr(args, "maximize", False),
        # Some PyTorch builds support fused Adam/AdamW
        "fused":     getattr(args, "fused", None),

        # SGD
        "nesterov":  getattr(args, "nesterov", False),

        # RMSprop
        "alpha":     getattr(args, "alpha", 0.99),
        "centered":  getattr(args, "centered", False),

        # ASGD
        "lambd":     getattr(args, "lambd", 1e-4),
        "t0":        getattr(args, "t0", 1e6),

        # PyTorch perf flags
        "foreach":    getattr(args, "foreach", None),
        "capturable": getattr(args, "capturable", None),

        # LARS-specific extras (harmlessly filtered for other opts)
        "eta":       getattr(args, "lars_eta", 0.001),
        "eps":       getattr(args, "lars_eps", 1e-9),  # note: filtered if not in signature
    }

    # Filter kwargs by optimizer __init__ signature
    sig = inspect.signature(opt_cls.__init__)
    accepted = {k: v for k, v in common_kwargs.items() if k in sig.parameters}
    accepted.update({k: v for k, v in maybe_kwargs.items() if (k in sig.parameters and v is not None)})

    # Build optimizer
    optimizer = opt_cls(param_groups, **accepted)



    
    optimizer_params = {
    "lr": args.learning_rate,
    "weight_decay": args.weight_decay,
    "momentum": args.momentum
    }
    logger.info("Optimizer params:\n%s", pformat(optimizer_params, indent=2, sort_dicts=True))
    logger.info("Using optimizer {}".format(optimizer))

    logger.info("Batch size: %d ", args.batch_size)
    logger.info("Patch size: %d ", args.patch_size)
    logger.info("Masking ratio: %d ", args.mask_ratio)

    logger.info("Gradient accumulation steps: %d ", args.accum)

    scheduler_params = {
    "name": args.scheduler,
    "T_max": args.max_iters,          
    "warmup_iters": args.warmup_iters,
    "warmup_mode": "linear",
    "warmup_factor": 0.0,             
    }
    scheduler = get_scheduler(optimizer, scheduler_params)
    
    logger.info("Scheduler params:\n%s", pformat(scheduler_params, indent=2, sort_dicts=True))
    logger.info("Using scheduler: %s", args.scheduler)

    transform = image_transform(args.image_size, args.RGB_mean, args.RGB_std, False)

    dataset = MIMImages(
        root=args.dir_dataset,
        transform=transform,
        image_size=args.image_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,         
        mask_mode="replace",      # pixel masking for conv front-ends; set "none" if you only use indices in the model
        replace_with="zeros",
        rgb_mean=tuple(args.RGB_mean),
    )

    loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_mim,
    )


    
    scaler = GradScaler(enabled=args.amp)
    model.train()
    it = iter(loader)
    step, t0 = 0, time.time()

    while step < args.max_iters:
        optimizer.zero_grad(set_to_none=True)

        running = {"loss": 0.0}
        for _ in range(args.accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            img      = batch["image"].to(device, non_blocking=True)         # (B,C,H,W)
            img_m    = batch["masked_image"].to(device, non_blocking=True)  # (B,C,H,W)
            mask_flat= batch["mask"].to(device, non_blocking=True)          # (B,N) bool

            with autocast(enabled=args.amp):
                pred = model(img_m)  # (B,C,H,W)

                # MIM objective: MSE over masked regions only
                loss = masked_mse_loss(pred, img, mask_flat, patch_size=args.patch_size)
            
            running["loss"] += loss.item()
            scaler.scale(loss / args.accum).backward()
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        c_lr = scheduler.get_lr()

        if (step + 1) % args.log_interval == 0:
            dt  = time.time() - t0
            fps = (args.log_interval * args.batch_size * args.accum) / max(dt, 1e-6)
            avg_loss = running["loss"] / args.accum
            msg = (
                f"[{step+1}/{args.max_iters}] "
                f"loss={avg_loss:.4f} "
                f"lr={float(c_lr[0]):.6f} pred={tuple(pred.shape)}  ~{fps:.1f} img/s"
            )
            logger.info(msg); print(msg); t0 = time.time()
            
        
        if (step + 1) % args.save_every == 0:
            path = save_checkpoint(
                args, step + 1, model,
            )
            logger.info(f"Saved checkpoint: {path}")

        step += 1 
    






if __name__ == "__main__":
    args = parse_args()

    logger = get_logger(args.dir_log)
    logger.info("Let's begin...")
    train(args,logger)







