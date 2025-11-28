import argparse
import numpy as np
import torch
import random
from pprint import pformat
from torch.utils.data import DataLoader
import os, time
from torch.cuda.amp import autocast, GradScaler

from models import get_model
from optimizers import get_optimizer
from schedulers import get_scheduler
from data.augmentation import image_transform,TwoCrops
from data.VIC_loader import UnlabeledImages
from losses.VICreg_loss import vicreg_loss
from utils import get_logger, save_checkpoint


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Rail Scene Network Pretraining with VICReg Style")
    parser.add_argument("--arch", default="TRIT-Net-Encoder", type=str)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--dir_dataset", default="/run/determined/workdir/nas2/Mohammadjavad/jpgs", type=str)
    parser.add_argument("--dir_log", default="./", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_iters", default=90000, type=int)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--optimizer", default="lars", type=str)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0.0005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--scheduler", default="cosine_annealing", type=str)
    parser.add_argument("--warmup_iters", default=6000, type=int)

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

    parser.add_argument("--image_size",  default=224, type=int)
    parser.add_argument("--num_workers", default=4,   type=int)
    parser.add_argument("--log_interval", default=100, type=int)

    parser.add_argument("--sim_weight", default=5.0, type=float)
    parser.add_argument("--var_weight", default=5.0, type=float)
    parser.add_argument("--cov_weight", default=0.01, type=float)

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


    base_lr = 0.2 * (args.batch_size * args.accum / 256.0) * 4 
    wd = 1e-6 if args.optimizer.lower() == "lars" else args.weight_decay

    enc_decay, enc_nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            enc_nodecay.append(p)   # BN/bias -> no weight decay
        else:
            enc_decay.append(p)

    param_groups = [
        {"params": enc_decay,   "lr": base_lr, "weight_decay": wd},
        {"params": enc_nodecay, "lr": base_lr, "weight_decay": 0.0},
    ]
    opt_cls = get_optimizer(args.optimizer)
    if args.optimizer.lower() == "sgd":
        optimizer = opt_cls(param_groups, momentum=args.momentum)
    else:
        optimizer = opt_cls(param_groups, lr=base_lr, momentum=args.momentum, weight_decay=wd)


    logger.info("Using optimizer {}".format(optimizer))
    logger.info("Effective learning rate: %.6f", base_lr)   # not %d
    logger.info("Weight decay: %.2e", wd)


    logger.info("Batch size: %d ", args.batch_size)
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

    logger.info("Similarity loss weight: %.3f ", args.sim_weight)
    logger.info("Variance loss weight: %.3f ", args.var_weight)
    logger.info("Covariance loss weight: %.3f ", args.cov_weight)



    transform = TwoCrops(image_transform(args.image_size, args.RGB_mean, args.RGB_std))
    dataset   = UnlabeledImages(args.dir_dataset, transform)

    loader    = DataLoader(dataset,
                           batch_size=args.batch_size,
                           shuffle=True,
                           num_workers=args.num_workers,
                           pin_memory=True,
                           drop_last=True)
    
    scaler = GradScaler(enabled=args.amp)
    model.train()
    it = iter(loader)
    step, t0 = 0, time.time()

    while step < args.max_iters:
        optimizer.zero_grad(set_to_none=True)

        running = {"loss": 0.0, "inv": 0.0, "var": 0.0, "cov": 0.0, "std1": 0.0, "std2": 0.0}
        for _ in range(args.accum):
            try:
                x1, x2 = next(it)
            except StopIteration:
                it = iter(loader)
                x1, x2 = next(it)

            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            with autocast(enabled=args.amp):
                z1 = model(x1)   
                z2 = model(x2)
                loss, parts = vicreg_loss(z1, z2, sim_coeff=args.sim_weight, var_coeff=args.var_weight, cov_coeff=args.cov_weight)
            
            running["loss"] += loss.item()
            running["inv"]  += parts["inv"]
            running["var"]  += parts["var"]
            running["cov"]  += parts["cov"]
            running["std1"] += parts["std1_mean"]
            running["std2"] += parts["std2_mean"]

            scaler.scale(loss / args.accum).backward()
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        c_lr = scheduler.get_last_lr()          

        if (step + 1) % args.log_interval == 0:
            dt  = time.time() - t0
            fps = (args.log_interval * args.batch_size * args.accum) / max(dt, 1e-6)
            avg = {k: v / args.accum for k, v in running.items()}
            msg = (
                f"[{step+1}/{args.max_iters}] "
                f"loss={avg['loss']:.4f} inv={avg['inv']:.4f} var={avg['var']:.4f} cov={avg['cov']:.4f} "
                f"std1={avg['std1']:.4f} std2={avg['std2']:.4f} "
                f"lr={c_lr[0]:.6f} z1={tuple(z1.shape)} z2={tuple(z2.shape)}  ~{fps:.1f} img/s"
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















