import argparse
import numpy as np
import torch
import random
from pprint import pformat
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler   # added for multi-GPU
import os, time
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn

import torch.distributed as dist                              # added for multi-GPU
from torch.nn.parallel import DistributedDataParallel as DDP  # added for multi-GPU

from models import get_model


from models import get_model
from optimizers import get_optimizer
from schedulers import get_scheduler
from data.augmentation import image_transform,TwoCrops
from data.VIC_loader import UnlabeledImages
from losses.VICreg_loss import vicreg_loss
from utils import get_logger, save_checkpoint

# added for multi-GPU
def setup_distributed():
    """
    Detect whether we're in distributed mode (torchrun or Slurm multi-GPU)
    and initialize torch.distributed if needed.

    Returns:
        is_dist (bool): are we in distributed mode?
        local_rank (int): index of GPU on this node
        global_rank (int): rank among all processes
        world_size (int): total number of processes
        device (torch.device): CUDA device for this process
    """
    # If RANK is not set, assume single-process / single-GPU
    if "RANK" not in os.environ:
        is_dist = False
        local_rank = 0
        global_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return is_dist, local_rank, global_rank, world_size, device

    # torchrun sets these
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize the process group
    dist.init_process_group(backend="nccl")

    # Each process should use its own GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    is_dist = True
    return is_dist, local_rank, global_rank, world_size, device


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
    parser.add_argument("--learning_rate", default=0.8, type=float)
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
    
    torch.autograd.set_detect_anomaly(True)

    is_dist, local_rank, global_rank, world_size, device = setup_distributed()   # added for multi-GPU
    
    # added for multi-GPU
    if global_rank == 0:
        print(f"Distributed: {is_dist}, world_size={world_size}")

    model = get_model(args.arch)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    
    # added for multi-GPU to initialize the Lazy modules
    dummy = torch.randn(2, 3, args.image_size, args.image_size, device=device)
    with torch.no_grad():
        _ = model(dummy)

    # added for multi-GPU
    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}")


    base_lr = args.learning_rate * args.batch_size * args.accum * world_size / 256.0 
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
    
    # added for multi-GPU
    if is_dist:
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
            drop_last=True,
        )
        shuffle = False  # DataLoader must NOT shuffle when using a sampler
    else:
        train_sampler = None
        shuffle = True
    
    # added for multi-GPU
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    scaler = GradScaler(enabled=args.amp)
    model.train()
    it = iter(loader)
    step, t0 = 0, time.time()
    
    # added for multi-GPU: track an artificial "epoch" for the sampler
    epoch = 0

    while step < args.max_iters:
        optimizer.zero_grad(set_to_none=True)

        running = {"loss": 0.0, "inv": 0.0, "var": 0.0, "cov": 0.0, "std1": 0.0, "std2": 0.0}
        for _ in range(args.accum):
            try:
                x1, x2 = next(it)
            except StopIteration:
                # added for multi-GPU: advance epoch & reshuffle for DistributedSampler
                if is_dist and train_sampler is not None:
                    epoch += 1
                    train_sampler.set_epoch(epoch)
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
            
            # added for multi-GPU
            if (not is_dist) or (global_rank == 0):
                logger.info(msg)
                print(msg)

            t0 = time.time()
        
        if (step + 1) % args.save_every == 0:
            # added for multi-GPU
            if (not is_dist) or (global_rank == 0):
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















