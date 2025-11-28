import os
import logging
import datetime
import torch
from collections import OrderedDict
import numpy as np

def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def get_state_dict(model: torch.nn.Module):
    # Works for plain, DataParallel, or DDP
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()

def save_checkpoint(args, step, model, extra=None):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    payload = {"state_dict": get_state_dict(model), "step": int(step)}
    if extra:
        payload.update(extra)

    # Write atomically: tmp -> rename
    tmp_path = os.path.join(args.ckpt_dir, f"{args.arch}_step{step:06d}.pth.tmp")
    final_path = tmp_path[:-4]  # strip ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)

    # Also (over)write a rolling "latest.pth"
    latest_tmp = os.path.join(args.ckpt_dir, "latest.pth.tmp")
    latest = latest_tmp[:-4]
    torch.save(payload, latest_tmp)
    os.replace(latest_tmp, latest)

    # Optional: prune old files to keep only last K
    if args.keep_last_k > 0:
        import re, glob
        pat = re.compile(rf"{re.escape(args.arch)}_step(\d+)\.pth$")
        files = sorted(
            [p for p in glob.glob(os.path.join(args.ckpt_dir, f"{args.arch}_step*.pth"))
             if pat.search(os.path.basename(p))],
            key=lambda p: int(pat.search(os.path.basename(p)).group(1))
        )
        for p in files[:-args.keep_last_k]:
            try: os.remove(p)
            except OSError: pass

    return final_path

def tensor_to_image_uint8(t: torch.Tensor, mean, std):

    t = t.detach().cpu().float()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    t = t.clamp(0.0, 1.0)
    return (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

def make_mim_sample_from_tensor(img_tensor: torch.Tensor, patch_size: int, mask_ratio: float, replace_value: float = 0.0):

    assert img_tensor.ndim == 3, "expect [C,H,W]"
    C, H, W = img_tensor.shape
    assert H % patch_size == 0 and W % patch_size == 0, "image_size must be divisible by patch_size"

    Hp, Wp = H // patch_size, W // patch_size
    N = Hp * Wp
    k = max(1, int(round(mask_ratio * N)))

    # choose k patch indices to mask
    idx = torch.randperm(N)[:k]
    mask_flat = torch.zeros(N, dtype=torch.bool)
    mask_flat[idx] = True

    # reshape to grid and upsample to pixel mask
    mask_grid = mask_flat.view(Hp, Wp).to(img_tensor.device)
    mask_pix = mask_grid.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)  # [H,W]
    mask_pix = mask_pix.unsqueeze(0)  # [1,H,W] for broadcast over channels

    # create masked image (replace with 0 -> equals channel mean after normalization)
    img_m = img_tensor.clone()
    img_m = torch.where(mask_pix, torch.full_like(img_m, replace_value), img_m)

    return {"image": img_tensor, "masked_image": img_m, "mask": mask_flat}