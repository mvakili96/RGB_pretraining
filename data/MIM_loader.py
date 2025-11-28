import os
import math
import random
from typing import List, Tuple, Optional, Dict
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def _list_images(root: str) -> List[str]:
    paths = []
    for dp, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(dp, f))
    if not paths:
        raise RuntimeError(f"No images found under: {root}")
    return sorted(paths)

class RandomMaskingGenerator:
    """
    MAE-style random masking over a fixed patch grid.
    Produces a boolean mask of shape (num_patches,) where True means 'masked'.
    """
    def __init__(self, num_patches: int, mask_ratio: float = 0.75):
        assert 0.0 < mask_ratio < 1.0
        self.N = num_patches
        self.M = int(round(mask_ratio * num_patches))

    def __call__(self) -> Tensor:
        idx = torch.randperm(self.N)
        mask = torch.zeros(self.N, dtype=torch.bool)
        mask[idx[:self.M]] = True  # first M are masked
        return mask

def apply_patch_mask_replace(
    img: Tensor,
    mask_flat: Tensor,
    patch_size: int,
    fill: Optional[Tensor] = None,
) -> Tensor:
    
    assert img.dim() == 3
    C, H, W = img.shape

    Gh = H // patch_size
    Gw = W // patch_size
    assert Gh * patch_size == H and Gw * patch_size == W, \
        "H and W must be divisible by patch_size"
    assert mask_flat.numel() == Gh * Gw, \
        f"Mask size {mask_flat.numel()} != Gh*Gw = {Gh*Gw}"

    if fill is None:
        fill = torch.zeros(C, device=img.device, dtype=img.dtype)

    out = img.clone()
    mask = mask_flat.view(Gh, Gw)
    ys, xs = torch.where(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0, y1 = y * patch_size, (y + 1) * patch_size
        x0, x1 = x * patch_size, (x + 1) * patch_size
        out[:, y0:y1, x0:x1] = fill.view(C, 1, 1)
    return out

class MIMImages(Dataset):
    """
    Masked-Image-Modeling dataset.
    Output images are H = 0.5*image_size, W = image_size from the transform.
    """
    def __init__(
        self,
        root: str,
        transform,                     # your updated image_transform(...)
        image_size: int = 224,         # interpreted as target WIDTH
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        mask_mode: str = "replace",
        replace_with: str = "mean",
        rgb_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    ):
        super().__init__()
        self.paths = _list_images(root)
        self.transform = transform
        self.image_size = image_size     # target W
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.replace_with = replace_with

        # Target height/width from the transform contract
        self.target_h = int(0.5 * image_size)
        self.target_w = int(image_size)

        # Patch grid (rectangular)
        Gh = self.target_h // patch_size
        Gw = self.target_w // patch_size
        assert Gh * patch_size == self.target_h and Gw * patch_size == self.target_w, \
            "0.5*image_size and image_size must be divisible by patch_size"

        self.Gh = Gh
        self.Gw = Gw
        self.num_patches = Gh * Gw
        self.masker = RandomMaskingGenerator(self.num_patches, mask_ratio)

        # Fill value in normalized space
        if replace_with == "mean":
            self.fill = torch.tensor(rgb_mean, dtype=torch.float32)
        elif replace_with == "zeros":
            self.fill = torch.zeros(3, dtype=torch.float32)
        else:
            raise ValueError("replace_with must be 'mean' or 'zeros'")

    def __len__(self):
        return len(self.paths)

    def _load(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        path = self.paths[idx]
        img = self._load(path)

        # Transform enforces H=0.5*image_size, W=image_size, normalized
        clean: Tensor = self.transform(img)  # (C,H,W)

        # (Optional) verify at runtime in debug phases
        # C, H, W = clean.shape
        # assert H == self.target_h and W == self.target_w

        # Build mask over rectangular patch grid
        mask_flat = self.masker()  # (N,) bool
        visible_idx = (~mask_flat).nonzero(as_tuple=False).view(-1)
        masked_idx  = (mask_flat).nonzero(as_tuple=False).view(-1)

        if self.mask_mode == "replace":
            masked_img = apply_patch_mask_replace(
                clean, mask_flat, self.patch_size, fill=self.fill.to(clean.device)
            )
        elif self.mask_mode == "none":
            masked_img = clean
        else:
            raise ValueError("mask_mode must be 'replace' or 'none'")

        sample = {
            "image": clean,
            "masked_image": masked_img,
            "mask": mask_flat,                 # (Gh*Gw,)
            "visible_idx": visible_idx.long(), # (Nv,)
            "masked_idx": masked_idx.long(),   # (Nm,)
            "meta": {
                "path": path,
                "Gh": self.Gh, "Gw": self.Gw,
                "target_h": self.target_h, "target_w": self.target_w
            }
        }
        return sample


def collate_mim(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    # stack everything that can be stacked; keep lists for variable lengths if needed
    out = {}
    for k in ("image", "masked_image"):
        out[k] = torch.stack([b[k] for b in batch], dim=0)  # (B,C,H,W)
    for k in ("mask",):
        out[k] = torch.stack([b[k] for b in batch], dim=0)  # (B,N)
    # variable-length idx tensors -> keep as list of tensors (your forward can handle per-sample gather)
    out["visible_idx"] = [b["visible_idx"] for b in batch]
    out["masked_idx"]  = [b["masked_idx"]  for b in batch]
    out["meta"] = [b["meta"] for b in batch]
    return out
