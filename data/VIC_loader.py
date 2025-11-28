# data/VIC_loader.py
import os
from typing import Sequence
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

IMG_EXTS: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

def _list_images(root: str):
    paths = []
    for dp, dirnames, files in os.walk(root):
        # prune Synology thumbnail dirs so we never descend into them
        dirnames[:] = [d for d in dirnames if d.lower() != "@eadir"]
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                p = os.path.join(dp, f)
                # keep only real files (skip any weird directory that looks like a file)
                if os.path.isfile(p):
                    paths.append(p)
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return sorted(paths)

class UnlabeledImages(Dataset):
    def __init__(self, root: str, transform):
        self.paths = _list_images(root)
        # final sanity: assert everything is a file
        bad = [p for p in self.paths if not os.path.isfile(p)]
        if bad:
            raise RuntimeError(f"Non-file paths slipped in: e.g., {bad[:3]}")
        self.transform = transform

    def __len__(self): 
        return len(self.paths)

    def __getitem__(self, i):
        # small defensive retry loop in case a file disappears or is a directory
        for _ in range(3):
            p = self.paths[i]
            if not os.path.isfile(p):
                i = np.random.randint(0, len(self.paths))
                continue
            try:
                img = Image.open(p).convert("RGB")
                x1, x2 = self.transform(img)
                return x1, x2
            except IsADirectoryError:
                # rare NAS oddity — resample a different index
                i = np.random.randint(0, len(self.paths))
                continue
        raise RuntimeError(f"Failed to load a valid image after retries; last path: {p}")

