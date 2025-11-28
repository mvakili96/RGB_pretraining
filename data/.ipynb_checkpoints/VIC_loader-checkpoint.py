from typing import Sequence
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

class UnlabeledImages(Dataset):
    IMG_EXTS: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, root: str, transform):
        self.root = Path(root)
        self.paths = [p for p in self.root.rglob("*") if p.suffix.lower() in self.IMG_EXTS]
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.root}")
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        x1, x2 = self.transform(img)                      
        return x1, x2