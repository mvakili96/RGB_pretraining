import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

class BottomHalfCrop:
    def __call__(self, img):
        # img is a PIL Image
        w, h = img.size
        return img.crop((0, h // 2, w, h))  # left, top, right, bottom

def image_transform(image_size: int, RBG_mean=[0.5, 0.5, 0.5], RGB_std=[0.5, 0.5, 0.5], is_augmented=True):
    target_h = int(0.5 * image_size)
    target_w = int(image_size)

    base = [
        BottomHalfCrop()
    ]

    if is_augmented:
        return T.Compose([
            *base,
            T.RandomResizedCrop(size=(target_h,target_w),scale=(0.08,1.0),ratio=(target_w/target_h, target_w/target_h),interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.5),
            T.RandomSolarize(threshold=128, p=0.1),  # torchvision>=0.12
            T.ToTensor(),
            T.Normalize(mean=RBG_mean, std=RGB_std),
        ])
    else:
        return T.Compose([
            *base,
            T.Resize((target_h, target_w), interpolation=InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=RBG_mean, std=RGB_std),
        ])


class TwoCrops:
    """Apply the same base transform twice, returning (view1, view2)."""
    def __init__(self, base_t): self.t = base_t
    def __call__(self, img):     return self.t(img), self.t(img)