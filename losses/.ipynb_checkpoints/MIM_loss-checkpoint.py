import torch

def mask_flat_to_spatial(mask_flat: torch.Tensor, patch_size: int, H: int, W: int) -> torch.Tensor:
    """
    mask_flat: (B, N) bool where True => masked. N == (H/ps)*(W/ps)
    returns: (B, 1, H, W) bool mask expanded per pixel
    """
    B, N = mask_flat.shape
    assert H % patch_size == 0 and W % patch_size == 0, "H/W must be divisible by patch_size"
    Gh, Gw = H // patch_size, W // patch_size
    assert N == Gh * Gw, f"mask length {N} != grid {Gh}x{Gw}"

    mask_grid = mask_flat.view(B, 1, Gh, Gw)           # (B,1,Gh,Gw)
    spatial = mask_grid.repeat_interleave(patch_size, dim=2)\
                        .repeat_interleave(patch_size, dim=3)   # (B,1,H,W)
    return spatial


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask_flat: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    pred/target: (B,C,H,W) in *normalized* space (same transform as input)
    mask_flat:   (B,N) bool (True => masked)
    Only masked regions contribute to loss (MAE-style).
    Returns scalar loss = sum((pred-target)^2 over masked pixels) / (#masked_pixels * C)
    """
    B, C, H, W = pred.shape
    print(B,C,H,W)
    print(gooz)
    spatial_mask = mask_flat_to_spatial(mask_flat, patch_size, H, W)  # (B,1,H,W) bool
    num_masked = spatial_mask.sum() * C  # count per-channel pixels

    # Avoid divide-by-zero if (pathological) no patches masked
    if num_masked.item() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    diff2 = (pred - target) ** 2  # (B,C,H,W)
    masked_diff2 = diff2 * spatial_mask  # broadcast over channel
    loss = masked_diff2.sum() / num_masked
    return loss
