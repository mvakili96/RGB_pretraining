import torch
import torch.nn.functional as F

def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float = 25.0,   # invariance weight
    var_coeff: float = 25.0,   # variance weight
    cov_coeff: float = 1.0,    # covariance weight
    eps: float = 1e-4
):
    """
    z1, z2: (B, D) projector outputs for two independent views of the same batch.
    Returns: total_loss, dict with per-term scalars.
    """
    assert z1.shape == z2.shape, f"Shapes must match, got {z1.shape} vs {z2.shape}"
    B, D = z1.shape
    if B < 2:
        raise ValueError("VICReg needs batch_size >= 2 for variance/covariance terms.")

    # 1) Invariance (MSE between pairs)
    inv = F.mse_loss(z1, z2)

    # 2) Variance (std >= 1 for each feature)
    std_z1 = torch.sqrt(z1.var(dim=0, unbiased=False) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0, unbiased=False) + eps)
    var = (F.relu(1.0 - std_z1).mean() + F.relu(1.0 - std_z2).mean())

    # Center features per branch for cov
    z1c = z1 - z1.mean(dim=0)
    z2c = z2 - z2.mean(dim=0)

    # 3) Covariance (suppress off-diagonals)
    cov_z1 = (z1c.T @ z1c) / (B - 1)   # (D, D)
    cov_z2 = (z2c.T @ z2c) / (B - 1)
    off_diag = ~torch.eye(D, dtype=torch.bool, device=z1.device)
    cov = (cov_z1[off_diag].pow(2).sum() / D) + (cov_z2[off_diag].pow(2).sum() / D)

    total = sim_coeff * inv + var_coeff * var + cov_coeff * cov
    return total, {"inv": float(inv.detach()), "var": float(var.detach()), "cov": float(cov.detach()), "std1_mean": std_z1.mean().item(), "std2_mean": std_z2.mean().item()}
