"""K-space data consistency analysis for hallucination detection.

Measures how well a reconstruction agrees with acquired k-space measurements.
High residual at acquired locations = model contradicts measured physics = hallucination.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.mri.data import fft2c, ifft2c


def compute_kspace_residual(
    reconstruction: torch.Tensor,
    measured_kspace: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute k-space residual at acquired locations.

    Args:
        reconstruction: (B, 1, H, W) reconstructed images.
        measured_kspace: (B, 1, H, W) original k-space (complex).
        mask: (B, 1, W) acquisition mask.

    Returns:
        (B, 1, H, W) residual map in image domain (magnitude).
    """
    # Forward project reconstruction to k-space
    recon_kspace = fft2c(reconstruction)

    # Compute residual only at acquired locations
    mask_expanded = mask.unsqueeze(2).expand_as(recon_kspace)

    residual_kspace = (recon_kspace - measured_kspace) * mask_expanded

    # Transform residual back to image domain for visualization
    residual_image = torch.abs(ifft2c(residual_kspace))

    return residual_image


def consistency_score(
    reconstruction: torch.Tensor,
    measured_kspace: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Scalar k-space consistency metric.

    Returns:
        Normalized L2 norm of k-space residual at acquired locations.
    """
    recon_kspace = fft2c(reconstruction)
    mask_expanded = mask.unsqueeze(2).expand_as(recon_kspace)
    residual = (recon_kspace - measured_kspace) * mask_expanded
    res_norm = torch.abs(residual).square().sum().sqrt()
    ref_norm = torch.abs(measured_kspace * mask_expanded).square().sum().sqrt()
    return float((res_norm / (ref_norm + 1e-10)).item())


def physics_informed_score(
    pixel_residual: np.ndarray,
    kspace_residual: np.ndarray,
    lam: float = 0.5,
) -> np.ndarray:
    """Blend pixel-domain and k-space nonconformity scores.

    This is the novel physics-informed conformal score:
    score = (1 - lam) * |y - y_hat| + lam * kspace_residual

    Args:
        pixel_residual: (H, W) absolute pixel error.
        kspace_residual: (H, W) k-space consistency residual.
        lam: Blending weight (0 = pure pixel, 1 = pure physics).

    Returns:
        (H, W) blended nonconformity score.
    """
    # Normalize both to similar scales
    px_norm = pixel_residual / (pixel_residual.max() + 1e-10)
    ks_norm = kspace_residual / (kspace_residual.max() + 1e-10)
    return (1 - lam) * px_norm + lam * ks_norm


@torch.no_grad()
def batch_consistency_analysis(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    """Run k-space consistency analysis on a dataset.

    Returns:
        Dict with per-image consistency scores, example residual maps,
        and correlation between residual and actual error.
    """
    model.eval()
    all_scores: list[float] = []
    all_residuals: list[np.ndarray] = []
    all_errors: list[np.ndarray] = []

    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        kspace = batch["kspace"].to(device)
        mask = batch["mask"].to(device)

        preds = model(inputs)

        # K-space consistency residual
        residual = compute_kspace_residual(preds, kspace, mask)

        # Actual reconstruction error
        error = torch.abs(preds - targets)

        for i in range(preds.shape[0]):
            score = consistency_score(
                preds[i : i + 1], kspace[i : i + 1], mask[i : i + 1]
            )
            all_scores.append(score)
            all_residuals.append(residual[i, 0].cpu().numpy())
            all_errors.append(error[i, 0].cpu().numpy())

    scores_arr = np.array(all_scores)
    residuals_arr = np.stack(all_residuals)
    errors_arr = np.stack(all_errors)

    # Compute correlation between residual and error
    corr = np.corrcoef(residuals_arr.ravel(), errors_arr.ravel())[0, 1]

    return {
        "scores": scores_arr,
        "mean_score": float(scores_arr.mean()),
        "residual_error_correlation": float(corr),
        "example_residual": residuals_arr[0],
        "example_error": errors_arr[0],
    }
