"""Evaluation metrics for MRI reconstruction."""

import torch

from src.mri.losses import compute_ssim


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio.

    Args:
        pred: (B, C, H, W) predicted images.
        target: (B, C, H, W) target images.
        data_range: Dynamic range of the images.

    Returns:
        PSNR in dB (mean over batch).
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return float("inf")
    return float(10 * torch.log10(torch.tensor(data_range**2 / mse)).item())


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Structural Similarity Index.

    Args:
        pred: (B, C, H, W) predicted images.
        target: (B, C, H, W) target images.
        data_range: Dynamic range of the images.

    Returns:
        SSIM value (mean over batch).
    """
    return float(compute_ssim(pred, target, data_range=data_range).item())


def nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Normalized Mean Squared Error.

    Returns:
        NMSE = ||pred - target||^2 / ||target||^2
    """
    return float((torch.sum((pred - target) ** 2) / torch.sum(target**2)).item())
