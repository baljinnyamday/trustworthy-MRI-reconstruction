"""Loss functions for MRI reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
    """Create a 1D Gaussian window."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def _create_window(size: int, channels: int) -> torch.Tensor:
    """Create a 2D Gaussian window for SSIM."""
    w1d = _gaussian_window(size)
    w2d = w1d.unsqueeze(1) * w1d.unsqueeze(0)
    return w2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size).contiguous()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Compute SSIM between predicted and target images.

    Args:
        pred: (B, C, H, W) predicted images.
        target: (B, C, H, W) target images.
        window_size: Size of the Gaussian window.
        data_range: Dynamic range of the images.

    Returns:
        Scalar SSIM value (mean over batch).
    """
    channels = pred.shape[1]
    window = _create_window(window_size, channels).to(pred.device, pred.dtype)
    pad = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_target = F.conv2d(target, window, padding=pad, groups=channels)

    mu_pred_sq = mu_pred**2
    mu_target_sq = mu_target**2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred**2, window, padding=pad, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target**2, window, padding=pad, groups=channels) - mu_target_sq
    sigma_cross = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_cross

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_cross + c1) * (2 * sigma_cross + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )

    return ssim_map.mean()


class ReconLoss(nn.Module):
    """Combined L1 + SSIM loss for MRI reconstruction.

    loss = (1 - ssim_weight) * L1 + ssim_weight * (1 - SSIM)
    """

    def __init__(self, ssim_weight: float = 0.5) -> None:
        super().__init__()
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        ssim_val = compute_ssim(pred, target)
        return (1 - self.ssim_weight) * l1 + self.ssim_weight * (1 - ssim_val)
