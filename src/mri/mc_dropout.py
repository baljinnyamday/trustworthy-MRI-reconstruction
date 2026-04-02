"""MC Dropout uncertainty estimation for comparison with conformal prediction.

Demonstrates that MC Dropout intervals lack formal coverage guarantees.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.mri.unet import UNet


@torch.no_grad()
def mc_predict(
    model: UNet,
    inputs: torch.Tensor,
    num_samples: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model with dropout enabled N times to estimate uncertainty.

    Args:
        model: U-Net with dropout layers.
        inputs: Input tensor (B, 1, H, W).
        num_samples: Number of stochastic forward passes.
        device: Torch device.

    Returns:
        (mean, variance) arrays of shape (B, H, W).
    """
    model.eval()
    model.enable_mc_dropout()

    inputs = inputs.to(device)
    predictions = torch.stack([
        model(inputs).squeeze(1) for _ in range(num_samples)
    ])  # (N, B, H, W)

    mean = predictions.mean(dim=0).cpu().numpy()
    variance = predictions.var(dim=0).cpu().numpy()

    return mean, variance


def mc_coverage(
    mean: np.ndarray,
    variance: np.ndarray,
    target: np.ndarray,
    alpha: float = 0.1,
) -> dict[str, float]:
    """Compute coverage of MC Dropout Gaussian intervals.

    Interval: mean ± z * sqrt(variance), where z is the Gaussian quantile.

    Args:
        mean: (B, H, W) mean predictions.
        variance: (B, H, W) prediction variance.
        target: (B, H, W) ground truth.
        alpha: Miscoverage rate (e.g., 0.1 for 90% nominal coverage).

    Returns:
        Dict with 'coverage' and 'mean_interval_width'.
    """
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    std = np.sqrt(variance)

    lower = mean - z * std
    upper = mean + z * std

    in_interval = (target >= lower) & (target <= upper)

    return {
        "coverage": float(in_interval.mean()),
        "mean_interval_width": float((upper - lower).mean()),
    }


@torch.no_grad()
def mc_dropout_analysis(
    model: UNet,
    loader: DataLoader,
    num_samples: int,
    device: torch.device,
    alpha: float = 0.1,
) -> dict[str, float | np.ndarray]:
    """Run full MC Dropout analysis on a dataset.

    Returns:
        Dict with coverage, interval width, and example uncertainty maps.
    """
    all_means: list[np.ndarray] = []
    all_vars: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in loader:
        mean, variance = mc_predict(model, batch["input"], num_samples, device)
        targets = batch["target"].squeeze(1).numpy()
        all_means.append(mean)
        all_vars.append(variance)
        all_targets.append(targets)

    means = np.concatenate(all_means)
    variances = np.concatenate(all_vars)
    targets = np.concatenate(all_targets)

    metrics = mc_coverage(means, variances, targets, alpha=alpha)

    return {
        "coverage": metrics["coverage"],
        "mean_interval_width": metrics["mean_interval_width"],
        "example_mean": means[0],
        "example_variance": variances[0],
        "example_target": targets[0],
    }
