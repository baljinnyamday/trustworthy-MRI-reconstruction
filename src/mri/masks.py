"""K-space undersampling mask generation."""

import numpy as np
import torch


def create_cartesian_mask(
    width: int,
    acceleration: int,
    center_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a 1D Cartesian undersampling mask.

    Always keeps center lines for low-frequency information.
    Randomly samples remaining lines to reach the target acceleration.

    Args:
        width: Number of k-space columns (e.g. 320).
        acceleration: Acceleration factor (e.g. 4 or 8).
        center_fraction: Fraction of center lines to always keep.
        rng: NumPy random generator for reproducibility.

    Returns:
        Boolean array of shape (width,). True = acquired line.
    """
    mask = np.zeros(width, dtype=bool)

    # Always keep center lines
    num_center = int(center_fraction * width)
    center_start = (width - num_center) // 2
    mask[center_start : center_start + num_center] = True

    # Randomly sample remaining lines to reach target count
    total_lines = width // acceleration
    remaining = total_lines - num_center
    if remaining > 0:
        available = np.where(~mask)[0]
        chosen = rng.choice(available, size=remaining, replace=False)
        mask[chosen] = True

    return mask


def apply_mask(kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out k-space at locations where mask is False.

    Args:
        kspace: Complex k-space tensor of shape (..., H, W).
        mask: Boolean tensor of shape (W,) or (1, W), broadcastable.

    Returns:
        Masked k-space (same shape as input).
    """
    return kspace * mask
