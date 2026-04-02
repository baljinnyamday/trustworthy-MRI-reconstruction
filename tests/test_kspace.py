"""Tests for k-space consistency analysis."""

import numpy as np
import torch

from src.mri.data import fft2c
from src.mri.kspace_consistency import (
    compute_kspace_residual,
    consistency_score,
    physics_informed_score,
)


def test_perfect_reconstruction_zero_residual() -> None:
    """If reconstruction matches ground truth, k-space residual should be ~0."""
    image = torch.randn(1, 1, 64, 64)
    kspace = fft2c(image)
    mask = torch.ones(1, 1, 64, dtype=torch.bool)

    residual = compute_kspace_residual(image, kspace, mask)
    assert residual.max() < 1e-5, "Perfect reconstruction should have zero residual"


def test_consistency_score_perfect() -> None:
    image = torch.randn(1, 1, 64, 64)
    kspace = fft2c(image)
    mask = torch.ones(1, 1, 64, dtype=torch.bool)

    score = consistency_score(image, kspace, mask)
    assert score < 1e-5, f"Perfect reconstruction should have score ~0, got {score}"


def test_corrupted_reconstruction_high_residual() -> None:
    """Corrupted reconstruction should have high k-space residual."""
    image = torch.randn(1, 1, 64, 64)
    kspace = fft2c(image)
    mask = torch.ones(1, 1, 64, dtype=torch.bool)

    corrupted = image + torch.randn_like(image) * 0.5
    score = consistency_score(corrupted, kspace, mask)
    assert score > 0.1, f"Corrupted reconstruction should have high score, got {score}"


def test_physics_informed_score_shape() -> None:
    pixel_res = np.random.rand(64, 64).astype(np.float32)
    kspace_res = np.random.rand(64, 64).astype(np.float32)
    blended = physics_informed_score(pixel_res, kspace_res, lam=0.5)
    assert blended.shape == (64, 64)
    assert blended.min() >= 0
    assert blended.max() <= 1.0 + 1e-6
