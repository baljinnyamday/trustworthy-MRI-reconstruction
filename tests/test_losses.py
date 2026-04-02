"""Tests for loss functions."""

import torch

from src.mri.losses import ReconLoss, compute_ssim
from src.mri.metrics import nmse, psnr, ssim


def test_ssim_identical_images() -> None:
    x = torch.rand(2, 1, 64, 64)
    val = compute_ssim(x, x)
    assert val.item() > 0.99, f"SSIM of identical images should be ~1.0, got {val.item()}"


def test_ssim_different_images() -> None:
    x = torch.rand(2, 1, 64, 64)
    y = torch.rand(2, 1, 64, 64)
    val = compute_ssim(x, y)
    assert val.item() < 0.5, "SSIM of random images should be low"


def test_recon_loss_identical() -> None:
    loss_fn = ReconLoss(ssim_weight=0.5)
    x = torch.rand(2, 1, 64, 64)
    loss = loss_fn(x, x)
    assert loss.item() < 0.01, f"Loss of identical images should be ~0, got {loss.item()}"


def test_recon_loss_gradient() -> None:
    loss_fn = ReconLoss(ssim_weight=0.5)
    x = torch.rand(2, 1, 64, 64, requires_grad=True)
    y = torch.rand(2, 1, 64, 64)
    loss = loss_fn(x, y)
    loss.backward()
    assert x.grad is not None, "Loss should be differentiable"


def test_psnr_identical() -> None:
    x = torch.rand(2, 1, 64, 64)
    val = psnr(x, x)
    assert val == float("inf") or val > 50


def test_nmse_identical() -> None:
    x = torch.rand(2, 1, 64, 64)
    val = nmse(x, x)
    assert val < 1e-8


def test_ssim_metric() -> None:
    x = torch.rand(2, 1, 64, 64)
    val = ssim(x, x)
    assert val > 0.99
