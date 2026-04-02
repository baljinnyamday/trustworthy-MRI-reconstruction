"""Tests for U-Net architecture."""

import torch
import torch.nn as nn

from src.mri.unet import UNet


def test_forward_shape() -> None:
    model = UNet(in_channels=1, out_channels=1, features=(32, 64, 128, 256))
    x = torch.randn(2, 1, 320, 320)
    out = model(x)
    assert out.shape == (2, 1, 320, 320)


def test_forward_small() -> None:
    model = UNet(in_channels=1, out_channels=1, features=(8, 16))
    x = torch.randn(1, 1, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 64, 64)


def test_enable_mc_dropout() -> None:
    model = UNet(features=(8, 16), dropout_rate=0.1)
    model.eval()

    # After eval(), dropout should be in eval mode
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            assert not m.training

    # After enable_mc_dropout(), dropout should be in train mode
    model.enable_mc_dropout()
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            assert m.training


def test_mc_dropout_produces_variance() -> None:
    model = UNet(features=(8, 16), dropout_rate=0.3)
    model.eval()
    model.enable_mc_dropout()

    x = torch.randn(1, 1, 64, 64)
    outputs = torch.stack([model(x) for _ in range(10)])
    variance = outputs.var(dim=0)
    assert variance.mean() > 0, "MC Dropout should produce non-zero variance"
