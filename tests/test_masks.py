"""Tests for k-space mask generation."""

import numpy as np
import torch

from src.mri.masks import apply_mask, create_cartesian_mask


def test_mask_shape() -> None:
    rng = np.random.default_rng(42)
    mask = create_cartesian_mask(320, acceleration=4, center_fraction=0.08, rng=rng)
    assert mask.shape == (320,)
    assert mask.dtype == bool


def test_mask_line_count_4x() -> None:
    rng = np.random.default_rng(42)
    mask = create_cartesian_mask(320, acceleration=4, center_fraction=0.08, rng=rng)
    assert mask.sum() == 320 // 4  # 80 lines


def test_mask_line_count_8x() -> None:
    rng = np.random.default_rng(42)
    mask = create_cartesian_mask(320, acceleration=8, center_fraction=0.08, rng=rng)
    assert mask.sum() == 320 // 8  # 40 lines


def test_center_lines_always_kept() -> None:
    rng = np.random.default_rng(42)
    width = 320
    center_fraction = 0.08
    mask = create_cartesian_mask(width, acceleration=8, center_fraction=center_fraction, rng=rng)

    num_center = int(center_fraction * width)
    center_start = (width - num_center) // 2
    center_region = mask[center_start : center_start + num_center]
    assert center_region.all(), "All center lines must be kept"


def test_mask_deterministic() -> None:
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    mask1 = create_cartesian_mask(320, acceleration=4, center_fraction=0.08, rng=rng1)
    mask2 = create_cartesian_mask(320, acceleration=4, center_fraction=0.08, rng=rng2)
    np.testing.assert_array_equal(mask1, mask2)


def test_apply_mask() -> None:
    kspace = torch.ones(1, 320, 320, dtype=torch.complex64)
    mask = torch.zeros(320, dtype=torch.bool)
    mask[100:200] = True

    result = apply_mask(kspace, mask)
    assert result[:, :, 100:200].abs().sum() > 0
    assert result[:, :, :100].abs().sum() == 0
    assert result[:, :, 200:].abs().sum() == 0
