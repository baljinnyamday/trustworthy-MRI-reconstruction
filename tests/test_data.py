"""Tests for FastMRI dataset and data utilities."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.mri.data import FastMRIDataset, fft2c, ifft2c, split_by_volume


DATASET_DIR = Path("dataset/fastmri_pd/train/h5")


def test_fft_ifft_roundtrip() -> None:
    image = torch.randn(1, 64, 64)
    reconstructed = torch.abs(ifft2c(fft2c(image)))
    torch.testing.assert_close(reconstructed, image.abs(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="Dataset not available")
def test_dataset_item_shapes() -> None:
    paths = sorted(DATASET_DIR.glob("*.h5"))[:5]
    ds = FastMRIDataset(paths, acceleration=4, center_fraction=0.08, seed=42)

    item = ds[0]
    assert item["input"].shape == (1, 320, 320)
    assert item["target"].shape == (1, 320, 320)
    assert item["mask"].shape == (1, 320)
    assert item["kspace"].shape == (1, 320, 320)
    assert item["input"].dtype == torch.float32
    assert item["target"].dtype == torch.float32
    assert item["kspace"].is_complex()


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="Dataset not available")
def test_dataset_deterministic() -> None:
    paths = sorted(DATASET_DIR.glob("*.h5"))[:3]
    ds1 = FastMRIDataset(paths, acceleration=4, center_fraction=0.08, seed=42)
    ds2 = FastMRIDataset(paths, acceleration=4, center_fraction=0.08, seed=42)
    torch.testing.assert_close(ds1[0]["input"], ds2[0]["input"])


def test_split_by_volume() -> None:
    # Create fake paths with 4 volumes, 3 slices each
    paths = [
        Path(f"vol{v:03d}_{s:03d}.h5")
        for v in range(4)
        for s in range(3)
    ]
    cal, test = split_by_volume(paths, seed=42, cal_fraction=0.5)

    # Should split into 2 volumes each
    cal_vols = {p.stem.rsplit("_", 1)[0] for p in cal}
    test_vols = {p.stem.rsplit("_", 1)[0] for p in test}

    assert len(cal_vols) == 2
    assert len(test_vols) == 2
    assert cal_vols.isdisjoint(test_vols), "No volume should appear in both splits"
    assert len(cal) == 6  # 2 volumes * 3 slices
    assert len(test) == 6
