"""FastMRI dataset loading and k-space simulation pipeline."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.mri.config import TrainConfig
from src.mri.masks import create_cartesian_mask


def fft2c(image: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT: image domain -> k-space."""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image)))


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    """Centered 2D IFFT: k-space -> image domain."""
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace)))


class FastMRIDataset(Dataset):
    """Lazy-loading dataset for fastMRI PD knee slices.

    Simulates undersampled MRI acquisition by:
    1. Loading ground truth RSS image
    2. Computing full k-space via FFT
    3. Applying a Cartesian mask (stochastic during training, fixed during eval)
    4. Computing zero-filled reconstruction via IFFT

    Each item returns a dict with:
        input: (1, 320, 320) zero-filled reconstruction (normalized)
        target: (1, 320, 320) ground truth RSS (normalized)
        mask: (1, 320) undersampling mask (bool)
        kspace: (1, 320, 320) masked k-space (complex64)
        mean: float, normalization mean
        std: float, normalization std
        fname: str, filename
    """

    def __init__(
        self,
        file_paths: list[Path],
        acceleration: int,
        center_fraction: float,
        seed: int = 42,
        augment: bool = False,
    ) -> None:
        self.file_paths = sorted(file_paths)
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.seed = seed
        self.augment = augment
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update epoch for stochastic mask generation during training."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | float]:
        h5_path = self.file_paths[idx]

        # Load ground truth
        with h5py.File(h5_path, "r") as f:
            target_np = np.array(f["image_rss"][:], dtype=np.float32)

        # Normalize: zero-mean, unit-std (invertible)
        mean = float(target_np.mean())
        std = float(target_np.std()) + 1e-11
        target_norm = (target_np - mean) / std

        # Convert to tensor and compute k-space
        target_t = torch.from_numpy(target_norm).unsqueeze(0)  # (1, H, W)
        kspace = fft2c(target_t)  # (1, H, W) complex

        # Mask: stochastic per-epoch during training, deterministic during eval
        if self.augment:
            rng = np.random.default_rng(self.seed + idx + self._epoch * len(self))
        else:
            rng = np.random.default_rng(self.seed + idx)

        mask_np = create_cartesian_mask(
            width=target_np.shape[1],
            acceleration=self.acceleration,
            center_fraction=self.center_fraction,
            rng=rng,
        )
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)  # (1, W)

        # Apply mask and compute zero-filled reconstruction
        undersampled_kspace = kspace * mask_t.unsqueeze(1)  # (1, H, W)
        zf_image = torch.abs(ifft2c(undersampled_kspace))  # (1, H, W)

        # Random flips during training (applied consistently to input + target)
        if self.augment and rng.random() > 0.5:
            target_t = torch.flip(target_t, [-1])
            zf_image = torch.flip(zf_image, [-1])
            undersampled_kspace = fft2c(target_t) * mask_t.unsqueeze(1)

        return {
            "input": zf_image.float(),
            "target": target_t.float(),
            "mask": mask_t,
            "kspace": undersampled_kspace,
            "mean": mean,
            "std": std,
            "fname": h5_path.name,
        }


def get_file_paths(h5_dir: Path) -> list[Path]:
    """Get sorted list of HDF5 file paths from a directory."""
    return sorted(h5_dir.glob("*.h5"))


def split_by_volume(
    file_paths: list[Path],
    seed: int = 42,
    cal_fraction: float = 0.5,
) -> tuple[list[Path], list[Path]]:
    """Split file paths into calibration and test sets at the volume level.

    Ensures no data leakage: all slices from a volume go to the same split.

    Returns:
        (calibration_paths, test_paths)
    """
    # Group by volume name
    volumes: dict[str, list[Path]] = {}
    for p in file_paths:
        vol_name = p.stem.rsplit("_", 1)[0]
        volumes.setdefault(vol_name, []).append(p)

    # Shuffle volumes deterministically
    vol_names = sorted(volumes.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(vol_names)

    # Split
    n_cal = int(len(vol_names) * cal_fraction)
    cal_vols = vol_names[:n_cal]
    test_vols = vol_names[n_cal:]

    cal_paths = [p for v in cal_vols for p in sorted(volumes[v])]
    test_paths = [p for v in test_vols for p in sorted(volumes[v])]

    return cal_paths, test_paths


def build_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders."""
    train_paths = get_file_paths(config.data_root / "train" / "h5")
    val_paths = get_file_paths(config.data_root / "val" / "h5")

    train_ds = FastMRIDataset(
        file_paths=train_paths,
        acceleration=config.acceleration,
        center_fraction=config.center_fraction,
        seed=config.seed,
        augment=True,
    )
    val_ds = FastMRIDataset(
        file_paths=val_paths,
        acceleration=config.acceleration,
        center_fraction=config.center_fraction,
        seed=config.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
