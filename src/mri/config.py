"""Hyperparameter configuration for MRI reconstruction experiments."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration."""

    # Paths
    data_root: Path = Path("dataset/fastmri_pd")
    checkpoint_dir: Path = Path("outputs/checkpoints")

    # Data
    acceleration: int = 4
    center_fraction: float = 0.08
    seed: int = 42

    # Model
    in_channels: int = 1
    out_channels: int = 1
    features: tuple[int, ...] = (32, 64, 128, 256)
    dropout_rate: float = 0.05

    # Training
    batch_size: int = 16
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    ssim_weight: float = 0.5
    num_workers: int = 2
    device: str = "cuda"


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation and trustworthy analysis configuration."""

    # Paths
    checkpoint_path: Path = Path("outputs/checkpoints/best_4x.pt")
    data_root: Path = Path("dataset/fastmri_pd")
    output_dir: Path = Path("outputs")

    # Data
    acceleration: int = 4
    center_fraction: float = 0.08
    seed: int = 42

    # Conformal prediction
    alpha: float = 0.1  # 90% coverage

    # MC Dropout
    mc_samples: int = 20

    # Physics-informed conformal
    kspace_blend_weight: float = 0.5  # lambda for blending pixel + kspace scores

    device: str = "cuda"
