"""Training loop for MRI reconstruction U-Net."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.mri.config import TrainConfig
from src.mri.data import build_dataloaders
from src.mri.losses import ReconLoss
from src.mri.metrics import psnr, ssim
from src.mri.unet import UNet


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch. Returns average loss and metrics."""
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    n_batches = 0

    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        data_range = float((targets.max() - targets.min()).item())
        total_loss += loss.item()
        total_psnr += psnr(outputs.detach(), targets, data_range=data_range)
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "psnr": total_psnr / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Validate model. Returns average loss and metrics."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches = 0

    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        data_range = float((targets.max() - targets.min()).item())
        total_loss += loss.item()
        total_psnr += psnr(outputs, targets, data_range=data_range)
        total_ssim += ssim(outputs, targets, data_range=data_range)
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "psnr": total_psnr / n_batches,
        "ssim": total_ssim / n_batches,
    }


def train(config: TrainConfig) -> Path:
    """Full training loop. Returns path to best checkpoint.

    Saves best model by validation SSIM.
    """
    device = torch.device(config.device)
    print(f"Training on {device} | {config.acceleration}x acceleration")

    # Data
    train_loader, val_loader = build_dataloaders(config)
    print(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")

    # Model
    model = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        features=config.features,
        dropout_rate=config.dropout_rate,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optimizer, loss, scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = ReconLoss(ssim_weight=config.ssim_weight)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    # Checkpoint dir
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = config.checkpoint_dir / f"best_{config.acceleration}x.pt"
    best_ssim = 0.0

    for epoch in range(1, config.num_epochs + 1):
        train_loader.dataset.set_epoch(epoch)
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["ssim"])

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{config.num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val PSNR: {val_metrics['psnr']:.2f} dB | "
            f"Val SSIM: {val_metrics['ssim']:.4f} | "
            f"LR: {lr:.1e}"
        )

        if val_metrics["ssim"] > best_ssim:
            best_ssim = val_metrics["ssim"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "acceleration": config.acceleration,
                        "features": config.features,
                        "dropout_rate": config.dropout_rate,
                    },
                    "epoch": epoch,
                    "val_ssim": best_ssim,
                    "val_psnr": val_metrics["psnr"],
                },
                best_path,
            )
            print(f"  -> Saved best model (SSIM: {best_ssim:.4f})")

    print(f"Training complete. Best SSIM: {best_ssim:.4f}")
    return best_path
