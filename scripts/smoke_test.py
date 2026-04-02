"""Fast end-to-end smoke test: train tiny model + run full trustworthy pipeline.

Validates the entire idea works before committing to a full Kaggle run.
Uses 50 train slices, 20 val slices, 3 epochs, tiny U-Net.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.mri.conformal import (
    compute_nonconformity_scores,
    compute_quantile,
    evaluate_coverage,
    predict_with_intervals,
)
from src.mri.data import FastMRIDataset, get_file_paths, split_by_volume
from src.mri.kspace_consistency import batch_consistency_analysis
from src.mri.losses import ReconLoss
from src.mri.metrics import psnr, ssim
from src.mri.train import train_one_epoch, validate
from src.mri.unet import UNet

DATA_ROOT = Path("dataset/fastmri_pd")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ACCELERATION = 4


def main() -> None:
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # --- Tiny dataset ---
    train_paths = get_file_paths(DATA_ROOT / "train" / "h5")[:50]
    val_paths = get_file_paths(DATA_ROOT / "val" / "h5")[:100]

    train_ds = FastMRIDataset(train_paths, ACCELERATION, center_fraction=0.08, seed=42)
    val_ds = FastMRIDataset(val_paths, ACCELERATION, center_fraction=0.08, seed=42)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=0)

    # --- Tiny model ---
    model = UNet(features=(16, 32), dropout_rate=0.1).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params (tiny)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = ReconLoss(ssim_weight=0.5)

    # --- Train 3 epochs ---
    print("\n--- Training (3 epochs) ---")
    for epoch in range(1, 4):
        t = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        v = validate(model, val_loader, criterion, DEVICE)
        print(f"  Epoch {epoch}: train_loss={t['loss']:.4f} | val_psnr={v['psnr']:.1f} dB | val_ssim={v['ssim']:.4f}")

    # --- Conformal prediction ---
    print("\n--- Conformal Prediction ---")
    cal_paths, test_paths = split_by_volume(val_paths, seed=42, cal_fraction=0.5)
    cal_ds = FastMRIDataset(cal_paths, ACCELERATION, center_fraction=0.08, seed=42)
    test_ds = FastMRIDataset(test_paths, ACCELERATION, center_fraction=0.08, seed=42)
    cal_loader = DataLoader(cal_ds, batch_size=4, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4, num_workers=0)

    scores = compute_nonconformity_scores(model, cal_loader, DEVICE)
    q_hat = compute_quantile(scores, alpha=0.1)
    print(f"  Quantile (90% coverage): {q_hat:.6f}")

    results = predict_with_intervals(model, test_loader, q_hat, DEVICE)
    metrics = evaluate_coverage(results)
    print(f"  Coverage: {metrics['coverage']:.2%} (target: 90%)")
    print(f"  Interval width: {metrics['mean_interval_width']:.6f}")

    coverage_ok = metrics["coverage"] >= 0.89
    print(f"  Coverage guarantee holds: {'YES' if coverage_ok else 'NO'}")

    # --- MC Dropout ---
    print("\n--- MC Dropout ---")
    model.eval()
    model.enable_mc_dropout()
    batch = next(iter(test_loader))
    inputs = batch["input"].to(DEVICE)
    targets = batch["target"].squeeze(1).numpy()

    mc_preds = torch.stack([model(inputs).squeeze(1).detach() for _ in range(10)])
    mc_mean = mc_preds.mean(dim=0).cpu().numpy()
    mc_var = mc_preds.var(dim=0).cpu().numpy()
    print(f"  Mean variance: {mc_var.mean():.6f}")
    print(f"  Variance range: [{mc_var.min():.6f}, {mc_var.max():.6f}]")
    has_variance = mc_var.mean() > 0
    print(f"  Produces uncertainty: {'YES' if has_variance else 'NO'}")

    # --- K-Space Consistency ---
    print("\n--- K-Space Consistency ---")
    model.eval()
    # Reset dropout to eval mode for consistency check
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout2d):
            m.eval()

    ks = batch_consistency_analysis(model, test_loader, DEVICE)
    print(f"  Mean consistency score: {ks['mean_score']:.6f}")
    print(f"  Residual-error correlation: {ks['residual_error_correlation']:.4f}")
    has_correlation = ks["residual_error_correlation"] > 0
    print(f"  Residual correlates with error: {'YES' if has_correlation else 'NO'}")

    # --- Summary ---
    print("\n" + "=" * 60)
    all_pass = coverage_ok and has_variance and has_correlation
    if all_pass:
        print("ALL CHECKS PASSED — idea validated, ready for full training!")
    else:
        print("SOME CHECKS FAILED — investigate before full training")
        if not coverage_ok:
            print("  - Conformal coverage below 90%")
        if not has_variance:
            print("  - MC Dropout not producing variance")
        if not has_correlation:
            print("  - K-space residual not correlating with error")


if __name__ == "__main__":
    main()
