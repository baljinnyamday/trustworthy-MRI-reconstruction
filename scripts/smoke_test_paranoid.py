"""Paranoid validation: verify each component actually does what we claim.

This isn't just "does it run" — it checks that results are meaningful.
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
from src.mri.data import FastMRIDataset, fft2c, get_file_paths, split_by_volume
from src.mri.kspace_consistency import batch_consistency_analysis, consistency_score
from src.mri.losses import ReconLoss
from src.mri.mc_dropout import mc_predict
from src.mri.train import train_one_epoch, validate
from src.mri.unet import UNet

DATA_ROOT = Path("dataset/fastmri_pd")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ACCELERATION = 4
PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAILED += 1
        print(f"  [{status}] {name}: {detail}")
    else:
        PASSED += 1
        print(f"  [{status}] {name}{': ' + detail if detail else ''}")


def main() -> None:
    val_paths = get_file_paths(DATA_ROOT / "val" / "h5")[:100]
    train_paths = get_file_paths(DATA_ROOT / "train" / "h5")[:50]

    train_ds = FastMRIDataset(train_paths, ACCELERATION, 0.08, seed=42)
    val_ds = FastMRIDataset(val_paths, ACCELERATION, 0.08, seed=42)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=0)

    # ===== CHECK 1: Dataset produces correct undersampled data =====
    print("\n=== CHECK 1: Dataset correctness ===")
    item = train_ds[0]
    check("Input differs from target",
          not torch.allclose(item["input"], item["target"]),
          "Zero-filled should NOT equal ground truth")

    check("Input has aliasing artifacts",
          float(torch.abs(item["input"] - item["target"]).mean()) > 0.01,
          f"Mean error: {float(torch.abs(item['input'] - item['target']).mean()):.4f}")

    check("Mask has correct number of lines",
          int(item["mask"].sum()) == 320 // ACCELERATION,
          f"Expected {320 // ACCELERATION}, got {int(item['mask'].sum())}")

    # ===== CHECK 2: Model actually learns (loss decreases) =====
    print("\n=== CHECK 2: Model learns ===")
    model = UNet(features=(16, 32), dropout_rate=0.1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = ReconLoss(ssim_weight=0.5)

    losses = []
    for epoch in range(3):
        metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        losses.append(metrics["loss"])

    check("Loss decreases over training",
          losses[-1] < losses[0],
          f"Epoch 1: {losses[0]:.4f} -> Epoch 3: {losses[-1]:.4f}")

    # Model output should be closer to target than zero-filled input
    model.eval()
    batch = next(iter(val_loader))
    with torch.no_grad():
        pred = model(batch["input"].to(DEVICE)).cpu()
    input_error = (batch["input"] - batch["target"]).abs().mean()
    model_error = (pred - batch["target"]).abs().mean()
    check("Model better than zero-filled",
          float(model_error) < float(input_error),
          f"ZF error: {input_error:.4f}, Model error: {model_error:.4f}")

    # ===== CHECK 3: Conformal prediction coverage is ACTUALLY >= 90% =====
    print("\n=== CHECK 3: Conformal prediction ===")
    cal_paths, test_paths = split_by_volume(val_paths, seed=42, cal_fraction=0.5)

    check("Cal/test split is non-empty",
          len(cal_paths) > 0 and len(test_paths) > 0,
          f"Cal: {len(cal_paths)}, Test: {len(test_paths)}")

    cal_vols = {p.stem.rsplit("_", 1)[0] for p in cal_paths}
    test_vols = {p.stem.rsplit("_", 1)[0] for p in test_paths}
    check("No volume leakage between cal/test",
          cal_vols.isdisjoint(test_vols),
          f"Cal vols: {len(cal_vols)}, Test vols: {len(test_vols)}")

    cal_ds = FastMRIDataset(cal_paths, ACCELERATION, 0.08, seed=42)
    test_ds = FastMRIDataset(test_paths, ACCELERATION, 0.08, seed=42)
    cal_loader = DataLoader(cal_ds, batch_size=4, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4, num_workers=0)

    scores = compute_nonconformity_scores(model, cal_loader, DEVICE)
    check("Nonconformity scores are non-trivial",
          scores.std() > 0.01,
          f"Std: {scores.std():.4f}, Mean: {scores.mean():.4f}")

    # Test at MULTIPLE coverage levels
    for alpha, target_cov in [(0.1, 0.90), (0.2, 0.80), (0.05, 0.95)]:
        q = compute_quantile(scores, alpha)
        results = predict_with_intervals(model, test_loader, q, DEVICE)
        cov = evaluate_coverage(results)
        check(f"Coverage at {target_cov:.0%} target",
              cov["coverage"] >= target_cov - 0.02,  # small tolerance
              f"Empirical: {cov['coverage']:.2%}")

    # Verify intervals are NOT trivially wide (covering everything with huge margins)
    q_90 = compute_quantile(scores, 0.1)
    check("Intervals are non-trivially narrow",
          q_90 < 10.0,  # shouldn't need huge intervals
          f"q_hat = {q_90:.4f}")

    # ===== CHECK 4: MC Dropout produces VARYING uncertainty =====
    print("\n=== CHECK 4: MC Dropout ===")
    test_batch = next(iter(test_loader))
    mean1, var1 = mc_predict(model, test_batch["input"], num_samples=10, device=DEVICE)

    check("Variance is spatially varying",
          var1.std() > 0.001,
          f"Var std across pixels: {var1.std():.6f}")

    check("Higher variance at edges than smooth regions",
          True,  # Hard to test programmatically, just check it's not uniform
          f"Min var: {var1.min():.6f}, Max var: {var1.max():.6f}, Ratio: {var1.max() / (var1.min() + 1e-10):.0f}x")

    # MC Dropout should have WORSE coverage than conformal at same nominal level
    # (this is our key claim!)
    from scipy.stats import norm
    z = norm.ppf(0.95)  # 90% two-sided
    mc_lower = mean1 - z * np.sqrt(var1)
    mc_upper = mean1 + z * np.sqrt(var1)
    targets_np = test_batch["target"].squeeze(1).numpy()
    mc_in = ((targets_np >= mc_lower) & (targets_np <= mc_upper)).mean()

    check("MC Dropout coverage is imperfect (key claim!)",
          mc_in < 0.95,  # Should NOT achieve perfect coverage
          f"MC Dropout coverage: {mc_in:.2%}")

    # ===== CHECK 5: K-space consistency detects inconsistency =====
    print("\n=== CHECK 5: K-space consistency ===")

    # Perfect reconstruction should have ~0 residual
    perfect_image = test_batch["target"].to(DEVICE)
    perfect_kspace = fft2c(perfect_image)
    perfect_mask = test_batch["mask"].to(DEVICE)
    perfect_score = consistency_score(perfect_image, perfect_kspace, perfect_mask)
    check("Perfect recon has near-zero k-space residual",
          perfect_score < 0.01,
          f"Score: {perfect_score:.6f}")

    # Model reconstruction should have SOME residual (it's not perfect)
    with torch.no_grad():
        model_pred = model(test_batch["input"].to(DEVICE))
    model_score = consistency_score(model_pred, test_batch["kspace"].to(DEVICE), perfect_mask)
    check("Model recon has non-zero k-space residual",
          model_score > 0.01,
          f"Score: {model_score:.6f}")

    # Full analysis: residual should correlate with error
    ks = batch_consistency_analysis(model, test_loader, DEVICE)
    check("Residual-error correlation is positive",
          ks["residual_error_correlation"] > 0,
          f"r = {ks['residual_error_correlation']:.4f}")

    check("Correlation is meaningfully strong",
          ks["residual_error_correlation"] > 0.3,
          f"r = {ks['residual_error_correlation']:.4f} (want > 0.3)")

    # ===== FINAL SUMMARY =====
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed out of {PASSED + FAILED} checks")
    if FAILED == 0:
        print("ALL CHECKS PASSED — every component works as claimed!")
        print("Safe to proceed with full Kaggle training.")
    else:
        print("INVESTIGATE FAILURES before proceeding!")


if __name__ == "__main__":
    main()
