"""Run full trustworthy evaluation: save EVERYTHING for figure generation.

Optimized to minimize GPU forward passes — collect predictions once, reuse everywhere.
"""

from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from torch.utils.data import DataLoader

from src.mri.config import EvalConfig
from src.mri.conformal import (
    adaptive_calibration_from_arrays,
    adaptive_coverage_from_arrays,
    compute_adaptive_scores,
    compute_nonconformity_scores,
    compute_quantile,
    gradient_sigma,
    smooth_sigma_maps,
)
from src.mri.data import FastMRIDataset, fft2c, get_file_paths, ifft2c, split_by_volume
from src.mri.kspace_consistency import compute_kspace_residual, consistency_score
from src.mri.mc_dropout import mc_coverage, mc_predict
from src.mri.metrics import nmse, psnr, ssim
from src.mri.unet import UNet

N_EXAMPLES = 10


def load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    """Load a trained U-Net from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    model = UNet(
        features=tuple(cfg["features"]),
        dropout_rate=cfg["dropout_rate"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


@torch.no_grad()
def collect_all_data(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, list[np.ndarray] | list[float]]:
    """Single pass: collect predictions, inputs, targets, masks, kspace residuals."""
    model.eval()
    out: dict[str, list] = {
        "preds": [], "targets": [], "inputs": [], "masks": [],
        "errors": [], "ks_residuals": [], "ks_scores": [],
    }

    for i, batch in enumerate(loader):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        kspace = batch["kspace"].to(device)
        mask = batch["mask"].to(device)

        preds = model(inputs)
        error = torch.abs(preds - targets)
        residual = compute_kspace_residual(preds, kspace, mask)

        preds_np = preds.squeeze(1).cpu().numpy()
        targets_np = targets.squeeze(1).cpu().numpy()
        inputs_np = inputs.squeeze(1).cpu().numpy()
        masks_np = batch["mask"].squeeze(1).cpu().numpy()
        error_np = error.squeeze(1).cpu().numpy()
        residual_np = residual.squeeze(1).cpu().numpy()

        for j in range(preds_np.shape[0]):
            out["preds"].append(preds_np[j])
            out["targets"].append(targets_np[j])
            out["inputs"].append(inputs_np[j])
            out["masks"].append(masks_np[j])
            out["errors"].append(error_np[j])
            out["ks_residuals"].append(residual_np[j])
            score = consistency_score(
                preds[j:j+1], kspace[j:j+1], mask[j:j+1],
            )
            out["ks_scores"].append(score)

        if (i + 1) % 10 == 0:
            done = min((i + 1) * loader.batch_size, len(loader.dataset))
            print(f"    {done}/{len(loader.dataset)} images processed")

    return out


def compute_per_image_metrics(
    preds: list[np.ndarray],
    targets: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute PSNR, SSIM, NMSE for each image."""
    psnr_vals, ssim_vals, nmse_vals = [], [], []
    for p, t in zip(preds, targets, strict=True):
        pt = torch.from_numpy(p).unsqueeze(0).unsqueeze(0)
        tt = torch.from_numpy(t).unsqueeze(0).unsqueeze(0)
        data_range = float(tt.max() - tt.min())
        psnr_vals.append(psnr(pt, tt, data_range=data_range))
        ssim_vals.append(ssim(pt, tt, data_range=data_range))
        nmse_vals.append(nmse(pt, tt))
    return {
        "psnr": np.array(psnr_vals),
        "ssim": np.array(ssim_vals),
        "nmse": np.array(nmse_vals),
    }


def cp_coverage_from_arrays(
    preds: list[np.ndarray],
    targets: list[np.ndarray],
    q_hat: float,
) -> dict[str, float]:
    """Compute CP coverage from pre-collected arrays. No inference needed."""
    covered = 0
    total = 0
    for p, t in zip(preds, targets, strict=True):
        in_interval = (t >= p - q_hat) & (t <= p + q_hat)
        covered += in_interval.sum()
        total += t.size
    return {
        "coverage": float(covered / total),
        "mean_interval_width": 2.0 * q_hat,
    }


def cp_calibration_from_arrays(
    cal_scores: np.ndarray,
    preds: list[np.ndarray],
    targets: list[np.ndarray],
    alphas: list[float],
) -> dict[str, list[float]]:
    """Compute CP calibration curve without re-running inference."""
    nominal, empirical = [], []
    for alpha in alphas:
        q = compute_quantile(cal_scores, alpha)
        metrics = cp_coverage_from_arrays(preds, targets, q)
        nominal.append(1 - alpha)
        empirical.append(metrics["coverage"])
    return {"nominal": nominal, "empirical": empirical}


@torch.no_grad()
def collect_mc_data(
    model: UNet,
    loader: DataLoader,
    num_samples: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single MC Dropout pass over entire dataset. Returns (means, variances, targets)."""
    all_means, all_vars, all_targets = [], [], []

    for i, batch in enumerate(loader):
        mean, variance = mc_predict(model, batch["input"], num_samples, device)
        targets = batch["target"].squeeze(1).numpy()
        all_means.append(mean)
        all_vars.append(variance)
        all_targets.append(targets)

        if (i + 1) % 10 == 0:
            done = min((i + 1) * loader.batch_size, len(loader.dataset))
            print(f"    {done}/{len(loader.dataset)} MC samples collected")

    return (
        np.concatenate(all_means),
        np.concatenate(all_vars),
        np.concatenate(all_targets),
    )


def run_evaluation(config: EvalConfig) -> None:
    """Run full evaluation pipeline with minimal inference passes."""
    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    acc = config.acceleration

    print(f"\n{'='*60}")
    print(f"EVALUATING {acc}x ACCELERATION")
    print(f"{'='*60}")

    model = load_model(config.checkpoint_path, device)
    print(f"Loaded model from {config.checkpoint_path}")

    # Split val set
    val_paths = get_file_paths(config.data_root / "val" / "h5")
    cal_paths, test_paths = split_by_volume(val_paths, seed=config.seed)
    print(f"Calibration: {len(cal_paths)} | Test: {len(test_paths)}")

    cal_ds = FastMRIDataset(cal_paths, config.acceleration, config.center_fraction, config.seed)
    test_ds = FastMRIDataset(test_paths, config.acceleration, config.center_fraction, config.seed)
    cal_loader = DataLoader(cal_ds, batch_size=16, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=16, num_workers=2)

    # ===== PASS 1: Deterministic predictions + k-space (test set) =====
    print("\n[1/6] Collecting test predictions + k-space residuals...")
    data = collect_all_data(model, test_loader, device)
    n_test = len(data["preds"])
    print(f"  Done: {n_test} images")

    # ===== PASS 2: Calibration predictions + scores (cal set) =====
    print("\n[2/6] Collecting calibration predictions...")
    cal_data = collect_all_data(model, cal_loader, device)
    cal_scores = np.stack(cal_data["errors"])  # |pred - target| = nonconformity scores
    print(f"  Done: {cal_scores.shape[0]} cal images, {cal_scores.size} total scores")

    # ===== Metrics (no inference) =====
    print("\n[3/6] Computing metrics + conformal prediction...")
    img_metrics = compute_per_image_metrics(data["preds"], data["targets"])
    print(f"  PSNR: {img_metrics['psnr'].mean():.2f} +/- {img_metrics['psnr'].std():.2f} dB")
    print(f"  SSIM: {img_metrics['ssim'].mean():.4f} +/- {img_metrics['ssim'].std():.4f}")
    print(f"  NMSE: {img_metrics['nmse'].mean():.6f}")

    # Conformal prediction at multiple levels (no inference!)
    alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    q_90 = compute_quantile(cal_scores, 0.1)
    q_95 = compute_quantile(cal_scores, 0.05)
    q_80 = compute_quantile(cal_scores, 0.2)

    cp_90 = cp_coverage_from_arrays(data["preds"], data["targets"], q_90)
    print(f"  CP coverage (90%): {cp_90['coverage']:.4f}")
    print(f"  CP interval width: {cp_90['mean_interval_width']:.6f}")

    # Per-image coverage at 90%
    per_img_coverage = np.array([
        float(((t >= p - q_90) & (t <= p + q_90)).mean())
        for p, t in zip(data["preds"], data["targets"], strict=True)
    ])
    print(f"  Per-image coverage: {per_img_coverage.mean():.4f} +/- {per_img_coverage.std():.4f}")

    # Calibration curve (pure numpy, no inference)
    cp_cal = cp_calibration_from_arrays(cal_scores, data["preds"], data["targets"], alphas)
    print("  CP calibration:")
    for nom, emp in zip(cp_cal["nominal"], cp_cal["empirical"], strict=True):
        print(f"    {nom:.0%} -> {emp:.2%}")

    # K-space results (already collected)
    ks_scores = np.array(data["ks_scores"])
    ks_residuals = np.stack(data["ks_residuals"])
    ks_errors = np.stack([data["errors"][i] for i in range(n_test)])
    ks_corr = float(np.corrcoef(ks_residuals.ravel(), ks_errors.ravel())[0, 1])
    print(f"  KS mean score: {ks_scores.mean():.6f}")
    print(f"  KS residual-error correlation: {ks_corr:.4f}")

    # ===== MC Dropout: load from previous run or recompute =====
    prev_results = output_dir / f"results_{acc}x.npz"
    if prev_results.exists():
        print("\n[4/6] Loading MC Dropout results from previous run...")
        prev = dict(np.load(prev_results, allow_pickle=True))
        mc_90 = {
            "coverage": float(prev["mc_coverage"]),
            "mean_interval_width": float(prev["mc_interval_width"]),
        }
        mc_cal_nominal = prev["mc_cal_nominal"].tolist()
        mc_cal_empirical = prev["mc_cal_empirical"].tolist()
        mc_means = prev.get("mc_example_means", np.zeros((N_EXAMPLES, 320, 320)))
        mc_vars = prev.get("mc_example_variances", np.zeros((N_EXAMPLES, 320, 320)))
        print(f"  MC coverage (90%): {mc_90['coverage']:.4f}")
        print(f"  MC interval width: {mc_90['mean_interval_width']:.6f}")
    else:
        print("\n[4/6] MC Dropout on test set (20 passes)...")
        mc_means, mc_vars, mc_targets = collect_mc_data(
            model, test_loader, config.mc_samples, device,
        )
        mc_90 = mc_coverage(mc_means, mc_vars, mc_targets, alpha=0.1)
        print(f"  MC coverage (90%): {mc_90['coverage']:.4f}")
        print(f"  MC interval width: {mc_90['mean_interval_width']:.6f}")

        mc_cal_nominal, mc_cal_empirical = [], []
        for alpha in alphas:
            m = mc_coverage(mc_means, mc_vars, mc_targets, alpha=alpha)
            mc_cal_nominal.append(1 - alpha)
            mc_cal_empirical.append(m["coverage"])
        print("  MC calibration:")
        for nom, emp in zip(mc_cal_nominal, mc_cal_empirical, strict=True):
            print(f"    {nom:.0%} -> {emp:.2%}")

    # ===== Adaptive CP via gradient-based difficulty (no extra inference) =====
    print("\n[5/6] Adaptive conformal prediction (gradient-based)...")
    cal_preds_arr = np.stack(cal_data["preds"])

    cal_grad_sigma = gradient_sigma(cal_preds_arr)
    test_preds_arr = np.stack(data["preds"])
    test_grad_sigma = gradient_sigma(test_preds_arr)

    print(f"  Grad sigma contrast (p90/p10): {np.percentile(test_grad_sigma, 90) / max(np.percentile(test_grad_sigma, 10), 1e-8):.1f}x")

    # Compute adaptive scores: residual / sigma (smooth=0 since grad_sigma already smoothed)
    adaptive_cal_scores = compute_adaptive_scores(cal_scores, cal_grad_sigma, smooth=0)
    adaptive_q_90 = compute_quantile(adaptive_cal_scores, 0.1)
    print(f"  Adaptive q_hat (90%): {adaptive_q_90:.4f}")
    print(f"  Uniform q_hat (90%):  {q_90:.4f}")

    test_sigma_list = [test_grad_sigma[i] for i in range(test_grad_sigma.shape[0])]

    adaptive_90 = adaptive_coverage_from_arrays(
        data["preds"], data["targets"], test_sigma_list, adaptive_q_90, smooth=0,
    )
    print(f"  Adaptive coverage (90%): {adaptive_90['coverage']:.4f}")
    print(f"  Adaptive mean width:     {adaptive_90['mean_interval_width']:.6f}")
    print(f"  Adaptive median width:   {adaptive_90['median_interval_width']:.6f}")
    print(f"  Uniform mean width:      {cp_90['mean_interval_width']:.6f}")
    reduction = (1 - adaptive_90['median_interval_width'] / cp_90['mean_interval_width']) * 100
    print(f"  Median width reduction:  {reduction:.1f}%")

    # Adaptive calibration curve
    adaptive_cal = adaptive_calibration_from_arrays(
        cal_scores, cal_grad_sigma,
        data["preds"], data["targets"], test_sigma_list,
        alphas, smooth=0,
    )
    print("  Adaptive CP calibration:")
    for nom, emp in zip(adaptive_cal["nominal"], adaptive_cal["empirical"], strict=True):
        print(f"    {nom:.0%} -> {emp:.2%}")

    # ===== Select examples =====
    print("\n[6/6] Selecting examples + saving...")
    psnr_sorted = np.argsort(img_metrics["psnr"])
    selected = [
        int(psnr_sorted[0]),
        int(psnr_sorted[n_test // 4]),
        int(psnr_sorted[n_test // 2]),
        int(psnr_sorted[3 * n_test // 4]),
        int(psnr_sorted[-1]),
    ]
    rng = np.random.default_rng(42)
    extra = rng.choice(n_test, size=min(5, n_test), replace=False).tolist()
    selected = list(dict.fromkeys(selected + extra))[:N_EXAMPLES]

    # Fixed dataset indices for cross-acceleration comparison.
    # These are the same physical slices regardless of PSNR ranking,
    # so 4x and 8x results can be compared on identical anatomy.
    shared_indices = [n_test // 4, n_test // 2, 3 * n_test // 4]

    # ===== Save =====
    save_dict = {
        # Examples (PSNR-ranked, may differ between 4x and 8x)
        "example_preds": np.stack([data["preds"][i] for i in selected]),
        "example_targets": np.stack([data["targets"][i] for i in selected]),
        "example_inputs": np.stack([data["inputs"][i] for i in selected]),
        "example_masks": np.stack([data["masks"][i] for i in selected]),
        "example_errors": np.stack([data["errors"][i] for i in selected]),
        "selected_indices": np.array(selected),
        # Shared examples (fixed indices, same physical slice across accelerations)
        "shared_preds": np.stack([data["preds"][i] for i in shared_indices]),
        "shared_targets": np.stack([data["targets"][i] for i in shared_indices]),
        "shared_inputs": np.stack([data["inputs"][i] for i in shared_indices]),
        "shared_errors": np.stack([data["errors"][i] for i in shared_indices]),
        "shared_indices": np.array(shared_indices),
        # Metrics
        "per_img_psnr": img_metrics["psnr"],
        "per_img_ssim": img_metrics["ssim"],
        "per_img_nmse": img_metrics["nmse"],
        "mean_psnr": np.array(img_metrics["psnr"].mean()),
        "mean_ssim": np.array(img_metrics["ssim"].mean()),
        "mean_nmse": np.array(img_metrics["nmse"].mean()),
        # Conformal
        "cal_scores_summary": np.percentile(cal_scores.ravel(), [25, 50, 75, 90, 95, 99]),
        "q_hat_90": np.array(q_90),
        "q_hat_95": np.array(q_95),
        "q_hat_80": np.array(q_80),
        "cp_coverage": np.array(cp_90["coverage"]),
        "cp_interval_width": np.array(cp_90["mean_interval_width"]),
        "per_img_coverage": per_img_coverage,
        "per_img_interval_width": np.full(n_test, 2.0 * q_90),
        "cp_cal_nominal": np.array(cp_cal["nominal"]),
        "cp_cal_empirical": np.array(cp_cal["empirical"]),
        "example_lowers": np.stack([data["preds"][i] - q_90 for i in selected]),
        "example_uppers": np.stack([data["preds"][i] + q_90 for i in selected]),
        # MC Dropout
        "mc_coverage": np.array(mc_90["coverage"]),
        "mc_interval_width": np.array(mc_90["mean_interval_width"]),
        "mc_cal_nominal": np.array(mc_cal_nominal),
        "mc_cal_empirical": np.array(mc_cal_empirical),
        "mc_example_means": mc_means[:N_EXAMPLES],
        "mc_example_variances": mc_vars[:N_EXAMPLES],
        # K-space
        "ks_scores": ks_scores,
        "ks_mean_score": np.array(ks_scores.mean()),
        "ks_correlation": np.array(ks_corr),
        "ks_example_residuals": np.stack([data["ks_residuals"][i] for i in selected]),
        "ks_example_errors": np.stack([data["errors"][i] for i in selected]),
        # Pixel error distribution
        "pixel_errors_flat": np.concatenate([
            data["errors"][i].ravel()[::10] for i in range(min(100, n_test))
        ]),
        # Adaptive conformal prediction
        "adaptive_q_hat_90": np.array(adaptive_q_90),
        "adaptive_coverage": np.array(adaptive_90["coverage"]),
        "adaptive_mean_interval_width": np.array(adaptive_90["mean_interval_width"]),
        "adaptive_median_interval_width": np.array(adaptive_90["median_interval_width"]),
        "adaptive_per_img_coverage": adaptive_90["per_image_coverage"],
        "adaptive_cal_nominal": np.array(adaptive_cal["nominal"]),
        "adaptive_cal_empirical": np.array(adaptive_cal["empirical"]),
        # Per-example adaptive interval widths (for visualisation)
        "example_grad_sigma": np.stack([test_grad_sigma[i] for i in selected]),
        "example_adaptive_lowers": np.stack([
            data["preds"][i] - adaptive_q_90 * np.maximum(test_grad_sigma[i], 1e-3)
            for i in selected
        ]),
        "example_adaptive_uppers": np.stack([
            data["preds"][i] + adaptive_q_90 * np.maximum(test_grad_sigma[i], 1e-3)
            for i in selected
        ]),
        "example_adaptive_widths": np.stack([
            2.0 * adaptive_q_90 * np.maximum(test_grad_sigma[i], 1e-3)
            for i in selected
        ]),
    }

    save_path = output_dir / f"results_{acc}x.npz"
    np.savez_compressed(save_path, **save_dict)
    print(f"  Saved to {save_path} ({save_path.stat().st_size / 1e6:.1f} MB)")

    print(f"\n{'='*60}")
    print(f"SUMMARY ({acc}x)")
    print(f"  PSNR:  {img_metrics['psnr'].mean():.2f} dB")
    print(f"  SSIM:  {img_metrics['ssim'].mean():.4f}")
    print(f"  CP uniform coverage (90%):   {cp_90['coverage']:.4f}  width={cp_90['mean_interval_width']:.4f}")
    print(f"  CP adaptive coverage (90%):  {adaptive_90['coverage']:.4f}  mean={adaptive_90['mean_interval_width']:.4f}  median={adaptive_90['median_interval_width']:.4f}")
    print(f"  MC coverage (90%):           {mc_90['coverage']:.4f}  width={mc_90['mean_interval_width']:.4f}")
    print(f"  KS correlation:              {ks_corr:.4f}")
    print(f"{'='*60}")


def auto_device() -> str:
    """Pick best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    device = auto_device()
    print(f"Using device: {device}")

    for acc, ckpt in [(4, "best_4x.pt"), (8, "best_8x.pt")]:
        config = EvalConfig(
            checkpoint_path=Path(f"outputs/checkpoints/{ckpt}"),
            acceleration=acc,
            device=device,
        )
        run_evaluation(config)


if __name__ == "__main__":
    main()
