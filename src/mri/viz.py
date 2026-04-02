"""Visualization functions for paper figures.

Generate as many figures as possible — we can pick the best ones later.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Consistent style for all figures
CMAP_IMG = "gray"
CMAP_ERR = "hot"
CMAP_UNC = "viridis"
COLOR_CP = "#2196F3"
COLOR_MC = "#FF5722"
COLOR_IDEAL = "black"
DPI = 300


def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure and clean up."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 1. RECONSTRUCTION FIGURES
# =====================================================================


def plot_reconstruction_comparison(
    ground_truth: np.ndarray,
    zero_filled: np.ndarray,
    reconstruction: np.ndarray,
    output_path: Path,
    acceleration: int = 4,
) -> None:
    """Side-by-side: GT / zero-filled / reconstruction / error map."""
    error = np.abs(ground_truth - reconstruction)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    fig.suptitle(f"MRI Reconstruction at {acceleration}x Acceleration", fontsize=13)

    titles = ["Ground Truth", "Zero-Filled", "U-Net Recon", "Reconstruction Error"]
    images = [ground_truth, zero_filled, reconstruction, error]
    cmaps = [CMAP_IMG, CMAP_IMG, CMAP_IMG, CMAP_ERR]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps, strict=True):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.colorbar(axes[3].images[0], ax=axes[3], shrink=0.8, label="Absolute Error")
    _save(fig, output_path)


def plot_reconstruction_grid(
    targets: np.ndarray,
    inputs: np.ndarray,
    preds: np.ndarray,
    errors: np.ndarray,
    output_path: Path,
    acceleration: int = 4,
    n_rows: int = 3,
) -> None:
    """Grid of multiple examples: each row is GT / ZF / Recon / Error."""
    n_rows = min(n_rows, len(targets))
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows), constrained_layout=True)
    fig.suptitle(f"Reconstruction Examples ({acceleration}x)", fontsize=14)

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Ground Truth", "Zero-Filled", "U-Net Recon", "Error"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11)

    for row in range(n_rows):
        axes[row, 0].imshow(targets[row], cmap=CMAP_IMG)
        axes[row, 1].imshow(inputs[row], cmap=CMAP_IMG)
        axes[row, 2].imshow(preds[row], cmap=CMAP_IMG)
        axes[row, 3].imshow(errors[row], cmap=CMAP_ERR)
        for col in range(4):
            axes[row, col].axis("off")

    _save(fig, output_path)


def plot_acceleration_comparison(
    target: np.ndarray,
    input_4x: np.ndarray,
    pred_4x: np.ndarray,
    input_8x: np.ndarray,
    pred_8x: np.ndarray,
    output_path: Path,
) -> None:
    """Same anatomy at 4x vs 8x: GT / ZF_4x / Recon_4x / ZF_8x / Recon_8x / Errors."""
    error_4x = np.abs(target - pred_4x)
    error_8x = np.abs(target - pred_8x)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    fig.suptitle("4x vs 8x Acceleration Comparison", fontsize=14)

    # Row 0: 4x
    axes[0, 0].imshow(target, cmap=CMAP_IMG)
    axes[0, 0].set_title("Ground Truth")
    axes[0, 1].imshow(input_4x, cmap=CMAP_IMG)
    axes[0, 1].set_title("Zero-Filled (4x)")
    axes[0, 2].imshow(pred_4x, cmap=CMAP_IMG)
    axes[0, 2].set_title("U-Net (4x)")
    axes[0, 3].imshow(error_4x, cmap=CMAP_ERR)
    axes[0, 3].set_title("Error (4x)")

    # Row 1: 8x
    axes[1, 0].imshow(target, cmap=CMAP_IMG)
    axes[1, 0].set_title("Ground Truth")
    axes[1, 1].imshow(input_8x, cmap=CMAP_IMG)
    axes[1, 1].set_title("Zero-Filled (8x)")
    axes[1, 2].imshow(pred_8x, cmap=CMAP_IMG)
    axes[1, 2].set_title("U-Net (8x)")
    axes[1, 3].imshow(error_8x, cmap=CMAP_ERR)
    axes[1, 3].set_title("Error (8x)")

    for ax in axes.ravel():
        ax.axis("off")

    _save(fig, output_path)


def plot_psnr_ssim_distribution(
    psnr_4x: np.ndarray,
    ssim_4x: np.ndarray,
    psnr_8x: np.ndarray,
    ssim_8x: np.ndarray,
    output_path: Path,
) -> None:
    """Box plots of PSNR and SSIM across test set, 4x vs 8x."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    # PSNR
    bp1 = axes[0].boxplot(
        [psnr_4x, psnr_8x],
        labels=["4x", "8x"],
        patch_artist=True,
    )
    bp1["boxes"][0].set_facecolor(COLOR_CP)
    bp1["boxes"][1].set_facecolor(COLOR_MC)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Reconstruction PSNR")
    axes[0].grid(True, alpha=0.3, axis="y")

    # SSIM
    bp2 = axes[1].boxplot(
        [ssim_4x, ssim_8x],
        labels=["4x", "8x"],
        patch_artist=True,
    )
    bp2["boxes"][0].set_facecolor(COLOR_CP)
    bp2["boxes"][1].set_facecolor(COLOR_MC)
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("Reconstruction SSIM")
    axes[1].grid(True, alpha=0.3, axis="y")

    _save(fig, output_path)


def plot_error_histogram(
    errors_4x: np.ndarray,
    errors_8x: np.ndarray,
    output_path: Path,
) -> None:
    """Pixel error distribution comparison: 4x vs 8x."""
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    ax.hist(errors_4x.ravel(), bins=100, alpha=0.6, color=COLOR_CP,
            label="4x", density=True, range=(0, np.percentile(errors_8x, 99)))
    ax.hist(errors_8x.ravel(), bins=100, alpha=0.6, color=COLOR_MC,
            label="8x", density=True, range=(0, np.percentile(errors_8x, 99)))

    ax.set_xlabel("Absolute Pixel Error")
    ax.set_ylabel("Density")
    ax.set_title("Pixel Error Distribution: 4x vs 8x")
    ax.legend()
    ax.set_yscale("log")

    _save(fig, output_path)


# =====================================================================
# 2. CONFORMAL PREDICTION FIGURES
# =====================================================================


def plot_conformal_intervals(
    reconstruction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Path,
    acceleration: int = 4,
) -> None:
    """Show reconstruction, interval width map, and coverage map."""
    interval_width = upper - lower
    covered = (ground_truth >= lower) & (ground_truth <= upper)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    fig.suptitle(f"Conformal Prediction Intervals ({acceleration}x)", fontsize=13)

    axes[0].imshow(reconstruction, cmap=CMAP_IMG)
    axes[0].set_title("Reconstruction")
    axes[0].axis("off")

    im_width = axes[1].imshow(interval_width, cmap=CMAP_UNC)
    axes[1].set_title("Interval Width")
    axes[1].axis("off")
    fig.colorbar(im_width, ax=axes[1], shrink=0.8)

    # Coverage map: green = covered, red = not covered
    coverage_rgb = np.zeros((*covered.shape, 3))
    coverage_rgb[covered] = [0.2, 0.8, 0.2]
    coverage_rgb[~covered] = [0.9, 0.1, 0.1]
    axes[2].imshow(coverage_rgb)
    axes[2].set_title(f"Coverage ({covered.mean():.1%})")
    axes[2].axis("off")

    error = np.abs(ground_truth - reconstruction)
    im_err = axes[3].imshow(error, cmap=CMAP_ERR)
    axes[3].set_title("Actual Error")
    axes[3].axis("off")
    fig.colorbar(im_err, ax=axes[3], shrink=0.8)

    _save(fig, output_path)


def plot_calibration_curve(
    nominal_cp: list[float],
    empirical_cp: list[float],
    nominal_mc: list[float],
    empirical_mc: list[float],
    output_path: Path,
    title: str = "Calibration: Conformal vs MC Dropout",
) -> None:
    """Calibration plot: nominal vs empirical coverage for CP and MC Dropout."""
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")

    ax.plot(nominal_cp, empirical_cp, "s-", color=COLOR_CP, markersize=8,
            label="Conformal Prediction", linewidth=2)
    ax.plot(nominal_mc, empirical_mc, "o-", color=COLOR_MC, markersize=8,
            label="MC Dropout", linewidth=2)

    ax.set_xlabel("Nominal Coverage", fontsize=12)
    ax.set_ylabel("Empirical Coverage", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    _save(fig, output_path)


def plot_coverage_histogram(
    per_img_coverage: np.ndarray,
    target_coverage: float,
    output_path: Path,
    acceleration: int = 4,
) -> None:
    """Distribution of per-image conformal coverage."""
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    ax.hist(per_img_coverage, bins=30, alpha=0.7, color=COLOR_CP, edgecolor="white")
    ax.axvline(target_coverage, color="red", linestyle="--", linewidth=2,
               label=f"Target ({target_coverage:.0%})")
    ax.axvline(per_img_coverage.mean(), color=COLOR_CP, linestyle="-", linewidth=2,
               label=f"Mean ({per_img_coverage.mean():.1%})")

    ax.set_xlabel("Per-Image Coverage")
    ax.set_ylabel("Count")
    ax.set_title(f"Coverage Distribution ({acceleration}x)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    _save(fig, output_path)


def plot_cp_vs_mc_intervals(
    reconstruction: np.ndarray,
    target: np.ndarray,
    cp_lower: np.ndarray,
    cp_upper: np.ndarray,
    mc_mean: np.ndarray,
    mc_variance: np.ndarray,
    output_path: Path,
    alpha: float = 0.1,
) -> None:
    """Side-by-side: CP intervals vs MC Dropout intervals on the same slice."""
    from scipy.stats import norm as sp_norm

    z = sp_norm.ppf(1 - alpha / 2)
    mc_std = np.sqrt(mc_variance)
    mc_lower = mc_mean - z * mc_std
    mc_upper = mc_mean + z * mc_std

    cp_width = cp_upper - cp_lower
    mc_width = mc_upper - mc_lower

    cp_covered = (target >= cp_lower) & (target <= cp_upper)
    mc_covered = (target >= mc_lower) & (target <= mc_upper)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f"Conformal vs MC Dropout Intervals ({1-alpha:.0%} nominal)", fontsize=14)

    # Row 0: Conformal
    axes[0, 0].imshow(reconstruction, cmap=CMAP_IMG)
    axes[0, 0].set_title("Reconstruction")
    im0 = axes[0, 1].imshow(cp_width, cmap=CMAP_UNC)
    axes[0, 1].set_title(f"CP Interval Width")
    fig.colorbar(im0, ax=axes[0, 1], shrink=0.8)
    cp_rgb = np.zeros((*cp_covered.shape, 3))
    cp_rgb[cp_covered] = [0.2, 0.8, 0.2]
    cp_rgb[~cp_covered] = [0.9, 0.1, 0.1]
    axes[0, 2].imshow(cp_rgb)
    axes[0, 2].set_title(f"CP Coverage ({cp_covered.mean():.1%})")

    # Row 1: MC Dropout
    axes[1, 0].imshow(mc_mean, cmap=CMAP_IMG)
    axes[1, 0].set_title("MC Mean")
    im1 = axes[1, 1].imshow(mc_width, cmap=CMAP_UNC)
    axes[1, 1].set_title(f"MC Interval Width")
    fig.colorbar(im1, ax=axes[1, 1], shrink=0.8)
    mc_rgb = np.zeros((*mc_covered.shape, 3))
    mc_rgb[mc_covered] = [0.2, 0.8, 0.2]
    mc_rgb[~mc_covered] = [0.9, 0.1, 0.1]
    axes[1, 2].imshow(mc_rgb)
    axes[1, 2].set_title(f"MC Coverage ({mc_covered.mean():.1%})")

    for ax in axes.ravel():
        ax.axis("off")

    _save(fig, output_path)


def plot_interval_width_comparison(
    width_map_4x: np.ndarray,
    width_map_8x: np.ndarray,
    output_path: Path,
) -> None:
    """Interval width maps at 4x vs 8x, shared colorbar."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    fig.suptitle("Conformal Interval Width: 4x vs 8x", fontsize=13)

    vmax = max(width_map_4x.max(), width_map_8x.max())

    im0 = axes[0].imshow(width_map_4x, cmap=CMAP_UNC, vmin=0, vmax=vmax)
    axes[0].set_title(f"4x (mean: {width_map_4x.mean():.4f})")
    axes[0].axis("off")

    im1 = axes[1].imshow(width_map_8x, cmap=CMAP_UNC, vmin=0, vmax=vmax)
    axes[1].set_title(f"8x (mean: {width_map_8x.mean():.4f})")
    axes[1].axis("off")

    fig.colorbar(im1, ax=axes, shrink=0.8, label="Interval Width")
    _save(fig, output_path)


# =====================================================================
# 3. UNCERTAINTY FIGURES
# =====================================================================


def plot_uncertainty_vs_error(
    uncertainty: np.ndarray,
    error: np.ndarray,
    output_path: Path,
    title: str = "Uncertainty vs Reconstruction Error",
) -> None:
    """2D histogram showing correlation between uncertainty and actual error."""
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    rng = np.random.default_rng(42)
    flat_unc = uncertainty.ravel()
    flat_err = error.ravel()
    idx = rng.choice(len(flat_unc), size=min(100_000, len(flat_unc)), replace=False)

    ax.hist2d(flat_unc[idx], flat_err[idx], bins=100, cmap=CMAP_UNC, norm=LogNorm())
    ax.set_xlabel("Uncertainty", fontsize=12)
    ax.set_ylabel("Absolute Error", fontsize=12)
    ax.set_title(title, fontsize=13)

    corr = np.corrcoef(flat_unc, flat_err)[0, 1]
    ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
            fontsize=12, verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})

    _save(fig, output_path)


def plot_mc_uncertainty_maps(
    reconstruction: np.ndarray,
    mc_mean: np.ndarray,
    mc_variance: np.ndarray,
    target: np.ndarray,
    output_path: Path,
) -> None:
    """MC Dropout: mean prediction, variance map, std overlaid on recon."""
    mc_std = np.sqrt(mc_variance)
    error = np.abs(target - mc_mean)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    fig.suptitle("MC Dropout Uncertainty", fontsize=13)

    axes[0].imshow(mc_mean, cmap=CMAP_IMG)
    axes[0].set_title("MC Mean")
    axes[0].axis("off")

    im1 = axes[1].imshow(mc_std, cmap=CMAP_UNC)
    axes[1].set_title("MC Std Dev")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(error, cmap=CMAP_ERR)
    axes[2].set_title("Actual Error")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    # Std/Error correlation overlay
    corr = np.corrcoef(mc_std.ravel(), error.ravel())[0, 1]
    axes[3].imshow(reconstruction, cmap=CMAP_IMG, alpha=0.5)
    im3 = axes[3].imshow(mc_std, cmap="Reds", alpha=0.5)
    axes[3].set_title(f"Uncertainty Overlay (r={corr:.2f})")
    axes[3].axis("off")

    _save(fig, output_path)


# =====================================================================
# 4. K-SPACE CONSISTENCY FIGURES
# =====================================================================


def plot_kspace_consistency(
    residual_4x: np.ndarray,
    residual_8x: np.ndarray,
    error_4x: np.ndarray,
    error_8x: np.ndarray,
    output_path: Path,
) -> None:
    """K-space consistency maps at 4x vs 8x alongside actual error maps."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    fig.suptitle("K-Space Consistency: Hallucination Detection", fontsize=13)

    vmax_res = max(residual_4x.max(), residual_8x.max())
    vmax_err = max(error_4x.max(), error_8x.max())

    axes[0, 0].imshow(residual_4x, cmap=CMAP_ERR, vmax=vmax_res)
    axes[0, 0].set_title("K-Space Residual (4x)")
    axes[0, 1].imshow(error_4x, cmap=CMAP_ERR, vmax=vmax_err)
    axes[0, 1].set_title("Actual Error (4x)")
    axes[1, 0].imshow(residual_8x, cmap=CMAP_ERR, vmax=vmax_res)
    axes[1, 0].set_title("K-Space Residual (8x)")
    im = axes[1, 1].imshow(error_8x, cmap=CMAP_ERR, vmax=vmax_err)
    axes[1, 1].set_title("Actual Error (8x)")

    for ax in axes.ravel():
        ax.axis("off")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Magnitude")
    _save(fig, output_path)


def plot_kspace_scatter(
    ks_scores: np.ndarray,
    per_img_errors: np.ndarray,
    output_path: Path,
) -> None:
    """Per-image scatter: k-space consistency score vs mean reconstruction error."""
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    ax.scatter(ks_scores, per_img_errors, alpha=0.5, s=20, color=COLOR_CP)

    # Fit line
    z = np.polyfit(ks_scores, per_img_errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(ks_scores.min(), ks_scores.max(), 100)
    ax.plot(x_line, p(x_line), color=COLOR_MC, linewidth=2, linestyle="--")

    corr = np.corrcoef(ks_scores, per_img_errors)[0, 1]
    ax.set_xlabel("K-Space Consistency Score")
    ax.set_ylabel("Mean Reconstruction Error")
    ax.set_title(f"K-Space Score vs Error (r = {corr:.3f})")
    ax.grid(True, alpha=0.3)

    _save(fig, output_path)


def plot_kspace_detail(
    reconstruction: np.ndarray,
    target: np.ndarray,
    residual: np.ndarray,
    error: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    acceleration: int = 4,
) -> None:
    """Detailed k-space analysis for a single slice."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    fig.suptitle(f"K-Space Consistency Detail ({acceleration}x)", fontsize=13)

    axes[0].imshow(target, cmap=CMAP_IMG)
    axes[0].set_title("Ground Truth")

    axes[1].imshow(reconstruction, cmap=CMAP_IMG)
    axes[1].set_title("Reconstruction")

    # Mask visualization
    mask_2d = np.broadcast_to(mask[np.newaxis, :] if mask.ndim == 1 else mask,
                              (reconstruction.shape[0], mask.shape[-1]))
    axes[2].imshow(mask_2d, cmap="gray", aspect="auto")
    axes[2].set_title(f"K-Space Mask ({acceleration}x)")

    im_res = axes[3].imshow(residual, cmap=CMAP_ERR)
    axes[3].set_title("K-Space Residual")
    fig.colorbar(im_res, ax=axes[3], shrink=0.8)

    im_err = axes[4].imshow(error, cmap=CMAP_ERR)
    axes[4].set_title("Actual Error")
    fig.colorbar(im_err, ax=axes[4], shrink=0.8)

    for ax in axes:
        ax.axis("off")

    _save(fig, output_path)


def plot_kspace_score_distribution(
    scores_4x: np.ndarray,
    scores_8x: np.ndarray,
    output_path: Path,
) -> None:
    """Distribution of per-image k-space consistency scores."""
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    ax.hist(scores_4x, bins=30, alpha=0.6, color=COLOR_CP, label="4x", edgecolor="white")
    ax.hist(scores_8x, bins=30, alpha=0.6, color=COLOR_MC, label="8x", edgecolor="white")

    ax.axvline(scores_4x.mean(), color=COLOR_CP, linestyle="--", linewidth=2)
    ax.axvline(scores_8x.mean(), color=COLOR_MC, linestyle="--", linewidth=2)

    ax.set_xlabel("K-Space Consistency Score")
    ax.set_ylabel("Count")
    ax.set_title("K-Space Consistency Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    _save(fig, output_path)


# =====================================================================
# 5. COMPREHENSIVE / COMBINED FIGURES
# =====================================================================


def plot_trustworthiness_dashboard(
    target: np.ndarray,
    zero_filled: np.ndarray,
    reconstruction: np.ndarray,
    cp_lower: np.ndarray,
    cp_upper: np.ndarray,
    mc_variance: np.ndarray,
    ks_residual: np.ndarray,
    output_path: Path,
    acceleration: int = 4,
) -> None:
    """All trustworthiness signals for one slice in a single figure."""
    error = np.abs(target - reconstruction)
    cp_width = cp_upper - cp_lower
    covered = (target >= cp_lower) & (target <= cp_upper)
    mc_std = np.sqrt(mc_variance)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    fig.suptitle(f"Trustworthiness Dashboard ({acceleration}x)", fontsize=15)

    # Row 0: Reconstruction
    axes[0, 0].imshow(target, cmap=CMAP_IMG)
    axes[0, 0].set_title("Ground Truth")
    axes[0, 1].imshow(zero_filled, cmap=CMAP_IMG)
    axes[0, 1].set_title("Zero-Filled Input")
    axes[0, 2].imshow(reconstruction, cmap=CMAP_IMG)
    axes[0, 2].set_title("U-Net Reconstruction")
    im_err = axes[0, 3].imshow(error, cmap=CMAP_ERR)
    axes[0, 3].set_title("Reconstruction Error")
    fig.colorbar(im_err, ax=axes[0, 3], shrink=0.8)

    # Row 1: Trustworthiness signals
    im_cp = axes[1, 0].imshow(cp_width, cmap=CMAP_UNC)
    axes[1, 0].set_title("CP Interval Width")
    fig.colorbar(im_cp, ax=axes[1, 0], shrink=0.8)

    coverage_rgb = np.zeros((*covered.shape, 3))
    coverage_rgb[covered] = [0.2, 0.8, 0.2]
    coverage_rgb[~covered] = [0.9, 0.1, 0.1]
    axes[1, 1].imshow(coverage_rgb)
    axes[1, 1].set_title(f"CP Coverage ({covered.mean():.1%})")

    im_mc = axes[1, 2].imshow(mc_std, cmap=CMAP_UNC)
    axes[1, 2].set_title("MC Dropout Std")
    fig.colorbar(im_mc, ax=axes[1, 2], shrink=0.8)

    im_ks = axes[1, 3].imshow(ks_residual, cmap=CMAP_ERR)
    axes[1, 3].set_title("K-Space Residual")
    fig.colorbar(im_ks, ax=axes[1, 3], shrink=0.8)

    for ax in axes.ravel():
        ax.axis("off")

    _save(fig, output_path)


def plot_mask_visualization(
    mask_4x: np.ndarray,
    mask_8x: np.ndarray,
    output_path: Path,
    height: int = 320,
) -> None:
    """Visualize k-space sampling masks at 4x and 8x."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    fig.suptitle("K-Space Sampling Masks", fontsize=13)

    for ax, mask, acc in [(axes[0], mask_4x, 4), (axes[1], mask_8x, 8)]:
        mask_2d = np.broadcast_to(mask[np.newaxis, :], (height, len(mask)))
        ax.imshow(mask_2d, cmap="gray", aspect="auto")
        n_lines = int(mask.sum())
        ax.set_title(f"{acc}x Acceleration ({n_lines}/{len(mask)} lines)")
        ax.set_xlabel("K-Space Column")
        ax.set_ylabel("K-Space Row")

    _save(fig, output_path)


# =====================================================================
# 6. TABLES
# =====================================================================


def plot_metrics_table(
    metrics: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Render metrics as a formatted table figure."""
    fig, ax = plt.subplots(figsize=(8, 2), constrained_layout=True)
    ax.axis("off")

    rows = list(metrics.keys())
    cols = list(next(iter(metrics.values())).keys())
    cell_text = [[f"{metrics[row][col]:.4f}" for col in cols] for row in rows]

    table = ax.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=cols,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    _save(fig, output_path)


COLOR_ADAPTIVE = "#4CAF50"


def plot_adaptive_vs_uniform_intervals(
    reconstruction: np.ndarray,
    target: np.ndarray,
    uniform_lower: np.ndarray,
    uniform_upper: np.ndarray,
    adaptive_lower: np.ndarray,
    adaptive_upper: np.ndarray,
    output_path: Path,
    acceleration: int = 4,
) -> None:
    """Side-by-side comparison of uniform vs adaptive conformal intervals.

    Shows: reconstruction, uniform width map + coverage, adaptive width map + coverage.
    """
    uniform_width = uniform_upper - uniform_lower
    adaptive_width = adaptive_upper - adaptive_lower

    uniform_covered = (target >= uniform_lower) & (target <= uniform_upper)
    adaptive_covered = (target >= adaptive_lower) & (target <= adaptive_upper)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(
        f"Uniform vs Adaptive Conformal Intervals ({acceleration}x)", fontsize=14,
    )

    # Row 0: Uniform CP
    axes[0, 0].imshow(reconstruction, cmap=CMAP_IMG)
    axes[0, 0].set_title("Reconstruction")
    im0 = axes[0, 1].imshow(uniform_width, cmap=CMAP_UNC)
    axes[0, 1].set_title(f"Uniform Width (mean: {uniform_width.mean():.3f})")
    fig.colorbar(im0, ax=axes[0, 1], shrink=0.8)
    u_rgb = np.zeros((*uniform_covered.shape, 3))
    u_rgb[uniform_covered] = [0.2, 0.8, 0.2]
    u_rgb[~uniform_covered] = [0.9, 0.1, 0.1]
    axes[0, 2].imshow(u_rgb)
    axes[0, 2].set_title(f"Uniform Coverage ({uniform_covered.mean():.1%})")

    # Row 1: Adaptive CP
    axes[1, 0].imshow(target, cmap=CMAP_IMG)
    axes[1, 0].set_title("Ground Truth")
    im1 = axes[1, 1].imshow(adaptive_width, cmap=CMAP_UNC)
    axes[1, 1].set_title(
        f"Adaptive Width (mean: {adaptive_width.mean():.3f}, "
        f"med: {np.median(adaptive_width):.3f})",
    )
    fig.colorbar(im1, ax=axes[1, 1], shrink=0.8)
    a_rgb = np.zeros((*adaptive_covered.shape, 3))
    a_rgb[adaptive_covered] = [0.2, 0.8, 0.2]
    a_rgb[~adaptive_covered] = [0.9, 0.1, 0.1]
    axes[1, 2].imshow(a_rgb)
    axes[1, 2].set_title(f"Adaptive Coverage ({adaptive_covered.mean():.1%})")

    for ax in axes.ravel():
        ax.axis("off")

    _save(fig, output_path)


def plot_adaptive_width_histogram(
    uniform_width: float,
    adaptive_widths_4x: np.ndarray,
    adaptive_widths_8x: np.ndarray,
    uniform_width_8x: float,
    output_path: Path,
) -> None:
    """Histogram of adaptive interval widths vs uniform (vertical line)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    fig.suptitle("Interval Width Distribution: Adaptive vs Uniform", fontsize=13)

    for ax, widths, uw, acc in [
        (axes[0], adaptive_widths_4x, uniform_width, 4),
        (axes[1], adaptive_widths_8x, uniform_width_8x, 8),
    ]:
        flat = widths.ravel()
        # Subsample for plotting speed
        rng = np.random.default_rng(42)
        if len(flat) > 500_000:
            flat = flat[rng.choice(len(flat), 500_000, replace=False)]

        ax.hist(flat, bins=200, density=True, alpha=0.7, color=COLOR_ADAPTIVE, label="Adaptive")
        ax.axvline(uw, color="red", linestyle="--", linewidth=2, label=f"Uniform ({uw:.3f})")
        ax.set_xlabel("Interval Width")
        ax.set_ylabel("Density")
        ax.set_title(f"{acc}x Acceleration")
        ax.legend()
        ax.set_xlim(0, min(uw * 3, flat.max()))

    _save(fig, output_path)


def plot_calibration_three_way(
    nominal_uniform: list[float],
    empirical_uniform: list[float],
    nominal_adaptive: list[float],
    empirical_adaptive: list[float],
    nominal_mc: list[float],
    empirical_mc: list[float],
    output_path: Path,
    title: str = "Calibration: Uniform CP vs Adaptive CP vs MC Dropout",
) -> None:
    """Three-way calibration curve: uniform CP, adaptive CP, MC Dropout."""
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)

    ax.plot([0, 1], [0, 1], "--", color=COLOR_IDEAL, alpha=0.5, label="Ideal")
    ax.plot(nominal_uniform, empirical_uniform, "o-", color=COLOR_CP,
            label="Uniform CP", markersize=6, linewidth=2)
    ax.plot(nominal_adaptive, empirical_adaptive, "s-", color=COLOR_ADAPTIVE,
            label="Adaptive CP", markersize=6, linewidth=2)
    ax.plot(nominal_mc, empirical_mc, "^-", color=COLOR_MC,
            label="MC Dropout", markersize=6, linewidth=2)

    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    _save(fig, output_path)


def plot_summary_table(
    r4: dict[str, np.ndarray],
    r8: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Comprehensive summary table with all key metrics."""
    fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
    ax.axis("off")

    cols = ["PSNR (dB)", "SSIM", "CP Cov.", "CP Width",
            "MC Cov.", "MC Width", "KS Corr."]
    rows = ["4x Acceleration", "8x Acceleration"]

    def fmt(v: np.ndarray) -> str:
        return f"{float(v):.4f}"

    cell_text = [
        [
            f"{float(r4['mean_psnr']):.2f}",
            fmt(r4["mean_ssim"]),
            f"{float(r4['cp_coverage']):.2%}",
            fmt(r4["cp_interval_width"]),
            f"{float(r4['mc_coverage']):.2%}",
            fmt(r4["mc_interval_width"]),
            fmt(r4["ks_correlation"]),
        ],
        [
            f"{float(r8['mean_psnr']):.2f}",
            fmt(r8["mean_ssim"]),
            f"{float(r8['cp_coverage']):.2%}",
            fmt(r8["cp_interval_width"]),
            f"{float(r8['mc_coverage']):.2%}",
            fmt(r8["mc_interval_width"]),
            fmt(r8["ks_correlation"]),
        ],
    ]

    table = ax.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=cols,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Color header
    for j in range(len(cols)):
        table[0, j].set_facecolor("#E3F2FD")

    _save(fig, output_path)
