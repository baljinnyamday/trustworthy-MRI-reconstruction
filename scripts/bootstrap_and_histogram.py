"""Compute bootstrap CIs for key metrics and generate per-image coverage histogram."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs/figures")
PAPER_DIR = Path("paper/figures")
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
RNG = np.random.default_rng(42)


def bootstrap_ci(
    data: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, ci: float = CI_LEVEL
) -> tuple[float, float, float]:
    """Compute bootstrap mean and confidence interval."""
    means = np.array([
        np.mean(RNG.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(means, 100 * alpha)
    hi = np.percentile(means, 100 * (1 - alpha))
    return float(np.mean(data)), float(lo), float(hi)


def bootstrap_correlation(
    x: np.ndarray, y: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, ci: float = CI_LEVEL
) -> tuple[float, float, float]:
    """Bootstrap CI for Pearson correlation."""
    n = len(x)
    corrs = np.array([
        np.corrcoef(
            x[idx := RNG.choice(n, size=n, replace=True)],
            y[idx],
        )[0, 1]
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(corrs, 100 * alpha)
    hi = np.percentile(corrs, 100 * (1 - alpha))
    return float(np.corrcoef(x, y)[0, 1]), float(lo), float(hi)


def process_acceleration(npz_path: str, accel: str) -> dict[str, tuple[float, float, float]]:
    """Compute bootstrap CIs for one acceleration factor."""
    d = np.load(npz_path, allow_pickle=True)

    results: dict[str, tuple[float, float, float]] = {}

    # Reconstruction metrics
    results["PSNR"] = bootstrap_ci(d["per_img_psnr"])
    results["SSIM"] = bootstrap_ci(d["per_img_ssim"])

    # CP coverage
    results["CP_coverage"] = bootstrap_ci(d["per_img_coverage"])

    # CP interval width
    results["CP_width"] = bootstrap_ci(d["per_img_interval_width"])

    # K-space: bootstrap correlation using per-image scores vs per-image errors
    # Use ks_scores (per-image) and per-image NMSE as proxy for error
    results["KS_correlation"] = bootstrap_correlation(
        d["ks_scores"], d["per_img_nmse"]
    )

    print(f"\n{'='*60}")
    print(f"Bootstrap 95% CIs — {accel} acceleration")
    print(f"{'='*60}")
    for name, (mean, lo, hi) in results.items():
        if "coverage" in name.lower():
            print(f"  {name}: {mean:.1%} [{lo:.1%}, {hi:.1%}]")
        elif "PSNR" in name:
            print(f"  {name}: {mean:.2f} [{lo:.2f}, {hi:.2f}]")
        elif "correlation" in name.lower():
            print(f"  {name}: {mean:.3f} [{lo:.3f}, {hi:.3f}]")
        else:
            print(f"  {name}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    return results


def plot_coverage_histogram(path_4x: str, path_8x: str) -> None:
    """Plot per-image CP coverage distribution for 4x and 8x."""
    d4 = np.load(path_4x, allow_pickle=True)
    d8 = np.load(path_8x, allow_pickle=True)

    cov_4x = d4["per_img_coverage"] * 100  # to percentage
    cov_8x = d8["per_img_coverage"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

    for ax, cov, accel, color in [
        (axes[0], cov_4x, "4×", "#2196F3"),
        (axes[1], cov_8x, "8×", "#FF9800"),
    ]:
        ax.hist(cov, bins=40, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(90, color="red", linestyle="--", linewidth=1.5, label="90% nominal")
        ax.axvline(np.mean(cov), color="black", linestyle="-", linewidth=1.5,
                   label=f"Mean: {np.mean(cov):.1f}%")

        ax.set_xlabel("Per-image coverage (%)", fontsize=10)
        ax.set_title(f"{accel} acceleration", fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xlim(70, 100)

        # Add stats text
        stats_text = (
            f"σ = {np.std(cov):.1f}%\n"
            f"5th pctl = {np.percentile(cov, 5):.1f}%\n"
            f"Min = {np.min(cov):.1f}%"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    axes[0].set_ylabel("Number of images", fontsize=10)

    fig.suptitle("Per-image conformal prediction coverage (nominal 90%)", fontsize=12, y=1.02)
    plt.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"fig_coverage_histogram.{fmt}",
                    dpi=200, bbox_inches="tight")
    fig.savefig(PAPER_DIR / "fig_coverage_histogram.pdf",
                dpi=200, bbox_inches="tight")

    print(f"\nHistogram saved to {OUTPUT_DIR / 'fig_coverage_histogram.pdf'}")
    plt.close(fig)


def main() -> None:
    path_4x = "outputs/results_4x.npz"
    path_8x = "outputs/results_8x.npz"

    process_acceleration(path_4x, "4×")
    process_acceleration(path_8x, "8×")
    plot_coverage_histogram(path_4x, path_8x)


if __name__ == "__main__":
    main()
