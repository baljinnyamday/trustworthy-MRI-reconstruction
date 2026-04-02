"""Generate scatter plot: per-image CP coverage vs k-space consistency score.

Shows whether conformal prediction and k-space consistency capture
different failure modes (complementary) or redundant information.

Uses saved evaluation results — no re-inference needed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Match paper figure style
COLOR_4X = "#2196F3"
COLOR_8X = "#FF9800"
DPI = 300
OUTPUT_DIR = Path("outputs/figures")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    r4 = dict(np.load("outputs/results_4x.npz", allow_pickle=True))
    r8 = dict(np.load("outputs/results_8x.npz", allow_pickle=True))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    for ax, r, acc, color, label in [
        (axes[0], r4, 4, COLOR_4X, "(a) 4×"),
        (axes[1], r8, 8, COLOR_8X, "(b) 8×"),
    ]:
        ks = r["ks_scores"]
        cp_cov = r["per_img_coverage"] * 100  # to percent

        # Correlation
        rho, pval = pearsonr(ks, cp_cov)

        ax.scatter(ks, cp_cov, s=8, alpha=0.3, c=color, edgecolors="none")
        ax.axhline(90, color="red", linestyle="--", linewidth=1, alpha=0.7, label="90% nominal")

        # Linear fit for trend line
        z = np.polyfit(ks, cp_cov, 1)
        x_fit = np.linspace(ks.min(), ks.max(), 100)
        ax.plot(x_fit, np.polyval(z, x_fit), color="black", linewidth=1.5,
                label=f"$r = {rho:.2f}$ ($p < 10^{{-4}}$)" if pval < 1e-4 else f"$r = {rho:.2f}$ ($p = {pval:.3f}$)")

        ax.set_xlabel("K-space consistency score", fontsize=10)
        ax.set_ylabel("Per-image CP coverage (%)", fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8, loc="lower left")
        ax.set_ylim(60, 102)

    fig.suptitle("CP Coverage vs K-Space Consistency Score", fontsize=12, y=1.02)

    out_path = OUTPUT_DIR / "fig_cp_vs_kspace_scatter.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")

    # Also print correlation stats for the paper
    for acc, r in [(4, r4), (8, r8)]:
        ks = r["ks_scores"]
        cp_cov = r["per_img_coverage"]
        rho_cp, _ = pearsonr(ks, cp_cov)

        # Also check: k-space vs PSNR (image-level error)
        psnr_vals = r["per_img_psnr"]
        rho_psnr, _ = pearsonr(ks, psnr_vals)

        # Adaptive CP coverage vs k-space
        adapt_cov = r["adaptive_per_img_coverage"]
        rho_adapt, _ = pearsonr(ks, adapt_cov)

        print(f"\n{acc}x:")
        print(f"  KS vs uniform CP coverage: r = {rho_cp:.3f}")
        print(f"  KS vs adaptive CP coverage: r = {rho_adapt:.3f}")
        print(f"  KS vs PSNR: r = {rho_psnr:.3f}")


if __name__ == "__main__":
    main()
