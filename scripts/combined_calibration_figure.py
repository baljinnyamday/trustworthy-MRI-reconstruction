"""Generate combined figure: calibration curve + coverage histogram."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs/figures")
PAPER_DIR = Path("paper/figures")


def main() -> None:
    d4 = np.load("outputs/results_4x.npz", allow_pickle=True)
    d8 = np.load("outputs/results_8x.npz", allow_pickle=True)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    # --- Panel (a): Calibration curve (4x) ---
    ax = axes[0]
    nominal = d4["cp_cal_nominal"] * 100
    cp_emp = d4["cp_cal_empirical"] * 100
    mc_emp = d4["mc_cal_empirical"] * 100

    ax.plot([40, 100], [40, 100], "k--", linewidth=1, alpha=0.5, label="Ideal")
    ax.plot(nominal, cp_emp, "o-", color="#2196F3", markersize=5, linewidth=1.5, label="Conformal")
    ax.plot(nominal, mc_emp, "s-", color="#FF9800", markersize=5, linewidth=1.5, label="MC Dropout")
    ax.set_xlabel("Nominal coverage (%)", fontsize=9)
    ax.set_ylabel("Empirical coverage (%)", fontsize=9)
    ax.set_title("(a) Calibration ($4{\\times}$)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(45, 100)
    ax.set_ylim(45, 100)
    ax.set_aspect("equal")

    # --- Panels (b,c): Coverage histograms ---
    for i, (d, accel, color, panel) in enumerate([
        (d4, "4×", "#2196F3", "(b)"),
        (d8, "8×", "#FF9800", "(c)"),
    ]):
        ax = axes[i + 1]
        cov = d["per_img_coverage"] * 100

        ax.hist(cov, bins=40, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(90, color="red", linestyle="--", linewidth=1.5, label="90% nominal")
        ax.axvline(np.mean(cov), color="black", linestyle="-", linewidth=1.5,
                   label=f"Mean: {np.mean(cov):.1f}%")

        ax.set_xlabel("Per-image coverage (%)", fontsize=9)
        ax.set_title(f"{panel} Coverage dist. ({accel})", fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlim(50, 100)

        stats_text = (
            f"$\\sigma$ = {np.std(cov):.1f}%\n"
            f"5th = {np.percentile(cov, 5):.1f}%\n"
            f"min = {np.min(cov):.1f}%"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    axes[1].set_ylabel("Number of images", fontsize=9)

    plt.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"fig_calibration_and_coverage.{fmt}",
                    dpi=200, bbox_inches="tight")
    fig.savefig(PAPER_DIR / "fig_calibration_and_coverage.pdf",
                dpi=200, bbox_inches="tight")

    print("Saved combined figure")
    plt.close(fig)


if __name__ == "__main__":
    main()
