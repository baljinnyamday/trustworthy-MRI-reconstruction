"""Generate ALL paper figures from saved evaluation results.

Run this after run_evaluate.py. Generates ~20+ figures so we can
pick the best ones for the paper later.
"""

from pathlib import Path

import numpy as np

from src.mri.viz import (
    plot_acceleration_comparison,
    plot_adaptive_vs_uniform_intervals,
    plot_adaptive_width_histogram,
    plot_calibration_curve,
    plot_calibration_three_way,
    plot_conformal_intervals,
    plot_coverage_histogram,
    plot_cp_vs_mc_intervals,
    plot_error_histogram,
    plot_interval_width_comparison,
    plot_kspace_consistency,
    plot_kspace_detail,
    plot_kspace_scatter,
    plot_kspace_score_distribution,
    plot_mask_visualization,
    plot_mc_uncertainty_maps,
    plot_psnr_ssim_distribution,
    plot_reconstruction_comparison,
    plot_reconstruction_grid,
    plot_summary_table,
    plot_trustworthiness_dashboard,
    plot_uncertainty_vs_error,
)

OUTPUT_DIR = Path("outputs/figures")


def load_results(acc: int) -> dict[str, np.ndarray]:
    """Load saved evaluation results for an acceleration factor."""
    return dict(np.load(f"outputs/results_{acc}x.npz", allow_pickle=True))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    r4 = load_results(4)
    r8 = load_results(8)

    # =================================================================
    # 1. RECONSTRUCTION FIGURES
    # =================================================================

    # Fig 1a/1b: Single best reconstruction comparison (4x and 8x)
    for acc, r in [(4, r4), (8, r8)]:
        # Use median quality example (index 2 = median in selected_indices)
        idx = 2  # median PSNR example
        plot_reconstruction_comparison(
            ground_truth=r["example_targets"][idx],
            zero_filled=r["example_inputs"][idx],
            reconstruction=r["example_preds"][idx],
            output_path=OUTPUT_DIR / f"fig_recon_{acc}x.png",
            acceleration=acc,
        )
        count += 1
    print(f"  [{count}] Reconstruction comparison (4x, 8x)")

    # Fig 2: Multi-example reconstruction grid (4x and 8x)
    for acc, r in [(4, r4), (8, r8)]:
        plot_reconstruction_grid(
            targets=r["example_targets"],
            inputs=r["example_inputs"],
            preds=r["example_preds"],
            errors=r["example_errors"],
            output_path=OUTPUT_DIR / f"fig_recon_grid_{acc}x.png",
            acceleration=acc,
            n_rows=4,
        )
        count += 1
    print(f"  [{count}] Reconstruction grids")

    # Fig 3: 4x vs 8x side-by-side (same physical slice via shared indices)
    shared_idx = 1  # middle shared example (dataset index n_test//2)
    if "shared_targets" in r4:
        target_for_comparison = r4["shared_targets"][shared_idx]
        input_4x_for_comparison = r4["shared_inputs"][shared_idx]
        pred_4x_for_comparison = r4["shared_preds"][shared_idx]
        input_8x_for_comparison = r8["shared_inputs"][shared_idx]
        pred_8x_for_comparison = r8["shared_preds"][shared_idx]
    else:
        # Fallback for old results without shared examples
        target_for_comparison = r4["example_targets"][2]
        input_4x_for_comparison = r4["example_inputs"][2]
        pred_4x_for_comparison = r4["example_preds"][2]
        input_8x_for_comparison = r8["example_inputs"][2]
        pred_8x_for_comparison = r8["example_preds"][2]
    plot_acceleration_comparison(
        target=target_for_comparison,
        input_4x=input_4x_for_comparison,
        pred_4x=pred_4x_for_comparison,
        input_8x=input_8x_for_comparison,
        pred_8x=pred_8x_for_comparison,
        output_path=OUTPUT_DIR / "fig_accel_comparison.png",
    )
    count += 1
    print(f"  [{count}] Acceleration comparison")

    # Fig 4: PSNR/SSIM box plots
    plot_psnr_ssim_distribution(
        psnr_4x=r4["per_img_psnr"],
        ssim_4x=r4["per_img_ssim"],
        psnr_8x=r8["per_img_psnr"],
        ssim_8x=r8["per_img_ssim"],
        output_path=OUTPUT_DIR / "fig_psnr_ssim_boxplot.png",
    )
    count += 1
    print(f"  [{count}] PSNR/SSIM box plots")

    # Fig 5: Pixel error distributions
    plot_error_histogram(
        errors_4x=r4["pixel_errors_flat"],
        errors_8x=r8["pixel_errors_flat"],
        output_path=OUTPUT_DIR / "fig_error_histogram.png",
    )
    count += 1
    print(f"  [{count}] Error histogram")

    # =================================================================
    # 2. CONFORMAL PREDICTION FIGURES
    # =================================================================

    # Fig 6a/6b: Conformal intervals (4x and 8x)
    for acc, r in [(4, r4), (8, r8)]:
        idx = 2
        plot_conformal_intervals(
            reconstruction=r["example_preds"][idx],
            lower=r["example_lowers"][idx],
            upper=r["example_uppers"][idx],
            ground_truth=r["example_targets"][idx],
            output_path=OUTPUT_DIR / f"fig_conformal_{acc}x.png",
            acceleration=acc,
        )
        count += 1
    print(f"  [{count}] Conformal interval figures")

    # Fig 6c/6d: Conformal intervals for WORST and BEST examples
    for label, idx in [("worst", 0), ("best", -1)]:
        plot_conformal_intervals(
            reconstruction=r4["example_preds"][idx],
            lower=r4["example_lowers"][idx],
            upper=r4["example_uppers"][idx],
            ground_truth=r4["example_targets"][idx],
            output_path=OUTPUT_DIR / f"fig_conformal_4x_{label}.png",
            acceleration=4,
        )
        count += 1
    print(f"  [{count}] Conformal intervals (worst/best)")

    # Fig 7: THE KEY FIGURE - Calibration curve (CP vs MC Dropout)
    # Combined (using 4x data as primary)
    plot_calibration_curve(
        nominal_cp=r4["cp_cal_nominal"].tolist(),
        empirical_cp=r4["cp_cal_empirical"].tolist(),
        nominal_mc=r4["mc_cal_nominal"].tolist(),
        empirical_mc=r4["mc_cal_empirical"].tolist(),
        output_path=OUTPUT_DIR / "fig_calibration_4x.png",
        title="Calibration: Conformal vs MC Dropout (4x)",
    )
    count += 1
    # Also for 8x
    plot_calibration_curve(
        nominal_cp=r8["cp_cal_nominal"].tolist(),
        empirical_cp=r8["cp_cal_empirical"].tolist(),
        nominal_mc=r8["mc_cal_nominal"].tolist(),
        empirical_mc=r8["mc_cal_empirical"].tolist(),
        output_path=OUTPUT_DIR / "fig_calibration_8x.png",
        title="Calibration: Conformal vs MC Dropout (8x)",
    )
    count += 1
    print(f"  [{count}] Calibration curves")

    # Fig 8: Coverage histograms
    for acc, r in [(4, r4), (8, r8)]:
        plot_coverage_histogram(
            per_img_coverage=r["per_img_coverage"],
            target_coverage=0.9,
            output_path=OUTPUT_DIR / f"fig_coverage_hist_{acc}x.png",
            acceleration=acc,
        )
        count += 1
    print(f"  [{count}] Coverage histograms")

    # Fig 9: CP vs MC Dropout intervals side-by-side
    for acc, r in [(4, r4), (8, r8)]:
        idx = 2
        plot_cp_vs_mc_intervals(
            reconstruction=r["example_preds"][idx],
            target=r["example_targets"][idx],
            cp_lower=r["example_lowers"][idx],
            cp_upper=r["example_uppers"][idx],
            mc_mean=r["mc_example_means"][min(idx, len(r["mc_example_means"]) - 1)],
            mc_variance=r["mc_example_variances"][min(idx, len(r["mc_example_variances"]) - 1)],
            output_path=OUTPUT_DIR / f"fig_cp_vs_mc_{acc}x.png",
        )
        count += 1
    print(f"  [{count}] CP vs MC Dropout intervals")

    # Fig 10: Interval width comparison 4x vs 8x
    # The interval width is uniform (q_hat) so show as scalar comparison
    width_4x = r4["example_uppers"][2] - r4["example_lowers"][2]
    width_8x = r8["example_uppers"][2] - r8["example_lowers"][2]
    plot_interval_width_comparison(
        width_map_4x=width_4x,
        width_map_8x=width_8x,
        output_path=OUTPUT_DIR / "fig_interval_width_comparison.png",
    )
    count += 1
    print(f"  [{count}] Interval width comparison")

    # =================================================================
    # 2b. ADAPTIVE CONFORMAL PREDICTION FIGURES
    # =================================================================

    # Fig: Adaptive vs Uniform intervals side-by-side
    for acc, r in [(4, r4), (8, r8)]:
        idx = 2
        if "example_adaptive_lowers" in r:
            plot_adaptive_vs_uniform_intervals(
                reconstruction=r["example_preds"][idx],
                target=r["example_targets"][idx],
                uniform_lower=r["example_lowers"][idx],
                uniform_upper=r["example_uppers"][idx],
                adaptive_lower=r["example_adaptive_lowers"][idx],
                adaptive_upper=r["example_adaptive_uppers"][idx],
                output_path=OUTPUT_DIR / f"fig_adaptive_vs_uniform_{acc}x.png",
                acceleration=acc,
            )
            count += 1
    print(f"  [{count}] Adaptive vs uniform intervals")

    # Fig: Adaptive interval width histogram
    if "example_adaptive_widths" in r4:
        plot_adaptive_width_histogram(
            uniform_width=float(r4["cp_interval_width"]),
            adaptive_widths_4x=r4["example_adaptive_widths"][2],
            adaptive_widths_8x=r8["example_adaptive_widths"][2],
            uniform_width_8x=float(r8["cp_interval_width"]),
            output_path=OUTPUT_DIR / "fig_adaptive_width_histogram.png",
        )
        count += 1
    print(f"  [{count}] Adaptive width histogram")

    # Fig: Three-way calibration curve (uniform CP vs adaptive CP vs MC)
    if "adaptive_cal_nominal" in r4:
        for acc, r in [(4, r4), (8, r8)]:
            plot_calibration_three_way(
                nominal_uniform=r["cp_cal_nominal"].tolist(),
                empirical_uniform=r["cp_cal_empirical"].tolist(),
                nominal_adaptive=r["adaptive_cal_nominal"].tolist(),
                empirical_adaptive=r["adaptive_cal_empirical"].tolist(),
                nominal_mc=r["mc_cal_nominal"].tolist(),
                empirical_mc=r["mc_cal_empirical"].tolist(),
                output_path=OUTPUT_DIR / f"fig_calibration_three_way_{acc}x.png",
                title=f"Calibration: Uniform vs Adaptive CP vs MC Dropout ({acc}x)",
            )
            count += 1
    print(f"  [{count}] Three-way calibration curves")

    # Fig: Adaptive coverage histogram
    if "adaptive_per_img_coverage" in r4:
        for acc, r in [(4, r4), (8, r8)]:
            plot_coverage_histogram(
                per_img_coverage=r["adaptive_per_img_coverage"],
                target_coverage=0.9,
                output_path=OUTPUT_DIR / f"fig_adaptive_coverage_hist_{acc}x.png",
                acceleration=acc,
            )
            count += 1
    print(f"  [{count}] Adaptive coverage histograms")

    # =================================================================
    # 3. UNCERTAINTY FIGURES
    # =================================================================

    # Fig 11: MC Dropout uncertainty maps
    for acc, r in [(4, r4), (8, r8)]:
        idx = min(2, len(r["mc_example_means"]) - 1)
        plot_mc_uncertainty_maps(
            reconstruction=r["example_preds"][idx],
            mc_mean=r["mc_example_means"][idx],
            mc_variance=r["mc_example_variances"][idx],
            target=r["example_targets"][idx],
            output_path=OUTPUT_DIR / f"fig_mc_uncertainty_{acc}x.png",
        )
        count += 1
    print(f"  [{count}] MC uncertainty maps")

    # Fig 12: Uncertainty vs error (MC variance vs actual error)
    for acc, r in [(4, r4), (8, r8)]:
        idx = 0  # Use first example
        plot_uncertainty_vs_error(
            uncertainty=r["mc_example_variances"][idx],
            error=r["example_errors"][min(idx, len(r["example_errors"]) - 1)],
            output_path=OUTPUT_DIR / f"fig_unc_vs_err_{acc}x.png",
            title=f"MC Dropout Uncertainty vs Error ({acc}x)",
        )
        count += 1
    print(f"  [{count}] Uncertainty vs error")

    # =================================================================
    # 4. K-SPACE CONSISTENCY FIGURES
    # =================================================================

    # Fig 13: K-space consistency comparison (4x vs 8x)
    plot_kspace_consistency(
        residual_4x=r4["ks_example_residuals"][0],
        residual_8x=r8["ks_example_residuals"][0],
        error_4x=r4["ks_example_errors"][0],
        error_8x=r8["ks_example_errors"][0],
        output_path=OUTPUT_DIR / "fig_kspace_consistency.png",
    )
    count += 1
    print(f"  [{count}] K-space consistency maps")

    # Fig 14: K-space detail for individual examples
    for acc, r in [(4, r4), (8, r8)]:
        for i in range(min(3, len(r["ks_example_residuals"]))):
            plot_kspace_detail(
                reconstruction=r["example_preds"][i],
                target=r["example_targets"][i],
                residual=r["ks_example_residuals"][i],
                error=r["ks_example_errors"][i],
                mask=r["example_masks"][i],
                output_path=OUTPUT_DIR / f"fig_kspace_detail_{acc}x_ex{i}.png",
                acceleration=acc,
            )
            count += 1
    print(f"  [{count}] K-space detail figures")

    # Fig 15: K-space score vs reconstruction error scatter
    for acc, r in [(4, r4), (8, r8)]:
        # Compute per-image mean error from per_img_nmse as proxy
        per_img_mean_error = r["per_img_nmse"]
        n = min(len(r["ks_scores"]), len(per_img_mean_error))
        plot_kspace_scatter(
            ks_scores=r["ks_scores"][:n],
            per_img_errors=per_img_mean_error[:n],
            output_path=OUTPUT_DIR / f"fig_kspace_scatter_{acc}x.png",
        )
        count += 1
    print(f"  [{count}] K-space scatter plots")

    # Fig 16: K-space score distributions
    plot_kspace_score_distribution(
        scores_4x=r4["ks_scores"],
        scores_8x=r8["ks_scores"],
        output_path=OUTPUT_DIR / "fig_kspace_score_dist.png",
    )
    count += 1
    print(f"  [{count}] K-space score distribution")

    # =================================================================
    # 5. COMPREHENSIVE FIGURES
    # =================================================================

    # Fig 17: Trustworthiness dashboard (all signals, one slice)
    for acc, r in [(4, r4), (8, r8)]:
        idx = 2
        mc_idx = min(idx, len(r["mc_example_variances"]) - 1)
        ks_idx = min(idx, len(r["ks_example_residuals"]) - 1)
        plot_trustworthiness_dashboard(
            target=r["example_targets"][idx],
            zero_filled=r["example_inputs"][idx],
            reconstruction=r["example_preds"][idx],
            cp_lower=r["example_lowers"][idx],
            cp_upper=r["example_uppers"][idx],
            mc_variance=r["mc_example_variances"][mc_idx],
            ks_residual=r["ks_example_residuals"][ks_idx],
            output_path=OUTPUT_DIR / f"fig_dashboard_{acc}x.png",
            acceleration=acc,
        )
        count += 1
    print(f"  [{count}] Trustworthiness dashboards")

    # Fig 18: Mask visualization
    plot_mask_visualization(
        mask_4x=r4["example_masks"][0],
        mask_8x=r8["example_masks"][0],
        output_path=OUTPUT_DIR / "fig_masks.png",
    )
    count += 1
    print(f"  [{count}] Mask visualization")

    # Fig 19: Summary table
    plot_summary_table(r4, r8, OUTPUT_DIR / "fig_summary_table.png")
    count += 1
    print(f"  [{count}] Summary table")

    # =================================================================
    # DONE
    # =================================================================

    print(f"\n{'='*60}")
    print(f"Generated {count} figures (PNG + PDF) in {OUTPUT_DIR}")
    print(f"{'='*60}")

    # List all generated files
    all_files = sorted(OUTPUT_DIR.glob("fig_*.png"))
    print(f"\nFigure inventory ({len(all_files)} PNG files):")
    for f in all_files:
        pdf = f.with_suffix(".pdf")
        pdf_status = "+" if pdf.exists() else "-"
        print(f"  [{pdf_status}pdf] {f.name}")


if __name__ == "__main__":
    main()
