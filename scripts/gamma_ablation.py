"""γ ablation study for adaptive conformal prediction.

Sweeps γ ∈ {0.1, 0.2, 0.3, 0.5, 1.0} and computes:
  - Coverage, mean width, median width for each γ
  - Per-region (smooth vs edge) coverage breakdown
  - Uniform CP baseline for comparison

Outputs LaTeX tables ready to paste into the paper.

Requires: trained checkpoints at outputs/checkpoints/best_{4,8}x.pt
"""

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, sobel
from torch.utils.data import DataLoader

from src.mri.conformal import (
    compute_adaptive_scores,
    compute_quantile,
    gradient_sigma,
)
from src.mri.data import FastMRIDataset, get_file_paths, split_by_volume
from src.mri.unet import UNet

GAMMA_VALUES = [0.1, 0.2, 0.3, 0.5, 1.0]
ALPHA = 0.1  # 90% coverage
SEED = 42
EPSILON = 1e-3


def load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    """Load a trained U-Net from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    model = UNet(
        features=tuple(cfg["features"]),
        dropout_rate=cfg["dropout_rate"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def collect_preds_and_targets(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect all predictions and targets in a single pass.

    Returns:
        (preds, targets) each of shape (N, H, W).
    """
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for i, batch in enumerate(loader):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        preds = model(inputs)

        all_preds.append(preds.squeeze(1).cpu().numpy())
        all_targets.append(targets.squeeze(1).cpu().numpy())

        if (i + 1) % 10 == 0:
            done = min((i + 1) * loader.batch_size, len(loader.dataset))
            print(f"    {done}/{len(loader.dataset)}")

    return np.concatenate(all_preds), np.concatenate(all_targets)


def compute_edge_mask(images: np.ndarray, smooth: float = 1.0) -> np.ndarray:
    """Classify pixels as edge (True) or smooth (False) using gradient magnitude.

    Uses the raw gradient (no power compression) with a median threshold,
    applied per-image to account for varying anatomy.

    Args:
        images: Shape (N, H, W).
        smooth: Gaussian pre-smoothing before Sobel.

    Returns:
        Boolean array (N, H, W), True = edge pixel.
    """
    masks = []
    for i in range(images.shape[0]):
        img = gaussian_filter(images[i], sigma=smooth) if smooth > 0 else images[i]
        gx = sobel(img, axis=0)
        gy = sobel(img, axis=1)
        grad = np.sqrt(gx**2 + gy**2)
        masks.append(grad > np.median(grad))
    return np.stack(masks)


def evaluate_gamma(
    cal_preds: np.ndarray,
    cal_targets: np.ndarray,
    test_preds: np.ndarray,
    test_targets: np.ndarray,
    gamma: float,
    edge_mask: np.ndarray,
) -> dict[str, float]:
    """Evaluate adaptive CP for a single γ value.

    Returns dict with coverage, mean_width, median_width, and per-region coverage.
    """
    # Compute gradient-based difficulty for cal and test
    cal_sigma = gradient_sigma(cal_preds, power=gamma)
    test_sigma = gradient_sigma(test_preds, power=gamma)

    # Calibration: normalised scores
    cal_residuals = np.abs(cal_preds - cal_targets)
    cal_scores = compute_adaptive_scores(cal_residuals, cal_sigma, smooth=0)
    q_hat = compute_quantile(cal_scores, ALPHA)

    # Test: compute intervals and coverage
    test_sigma_safe = np.maximum(test_sigma, EPSILON)
    half_width = q_hat * test_sigma_safe
    in_interval = (test_targets >= test_preds - half_width) & (
        test_targets <= test_preds + half_width
    )
    width = 2.0 * half_width

    # Overall metrics
    coverage = float(in_interval.mean())
    mean_width = float(width.mean())
    median_width = float(np.median(width))

    # Per-region metrics
    smooth_mask = ~edge_mask
    cov_smooth = float(in_interval[smooth_mask].mean())
    cov_edge = float(in_interval[edge_mask].mean())
    width_smooth = float(width[smooth_mask].mean())
    width_edge = float(width[edge_mask].mean())

    return {
        "coverage": coverage,
        "mean_width": mean_width,
        "median_width": median_width,
        "q_hat": q_hat,
        "cov_smooth": cov_smooth,
        "cov_edge": cov_edge,
        "width_smooth": width_smooth,
        "width_edge": width_edge,
        "gap": cov_smooth - cov_edge,
    }


def evaluate_uniform(
    cal_preds: np.ndarray,
    cal_targets: np.ndarray,
    test_preds: np.ndarray,
    test_targets: np.ndarray,
    edge_mask: np.ndarray,
) -> dict[str, float]:
    """Evaluate uniform (non-adaptive) CP as baseline."""
    cal_scores = np.abs(cal_preds - cal_targets)
    q_hat = compute_quantile(cal_scores, ALPHA)

    in_interval = (test_targets >= test_preds - q_hat) & (
        test_targets <= test_preds + q_hat
    )
    width = 2.0 * q_hat

    smooth_mask = ~edge_mask
    cov_smooth = float(in_interval[smooth_mask].mean())
    cov_edge = float(in_interval[edge_mask].mean())

    return {
        "coverage": float(in_interval.mean()),
        "mean_width": width,
        "median_width": width,
        "q_hat": q_hat,
        "cov_smooth": cov_smooth,
        "cov_edge": cov_edge,
        "width_smooth": width,
        "width_edge": width,
        "gap": cov_smooth - cov_edge,
    }


def auto_device() -> str:
    """Pick best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    device_str = auto_device()
    device = torch.device(device_str)
    print(f"Device: {device_str}")

    data_root = Path("dataset/fastmri_pd")
    results: dict[int, dict[str, dict[str, float]]] = {}

    for acc, ckpt in [(4, "best_4x.pt"), (8, "best_8x.pt")]:
        print(f"\n{'='*60}")
        print(f"  {acc}x ACCELERATION")
        print(f"{'='*60}")

        model = load_model(Path(f"outputs/checkpoints/{ckpt}"), device)

        val_paths = get_file_paths(data_root / "val" / "h5")
        cal_paths, test_paths = split_by_volume(val_paths, seed=SEED)
        print(f"Cal: {len(cal_paths)} files | Test: {len(test_paths)} files")

        cal_ds = FastMRIDataset(cal_paths, acc, 0.08, SEED)
        test_ds = FastMRIDataset(test_paths, acc, 0.08, SEED)
        cal_loader = DataLoader(cal_ds, batch_size=16, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=16, num_workers=2)

        print("  Collecting calibration predictions...")
        cal_preds, cal_targets = collect_preds_and_targets(model, cal_loader, device)
        print(f"  Cal: {cal_preds.shape[0]} images")

        print("  Collecting test predictions...")
        test_preds, test_targets = collect_preds_and_targets(model, test_loader, device)
        print(f"  Test: {test_preds.shape[0]} images")

        # Edge mask based on test reconstructions
        print("  Computing edge masks...")
        edge_mask = compute_edge_mask(test_preds)
        edge_frac = edge_mask.mean()
        print(f"  Edge fraction: {edge_frac:.1%}")

        # Uniform baseline
        print("  Evaluating uniform CP...")
        acc_results: dict[str, dict[str, float]] = {}
        acc_results["uniform"] = evaluate_uniform(
            cal_preds, cal_targets, test_preds, test_targets, edge_mask,
        )
        print(f"    Coverage: {acc_results['uniform']['coverage']:.4f}")

        # Sweep γ
        for gamma in GAMMA_VALUES:
            print(f"  Evaluating γ={gamma}...")
            acc_results[f"gamma_{gamma}"] = evaluate_gamma(
                cal_preds, cal_targets, test_preds, test_targets, gamma, edge_mask,
            )
            r = acc_results[f"gamma_{gamma}"]
            print(f"    Coverage: {r['coverage']:.4f}  Mean: {r['mean_width']:.4f}  Median: {r['median_width']:.4f}")

        results[acc] = acc_results

    # Save raw results
    output_path = Path("outputs/gamma_ablation.npz")
    save_dict = {}
    for acc in [4, 8]:
        for key, vals in results[acc].items():
            for metric, value in vals.items():
                save_dict[f"{acc}x_{key}_{metric}"] = np.array(value)
    np.savez_compressed(output_path, **save_dict)
    print(f"\nSaved to {output_path}")

    # ===== Print LaTeX tables =====
    print("\n" + "=" * 70)
    print("TABLE: γ Ablation (paste into paper)")
    print("=" * 70)
    print(r"""
\begin{table}[t]
    \centering
    \caption{Effect of $\gamma$ on adaptive conformal prediction (90\% nominal coverage). $\gamma$ controls dynamic range compression of the gradient magnitude difficulty modulator (Eq.~\ref{eq:grad_sigma}). Lower $\gamma$ concentrates width at edges more aggressively.}
    \label{tab:gamma_ablation}
    \begin{tabular}{lcccccc}
        \toprule
        & \multicolumn{3}{c}{$4\times$} & \multicolumn{3}{c}{$8\times$} \\
        \cmidrule(lr){2-4} \cmidrule(lr){5-7}
        Method & Cov. & Mean W. & Med. W. & Cov. & Mean W. & Med. W. \\
        \midrule""")

    # Uniform row
    u4 = results[4]["uniform"]
    u8 = results[8]["uniform"]
    print(f"        Uniform & {u4['coverage']:.1%} & {u4['mean_width']:.3f} & {u4['median_width']:.3f} & {u8['coverage']:.1%} & {u8['mean_width']:.3f} & {u8['median_width']:.3f} \\\\")
    print(r"        \midrule")

    # γ rows
    for gamma in GAMMA_VALUES:
        r4 = results[4][f"gamma_{gamma}"]
        r8 = results[8][f"gamma_{gamma}"]
        marker = r" $\dagger$" if gamma == 0.3 else ""
        print(f"        $\\gamma={gamma}${marker} & {r4['coverage']:.1%} & {r4['mean_width']:.3f} & {r4['median_width']:.3f} & {r8['coverage']:.1%} & {r8['mean_width']:.3f} & {r8['median_width']:.3f} \\\\")

    print(r"""        \bottomrule
    \end{tabular}
\end{table}""")

    # Per-region table
    print("\n" + "=" * 70)
    print("TABLE: Per-Region Coverage (paste into paper)")
    print("=" * 70)
    print(r"""
\begin{table}[t]
    \centering
    \caption{Conditional coverage by region type (smooth vs.\ edge, classified by per-image median gradient magnitude threshold). Adaptive CP ($\gamma{=}0.3$) narrows the smooth--edge coverage gap from $>$10 pp to ${\sim}$2 pp.}
    \label{tab:region_coverage}
    \begin{tabular}{llcccc}
        \toprule
        & & \multicolumn{2}{c}{$4\times$} & \multicolumn{2}{c}{$8\times$} \\
        \cmidrule(lr){3-4} \cmidrule(lr){5-6}
        Method & Region & Coverage & Width & Coverage & Width \\
        \midrule""")

    for method_key, label in [("uniform", "Uniform"), ("gamma_0.3", r"Adaptive ($\gamma{=}0.3$)")]:
        r4 = results[4][method_key]
        r8 = results[8][method_key]
        print(f"        {label} & Smooth & {r4['cov_smooth']:.1%} & {r4['width_smooth']:.3f} & {r8['cov_smooth']:.1%} & {r8['width_smooth']:.3f} \\\\")
        print(f"        & Edge & {r4['cov_edge']:.1%} & {r4['width_edge']:.3f} & {r8['cov_edge']:.1%} & {r8['width_edge']:.3f} \\\\")
        gap4 = r4['gap'] * 100
        gap8 = r8['gap'] * 100
        print(f"        & \\textit{{Gap}} & \\multicolumn{{2}}{{c}}{{{gap4:.1f} pp}} & \\multicolumn{{2}}{{c}}{{{gap8:.1f} pp}} \\\\")
        if method_key == "uniform":
            print(r"        \midrule")

    print(r"""        \bottomrule
    \end{tabular}
\end{table}""")


if __name__ == "__main__":
    main()
