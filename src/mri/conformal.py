"""Conformal prediction for MRI reconstruction with coverage guarantees.

Provides pixel-wise prediction intervals that are mathematically guaranteed
to contain the true value at the specified coverage level (e.g., 90%).
"""

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter, sobel
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_nonconformity_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Compute per-pixel absolute residuals on calibration set.

    Args:
        model: Trained reconstruction model.
        loader: DataLoader for calibration set.
        device: Torch device.

    Returns:
        Array of shape (N, H, W) with per-pixel nonconformity scores.
    """
    model.eval()
    all_scores: list[np.ndarray] = []

    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        preds = model(inputs)

        # Per-pixel absolute residual
        scores = torch.abs(preds - targets).squeeze(1).cpu().numpy()
        all_scores.append(scores)

    return np.concatenate(all_scores, axis=0)


def compute_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute the conformal quantile for a given coverage level.

    Uses the finite-sample correction: q = ceil((1-alpha)(1+1/n)) quantile.

    Args:
        scores: Flattened array of nonconformity scores.
        alpha: Miscoverage rate (e.g., 0.1 for 90% coverage).

    Returns:
        The quantile threshold q_hat.
    """
    flat = scores.ravel()
    n = len(flat)
    level = min(math.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(flat, level))


@torch.no_grad()
def predict_with_intervals(
    model: nn.Module,
    loader: DataLoader,
    q_hat: float,
    device: torch.device,
) -> list[dict[str, np.ndarray]]:
    """Generate predictions with conformal prediction intervals.

    Args:
        model: Trained reconstruction model.
        loader: DataLoader for test set.
        q_hat: Conformal quantile threshold.
        device: Torch device.

    Returns:
        List of dicts with 'pred', 'lower', 'upper', 'target' arrays (H, W each).
    """
    model.eval()
    results: list[dict[str, np.ndarray]] = []

    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        preds = model(inputs)

        preds_np = preds.squeeze(1).cpu().numpy()
        targets_np = targets.squeeze(1).cpu().numpy()

        for i in range(preds_np.shape[0]):
            results.append({
                "pred": preds_np[i],
                "lower": preds_np[i] - q_hat,
                "upper": preds_np[i] + q_hat,
                "target": targets_np[i],
            })

    return results


def evaluate_coverage(results: list[dict[str, np.ndarray]]) -> dict[str, float]:
    """Evaluate empirical coverage and interval width.

    Args:
        results: Output from predict_with_intervals.

    Returns:
        Dict with 'coverage' (fraction of pixels covered) and
        'mean_interval_width' (average interval width).
    """
    covered_pixels = 0
    total_pixels = 0
    total_width = 0.0

    for r in results:
        target = r["target"]
        in_interval = (target >= r["lower"]) & (target <= r["upper"])
        covered_pixels += in_interval.sum()
        total_pixels += target.size
        total_width += (r["upper"] - r["lower"]).sum()

    return {
        "coverage": float(covered_pixels / total_pixels),
        "mean_interval_width": float(total_width / total_pixels),
    }


def calibration_curve(
    cal_scores: np.ndarray,
    results_fn,
    alphas: list[float] | None = None,
) -> dict[str, list[float]]:
    """Compute calibration curve: nominal vs empirical coverage.

    Args:
        cal_scores: Nonconformity scores from calibration set.
        results_fn: Callable(q_hat) -> list[dict] that runs prediction at a given q_hat.
        alphas: List of miscoverage rates to evaluate.

    Returns:
        Dict with 'nominal' and 'empirical' coverage lists.
    """
    if alphas is None:
        alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    nominal: list[float] = []
    empirical: list[float] = []

    for alpha in alphas:
        q_hat = compute_quantile(cal_scores, alpha)
        results = results_fn(q_hat)
        metrics = evaluate_coverage(results)
        nominal.append(1 - alpha)
        empirical.append(metrics["coverage"])

    return {"nominal": nominal, "empirical": empirical}


# ---------------------------------------------------------------------------
# Adaptive (normalised) conformal prediction
# ---------------------------------------------------------------------------

DEFAULT_EPSILON = 1e-3
DEFAULT_SMOOTH_SIGMA = 5.0


def smooth_sigma_maps(
    sigma: np.ndarray,
    smooth_sigma: float = DEFAULT_SMOOTH_SIGMA,
) -> np.ndarray:
    """Gaussian-smooth per-pixel difficulty estimates to reduce noise.

    MC Dropout at low p produces noisy per-pixel std. Smoothing preserves
    the structural pattern (edges vs smooth) while averaging out noise.

    Args:
        sigma: Per-pixel difficulty estimate, shape (N, H, W) or (H, W).
        smooth_sigma: Gaussian kernel std in pixels.

    Returns:
        Smoothed sigma, same shape.
    """
    if smooth_sigma <= 0:
        return sigma

    if sigma.ndim == 2:
        return gaussian_filter(sigma, sigma=smooth_sigma)

    return np.stack([
        gaussian_filter(sigma[i], sigma=smooth_sigma)
        for i in range(sigma.shape[0])
    ])


DEFAULT_GRAD_POWER = 0.3


def gradient_sigma(
    image: np.ndarray,
    smooth: float = 1.0,
    power: float = DEFAULT_GRAD_POWER,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Compute gradient-based difficulty estimate for adaptive conformal prediction.

    Uses |∇image|^power as the difficulty modulator. The power < 1 compresses
    the dynamic range so edges don't dominate excessively.

    Args:
        image: Shape (H, W) or (N, H, W).
        smooth: Gaussian pre-smoothing before gradient (reduces noise).
        power: Exponent for dynamic range compression. 0.3 works well empirically.
        epsilon: Small constant added before power to avoid zero^power issues.

    Returns:
        Difficulty estimate, same shape as input.
    """
    def _grad_sigma_2d(img: np.ndarray) -> np.ndarray:
        if smooth > 0:
            img = gaussian_filter(img, sigma=smooth)
        gx = sobel(img, axis=0)
        gy = sobel(img, axis=1)
        grad = np.sqrt(gx**2 + gy**2)
        return np.power(grad + epsilon, power)

    if image.ndim == 2:
        return _grad_sigma_2d(image)

    return np.stack([_grad_sigma_2d(image[i]) for i in range(image.shape[0])])


def compute_adaptive_scores(
    residuals: np.ndarray,
    sigma: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
    smooth: float = DEFAULT_SMOOTH_SIGMA,
) -> np.ndarray:
    """Normalise absolute residuals by a smoothed pixel-wise difficulty estimate.

    score_j = |pred_j - target_j| / max(smooth(sigma_j), epsilon)

    Args:
        residuals: Per-pixel absolute residuals, shape (N, H, W).
        sigma: Per-pixel difficulty estimate (e.g. MC Dropout std), same shape.
        epsilon: Floor to avoid division by near-zero sigma.
        smooth: Gaussian kernel std for smoothing sigma. 0 to disable.

    Returns:
        Normalised scores, same shape as residuals.
    """
    sigma_smooth = smooth_sigma_maps(sigma, smooth)
    sigma_safe = np.maximum(sigma_smooth, epsilon)
    return residuals / sigma_safe


def adaptive_coverage_from_arrays(
    preds: list[np.ndarray],
    targets: list[np.ndarray],
    sigma: list[np.ndarray],
    q_hat: float,
    epsilon: float = DEFAULT_EPSILON,
    smooth: float = DEFAULT_SMOOTH_SIGMA,
) -> dict[str, float | np.ndarray]:
    """Evaluate adaptive conformal coverage with pixel-wise interval widths.

    Interval at pixel j: [pred_j - q_hat * smooth(sigma_j), pred_j + q_hat * smooth(sigma_j)]

    Args:
        preds: List of (H, W) prediction arrays.
        targets: List of (H, W) ground truth arrays.
        sigma: List of (H, W) difficulty estimates (raw, will be smoothed).
        q_hat: Adaptive conformal quantile (from normalised scores).
        epsilon: Floor for sigma values.
        smooth: Gaussian kernel std for smoothing sigma. 0 to disable.

    Returns:
        Dict with 'coverage', 'mean_interval_width', 'median_interval_width',
        and 'per_image_coverage' array.
    """
    covered = 0
    total = 0
    total_width = 0.0
    all_widths: list[float] = []
    per_image_cov: list[float] = []

    for p, t, s in zip(preds, targets, sigma, strict=True):
        s_smooth = smooth_sigma_maps(s, smooth)
        s_safe = np.maximum(s_smooth, epsilon)
        half_width = q_hat * s_safe
        in_interval = (t >= p - half_width) & (t <= p + half_width)

        covered += in_interval.sum()
        total += t.size
        width = 2.0 * half_width
        total_width += width.sum()
        all_widths.append(float(np.median(width).item()))
        per_image_cov.append(float(in_interval.mean()))

    return {
        "coverage": float(covered / total),
        "mean_interval_width": float(total_width / total),
        "median_interval_width": float(np.median(all_widths)),
        "per_image_coverage": np.array(per_image_cov),
    }


def adaptive_calibration_from_arrays(
    cal_residuals: np.ndarray,
    cal_sigma: np.ndarray,
    preds: list[np.ndarray],
    targets: list[np.ndarray],
    sigma: list[np.ndarray],
    alphas: list[float],
    epsilon: float = DEFAULT_EPSILON,
    smooth: float = DEFAULT_SMOOTH_SIGMA,
) -> dict[str, list[float]]:
    """Compute adaptive CP calibration curve without re-running inference.

    Args:
        cal_residuals: Calibration absolute residuals (N_cal, H, W).
        cal_sigma: Calibration difficulty estimates (N_cal, H, W).
        preds: Test predictions.
        targets: Test ground truth.
        sigma: Test difficulty estimates (raw, will be smoothed).
        alphas: Miscoverage rates to sweep.
        epsilon: Floor for sigma values.
        smooth: Gaussian kernel std for smoothing sigma. 0 to disable.

    Returns:
        Dict with 'nominal' and 'empirical' coverage lists.
    """
    adaptive_cal_scores = compute_adaptive_scores(
        cal_residuals, cal_sigma, epsilon, smooth,
    )

    nominal: list[float] = []
    empirical: list[float] = []

    for alpha in alphas:
        q = compute_quantile(adaptive_cal_scores, alpha)
        metrics = adaptive_coverage_from_arrays(
            preds, targets, sigma, q, epsilon, smooth,
        )
        nominal.append(1 - alpha)
        empirical.append(metrics["coverage"])

    return {"nominal": nominal, "empirical": empirical}
