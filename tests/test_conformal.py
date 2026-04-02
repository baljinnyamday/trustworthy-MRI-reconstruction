"""Tests for conformal prediction."""

import numpy as np

from src.mri.conformal import (
    adaptive_coverage_from_arrays,
    compute_adaptive_scores,
    compute_quantile,
    evaluate_coverage,
)


def test_quantile_known_distribution() -> None:
    # Uniform [0, 1] scores: 90th percentile should be ~0.9
    rng = np.random.default_rng(42)
    scores = rng.uniform(0, 1, size=10000)
    q = compute_quantile(scores, alpha=0.1)
    assert 0.88 < q < 0.92, f"Expected ~0.9, got {q}"


def test_quantile_decreases_with_alpha() -> None:
    rng = np.random.default_rng(42)
    scores = rng.uniform(0, 1, size=10000)
    q_90 = compute_quantile(scores, alpha=0.1)  # 90% coverage
    q_80 = compute_quantile(scores, alpha=0.2)  # 80% coverage
    assert q_90 > q_80, "Higher coverage should require wider intervals"


def test_coverage_guarantee() -> None:
    # Simulate: target = pred + noise, nonconformity scores = |noise|
    rng = np.random.default_rng(42)
    n_cal = 1000
    n_test = 500

    noise_cal = rng.normal(0, 1, size=(n_cal, 32, 32))
    scores_cal = np.abs(noise_cal)
    q_hat = compute_quantile(scores_cal, alpha=0.1)

    # Generate test data
    preds = rng.uniform(0, 1, size=(n_test, 32, 32))
    noise_test = rng.normal(0, 1, size=(n_test, 32, 32))
    targets = preds + noise_test

    results = [
        {
            "pred": preds[i],
            "lower": preds[i] - q_hat,
            "upper": preds[i] + q_hat,
            "target": targets[i],
        }
        for i in range(n_test)
    ]

    metrics = evaluate_coverage(results)
    assert metrics["coverage"] >= 0.89, (
        f"Coverage should be >= 90%, got {metrics['coverage']:.2%}"
    )


def test_evaluate_coverage_perfect() -> None:
    # With very wide intervals, coverage should be 100%
    results = [
        {
            "pred": np.zeros((32, 32)),
            "lower": np.full((32, 32), -1000.0),
            "upper": np.full((32, 32), 1000.0),
            "target": np.random.randn(32, 32),
        }
    ]
    metrics = evaluate_coverage(results)
    assert metrics["coverage"] == 1.0


# --- Adaptive conformal prediction tests ---


def test_adaptive_scores_normalises_by_sigma() -> None:
    residuals = np.array([[[0.1, 0.2], [0.3, 0.4]]])
    sigma = np.array([[[1.0, 0.5], [0.1, 2.0]]])
    scores = compute_adaptive_scores(residuals, sigma, smooth=0)
    expected = np.array([[[0.1, 0.4], [3.0, 0.2]]])
    np.testing.assert_allclose(scores, expected, atol=1e-6)


def test_adaptive_scores_epsilon_floor() -> None:
    residuals = np.array([[[0.5]]])
    sigma = np.array([[[0.0]]])  # would cause division by zero
    scores = compute_adaptive_scores(residuals, sigma, epsilon=0.01, smooth=0)
    # 0.5 / 0.01 = 50.0
    np.testing.assert_allclose(scores, [[[50.0]]], atol=1e-6)


def test_adaptive_coverage_guarantee() -> None:
    """Adaptive CP should maintain coverage guarantee with heterogeneous sigma."""
    rng = np.random.default_rng(42)
    n_cal = 1000
    n_test = 500
    h, w = 32, 32

    # Simulate spatially varying difficulty: left half easy, right half hard
    sigma_pattern = np.ones((h, w)) * 0.1
    sigma_pattern[:, w // 2:] = 1.0

    # Calibration data
    cal_preds = rng.uniform(0, 1, size=(n_cal, h, w))
    cal_noise = rng.normal(0, 1, size=(n_cal, h, w)) * sigma_pattern
    cal_targets = cal_preds + cal_noise
    cal_residuals = np.abs(cal_preds - cal_targets)
    cal_sigma = np.broadcast_to(sigma_pattern, (n_cal, h, w)).copy()

    adaptive_scores = compute_adaptive_scores(cal_residuals, cal_sigma)
    q_hat = compute_quantile(adaptive_scores, alpha=0.1)

    # Test data
    test_preds = [rng.uniform(0, 1, size=(h, w)) for _ in range(n_test)]
    test_noise = [rng.normal(0, 1, size=(h, w)) * sigma_pattern for _ in range(n_test)]
    test_targets = [p + n for p, n in zip(test_preds, test_noise)]
    test_sigma = [sigma_pattern.copy() for _ in range(n_test)]

    metrics = adaptive_coverage_from_arrays(
        test_preds, test_targets, test_sigma, q_hat,
    )
    assert metrics["coverage"] >= 0.89, (
        f"Adaptive CP coverage should be >= 90%, got {metrics['coverage']:.2%}"
    )


def test_adaptive_intervals_are_heterogeneous() -> None:
    """Adaptive intervals should be wider in high-sigma regions."""
    h, w = 32, 32

    preds = [np.zeros((h, w))]
    targets = [np.zeros((h, w))]

    # 75% low sigma, 25% high sigma — median pixel should be low-sigma
    sigma = np.ones((h, w)) * 0.01
    sigma[:, 3 * w // 4:] = 1.0

    metrics = adaptive_coverage_from_arrays(
        preds, targets, [sigma], q_hat=2.0,
    )
    # Low region: 2 * 2.0 * 0.01 = 0.04, High region: 2 * 2.0 * 1.0 = 4.0
    # Mean ≈ 0.75 * 0.04 + 0.25 * 4.0 = 1.03
    assert metrics["mean_interval_width"] > 0.5
    # Median per-image width reflects majority low-sigma pixels
    assert metrics["median_interval_width"] < metrics["mean_interval_width"]
