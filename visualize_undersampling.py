"""Visualize the effect of k-space undersampling at different acceleration factors."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


DATASET_ROOT = Path("dataset/fastmri_pd")
OUTPUT_PATH = Path("dataset/undersampling_demo.png")

ACCELERATIONS = [1, 4, 8]
CENTER_FRACTION = 0.08
SEED = 42


def load_rss(h5_path: Path) -> np.ndarray:
    """Load the RSS ground truth image."""
    with h5py.File(h5_path, "r") as f:
        return f["image_rss"][:]


def create_cartesian_mask(
    width: int,
    acceleration: int,
    center_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a 1D Cartesian undersampling mask.

    Always keeps center lines. Randomly samples remaining lines
    to reach the target acceleration factor.
    """
    mask = np.zeros(width, dtype=bool)

    # Always keep center lines
    num_center = int(center_fraction * width)
    center_start = (width - num_center) // 2
    mask[center_start : center_start + num_center] = True

    # Randomly sample remaining lines to reach target count
    total_lines = width // acceleration
    remaining = total_lines - num_center
    if remaining > 0:
        available = np.where(~mask)[0]
        chosen = rng.choice(available, size=remaining, replace=False)
        mask[chosen] = True

    return mask


def undersample(image: np.ndarray, acceleration: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate undersampled MRI acquisition.

    Returns (zero_filled_image, undersampled_kspace, mask).
    """
    if acceleration == 1:
        kspace = np.fft.fftshift(np.fft.fft2(image))
        mask = np.ones(image.shape[1], dtype=bool)
        return image, kspace, mask

    kspace = np.fft.fftshift(np.fft.fft2(image))
    mask = create_cartesian_mask(image.shape[1], acceleration, CENTER_FRACTION, rng)

    # Apply mask (zero out unacquired columns)
    undersampled_kspace = kspace * mask[np.newaxis, :]

    # Zero-filled reconstruction
    zero_filled = np.abs(np.fft.ifft2(np.fft.ifftshift(undersampled_kspace)))

    return zero_filled, undersampled_kspace, mask


def main() -> None:
    # Pick a mid-volume slice with good anatomy
    h5_dir = DATASET_ROOT / "train" / "h5"
    files = sorted(h5_dir.glob("file1000003_*.h5"))
    ground_truth = load_rss(files[10])  # slice 10 — good knee cross-section

    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(len(ACCELERATIONS), 4, figsize=(16, 4 * len(ACCELERATIONS)))
    fig.suptitle("Effect of K-Space Undersampling on MRI Reconstruction", fontsize=14, fontweight="bold")

    col_titles = ["K-Space Mask", "Undersampled K-Space", "Zero-Filled Recon", "Error vs Ground Truth"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12)

    for row, acc in enumerate(ACCELERATIONS):
        zero_filled, us_kspace, mask = undersample(ground_truth, acc, rng)
        kspace_log = np.log1p(np.abs(us_kspace) / np.abs(us_kspace).max() * 1e4)

        # Normalize for fair comparison
        gt_norm = ground_truth / ground_truth.max()
        zf_norm = zero_filled / zero_filled.max() if zero_filled.max() > 0 else zero_filled
        error = np.abs(gt_norm - zf_norm)

        lines_kept = mask.sum()
        label = f"{acc}x ({lines_kept}/{len(mask)} lines)" if acc > 1 else f"Fully sampled ({lines_kept} lines)"

        # Mask visualization — expand to 2D for visibility
        mask_2d = np.repeat(mask[np.newaxis, :], 60, axis=0)
        axes[row, 0].imshow(mask_2d, cmap="gray", aspect="auto")
        axes[row, 0].set_ylabel(label, fontsize=11, fontweight="bold")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # K-space
        axes[row, 1].imshow(kspace_log, cmap="inferno")
        axes[row, 1].axis("off")

        # Zero-filled reconstruction
        axes[row, 2].imshow(zf_norm, cmap="gray")
        psnr = 10 * np.log10(1.0 / np.mean((gt_norm - zf_norm) ** 2)) if acc > 1 else float("inf")
        psnr_text = f"PSNR: {psnr:.1f} dB" if np.isfinite(psnr) else "Ground Truth"
        axes[row, 2].text(5, 15, psnr_text, color="yellow", fontsize=9, fontweight="bold",
                          bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.7})
        axes[row, 2].axis("off")

        # Error map
        im_err = axes[row, 3].imshow(error, cmap="hot", vmin=0, vmax=0.3)
        axes[row, 3].axis("off")

    fig.colorbar(im_err, ax=axes[:, 3], label="Absolute Error", shrink=0.6)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
