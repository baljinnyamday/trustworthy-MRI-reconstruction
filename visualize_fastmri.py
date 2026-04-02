"""Visualize fastMRI proton-density (PD) knee dataset slices.

Generates a PNG showing RSS magnitude, complex magnitude, and phase
for a sample of slices from one volume.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


DATASET_ROOT = Path("dataset/fastmri_pd")
OUTPUT_PATH = Path("dataset/fastmri_preview.png")

SLICES_TO_SHOW = [0, 5, 10, 15, 19]


def load_slice(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load RSS and complex images from a single-slice HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        rss = f["image_rss"][:]
        cx = f["image_complex"][:]
    return rss, cx


def find_volume_slices(split: str = "train", max_slices: int = 20) -> list[Path]:
    """Find all slice files for the first volume in a split."""
    h5_dir = DATASET_ROOT / split / "h5"
    files = sorted(h5_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {h5_dir}")

    # Extract volume name from first file (e.g. "file1000003" from "file1000003_000.h5")
    first_volume = files[0].stem.rsplit("_", 1)[0]

    volume_files = [f for f in files if f.stem.startswith(first_volume + "_")]
    return volume_files[:max_slices]


def main() -> None:
    volume_files = find_volume_slices()
    volume_name = volume_files[0].stem.rsplit("_", 1)[0]

    selected = [volume_files[i] for i in SLICES_TO_SHOW if i < len(volume_files)]
    n_slices = len(selected)

    fig, axes = plt.subplots(n_slices, 3, figsize=(12, 4 * n_slices), constrained_layout=True)
    fig.suptitle(f"fastMRI PD Knee — Volume: {volume_name}", fontsize=14, fontweight="bold")

    col_titles = ["RSS Magnitude (Reconstructed)", "K-Space (Fourier Domain)", "Phase"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12)

    for row, h5_path in enumerate(selected):
        rss, cx = load_slice(h5_path)
        kspace = np.fft.fftshift(np.fft.fft2(rss))
        kspace_mag = np.abs(kspace)
        kspace_log = np.log1p(kspace_mag / kspace_mag.max() * 1e4)
        phase = np.angle(cx)
        slice_idx = h5_path.stem.rsplit("_", 1)[1]

        axes[row, 0].imshow(rss, cmap="gray")
        axes[row, 0].set_ylabel(f"Slice {slice_idx}", fontsize=11)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        axes[row, 1].imshow(kspace_log, cmap="inferno", vmin=kspace_log.min(), vmax=kspace_log.max())
        axes[row, 1].axis("off")


        im_phase = axes[row, 2].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[row, 2].axis("off")

    fig.colorbar(im_phase, ax=axes[:, 2], label="Phase (rad)", shrink=0.6)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

    # Print summary stats
    rss_sample, cx_sample = load_slice(selected[len(selected) // 2])
    print(f"\n--- Dataset Summary ---")
    print(f"Location:    {DATASET_ROOT}")
    print(f"Format:      HDF5 (.h5), one slice per file")
    print(f"Splits:      train (9680 slices, 484 volumes), val (2000 slices, 100 volumes)")
    print(f"Image size:  {rss_sample.shape[0]}x{rss_sample.shape[1]}")
    print(f"Channels:    image_rss (float32), image_complex (complex64)")
    print(f"RSS range:   [{rss_sample.min():.6f}, {rss_sample.max():.6f}]")
    print(f"Mag range:   [{np.abs(cx_sample).min():.6f}, {np.abs(cx_sample).max():.6f}]")


if __name__ == "__main__":
    main()
