# Trustworthy MRI Reconstruction: Trustworthy AI for MRI Reconstruction

This repository contains the code, experiments, and paper assets for the MRI reconstruction project.

If you are reviewing this as a TA/instructor: **the core code is in `src/mri/`** and runnable pipelines are in `scripts/`.

## Repository map (what is where)

| Path | Purpose |
|---|---|
| `src/mri/` | Main Python package (data loading, U-Net, training, conformal prediction, metrics, visualization helpers). |
| `scripts/` | End-to-end runnable scripts (train, evaluate, figure generation, smoke tests, ablations). |
| `tests/` | Unit tests for masks, losses, conformal logic, data loading, and model components. |
| `dataset/` | Dataset folder structure + READMEs only (large/raw data is not committed). |
| `notebooks/` | Experiment notebooks and notes. |
| `outputs/` | Generated checkpoints/results/figures from local runs (ignored in Git). |
| `paper/` | LaTeX paper source, bibliography, and paper figures/PDF. |
| `future_works/` | Ideas for paper improvements and next research directions. |

## Reproducibility quick start

Run all commands from repository root.

1. Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

2. Prepare data (see `dataset/README.md`):
   - Put fastMRI `.h5` files under:
     - `dataset/fastmri_pd/train/h5/`
     - `dataset/fastmri_pd/val/h5/`

3. Train models (4x and 8x):

```bash
python scripts/run_train.py
```

This writes checkpoints such as:
- `outputs/checkpoints/best_4x.pt`
- `outputs/checkpoints/best_8x.pt`

4. Run full evaluation pipeline:

```bash
python scripts/run_evaluate.py
```

This saves consolidated results:
- `outputs/results_4x.npz`
- `outputs/results_8x.npz`

5. Generate figures from saved results:

```bash
python scripts/generate_figures.py
```

Figures are written to:
- `outputs/figures/`

## Fast sanity check (small run)

For a smaller end-to-end validation run:

```bash
python scripts/smoke_test.py
```

## How to navigate the code quickly

- Start at `scripts/run_train.py` and `scripts/run_evaluate.py` for the full pipeline.
- Then inspect `src/mri/train.py`, `src/mri/data.py`, and `src/mri/conformal.py` for core methodology.
- Plotting helpers used by scripts are in `src/mri/viz.py`.

## Notes for reviewers

- Large datasets, checkpoints, and outputs are intentionally excluded from GitHub.
- This is expected; use the folder READMEs and commands above to reproduce runs locally.
