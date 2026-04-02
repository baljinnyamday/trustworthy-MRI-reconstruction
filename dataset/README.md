# Dataset layout

This repository is prepared for sharing with the professor and paper review.

The project uses two kinds of data:

1. `fastmri_pd/` (raw fastMRI HDF5 files)
2. `processed_data/` (derived `.npz` slices for experiments/visualization)

The full datasets are **not committed** to GitHub because they are large (the current local dataset is ~16 GB) and include data that should be managed via external storage.

Only documentation and folder structure are tracked in `dataset/`.

## Expected structure

```text
dataset/
  fastmri_pd/
    train/h5/
    val/h5/
  processed_data/
    ct_256/
    mr_256/
  visualizations/
```

## How to obtain data

1. Download fastMRI knee single-coil data from the official source.
2. Place `.h5` files in:
   - `dataset/fastmri_pd/train/h5/`
   - `dataset/fastmri_pd/val/h5/`
3. Run the project preprocessing scripts (if needed) to populate `processed_data/`.

## Notes

- Keep personal/temporary files out of this directory.
- `.DS_Store` files are not needed and should not be tracked.
