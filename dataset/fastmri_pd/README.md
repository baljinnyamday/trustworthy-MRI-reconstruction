# fastMRI raw data

Place raw fastMRI `.h5` files here.

Expected layout:

```text
fastmri_pd/
  train/h5/*.h5
  val/h5/*.h5
```

Used by:

- `scripts/run_train.py`
- `scripts/run_evaluate.py`
- `scripts/smoke_test.py`

The actual `.h5` files are ignored in Git.
