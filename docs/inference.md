# Inference

Inference is driven by [`scripts/predict.py`](../scripts/predict.py) and configured via [`configs/infer.yaml`](../configs/infer.yaml). It takes a single training-run directory as input and derives everything else (training config, checkpoints, transforms) from there.

## Required input

| Field | Meaning |
|---|---|
| `run_dir` | Path to a training run directory produced by `scripts/train.py`. Must contain `Configs/config.yaml` (Hydra snapshot) and `folds/<k>/*.ckpt`. |

## Optional inputs

| Field | Default | Meaning |
|---|---|---|
| `fold` | `0` | Which fold's checkpoint to load (`<run_dir>/folds/<fold>/*.ckpt`). |
| `pred_dir` | `<run_dir>/predictions/` | Where to write per-case predictions, age-bin CSVs/plots, and the per-split summary CSV. |
| `data_dir` | `null` | If set, predict on every `.b2nd` under this directory instead of running val + test. |
| `metrics` | `['mae', 'mse']` | Metric names forwarded to the data config when re-instantiating the model. |

## Two modes

### Mode A — `data_dir: null` (default): re-run val + test from the training CSV

For each of `val` and `test`, `predict.py`:
1. Builds the datamodule from the snapshotted training config.
2. Runs `trainer.predict()` and reads ground-truth labels from the CSV.
3. Writes:
   - `predictions_<split>_fold<k>.xlsx` — per-case `PatientID`, `GroundTruth`, `Prediction`, `Error`, `AbsError`.
   - `error_by_age_bin_<split>_fold<k>.csv` + matching `_MAE.png` / `_MeanError.png` bar charts.
   - `summary_<split>.csv` — `N`, `MAE`, `RMSE`, `MeanError`, `Pearson_r` for the split.

This is the mode you want for evaluating a trained model on its own held-out splits.

### Mode B — `data_dir: /some/dir`: predict on unlabeled images

When you set `data_dir` to a directory of preprocessed `.b2nd` files (no CSV needed):
1. `predict.py` writes a synthetic manifest at `<pred_dir>/_predict_manifest.csv` listing every `.b2nd` under `data_dir` with dummy labels (`split: test`, `fold: 0`, `label: 0`).
2. Loads `test_transforms` from the snapshotted training config so preprocessing matches what the model saw at training time.
3. Runs `trainer.predict()` and writes `predictions_data_dir_fold<k>.xlsx` with two columns: `PatientID`, `Prediction`.

No metrics, no age-bin reports — there are no ground-truth labels to compute against. This is the mode you want for running a trained model on new cases.

## Example commands

Default — run val + test from the training CSV, using paths set inside `infer.yaml`:
```bash
python scripts/predict.py
```

Override `run_dir` on the CLI:
```bash
python scripts/predict.py run_dir=/path/to/<output_dir>/<dataset>/<trainer.logger.name>
```

Predict on a directory of new images:
```bash
python scripts/predict.py \
    run_dir=/path/to/<output_dir>/<dataset>/<trainer.logger.name> \
    data_dir=/path/to/new/preprocessed/b2nd/files
```

Pick a different fold:
```bash
python scripts/predict.py fold=2
```
