# 3D medical image regression and ordinal regression repository
<sub>Copyright German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its license.<sub>

This repository extends and builds on [constantinulrich/SSL3D_classification](https://github.com/constantinulrich/SSL3D_classification), refocused on **regression** and **ordinal regression** tasks. The upstream repository in turn builds on the [IMAGE CLASSIFICATION FRAMEWORK BY HELMHOLTZ IMAGING](https://github.com/MIC-DKFZ/image_classification) and supports fine-tuning checkpoints from [nnssl](https://github.com/MIC-DKFZ/nnssl).

The main differences from upstream:
- A `Regression` task with MSE loss.
- An `Ordinal_Regression` task using the [CORAL](https://arxiv.org/abs/1901.07884) formulation, with several alternative ordinal losses (focal, top-k, weighted BCE, BCE+MAE, etc.) selectable via a `loss_fn` config field.
- Model variants: `ResEncoder_Regressor` (plain regression head), `ResEncoder_OrdinalRegressor` (CORAL-style head), and `ResEncoder_OrdinalRegressor_MLP` (CORAL head with an MLP projection).
- Inference script `scripts/predict.py` that either re-runs val + test from the training CSV (with metrics + age-bin reports) or predicts on a directory of unlabeled `.b2nd` images.
- The original `Classification` task and its associated models/losses/metrics have been removed.

# Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for environment and package management.

```shell
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# From inside the cloned repo
uv venv --python 3.11
source .venv/bin/activate

# Install PyTorch matching your CUDA driver (auto-detected)
uv pip install torch torchvision --torch-backend=auto

# Install the rest of the requirements
uv pip install -r requirements.txt
```

You can pin a specific CUDA build explicitly if `--torch-backend=auto` doesn't pick what you want:

```shell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

To deactivate the env later, run `deactivate`. To resume work, `cd` into the repo and `source .venv/bin/activate` again.

## Verifying the install

```shell
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
```

You should see your torch version, `cuda: True`, and a non-zero device count if you have a GPU available.

# Data CSV format

Splits, labels, and fold assignments are all driven by a single CSV file per dataset. The CSV must contain these columns:

| Column       | Type   | Description                                                                 |
|--------------|--------|-----------------------------------------------------------------------------|
| `image_name` | string | Subject/image identifier. Must match the `.b2nd` filename stem on disk (no extension). |
| `split`      | string | One of `train`, `val`, `test`.                                              |
| `fold`       | int    | Fold index (`0`, `1`, `2`, ...). Used for cross-validation.                 |
| `<label>`    | float  | Target value. Column name is configurable (default `label`); rounded to int internally for ordinal regression. |

Example:

```csv
image_name,split,label,fold
sub-001,train,42.3,0
sub-002,val,67.8,0
sub-003,test,55.1,0
sub-004,train,29.5,1
sub-005,val,71.0,1
```

Notes:
- For ordinal regression, labels are rounded to the nearest integer (`int(round(x))`) and must fall in `[0, num_classes - 1]`.
- Splits are **global** across folds — an image's `split` value applies regardless of which fold is being trained. If you need per-fold splits, you'll need to extend the CSV schema.
- The label column name can be customized via the `label_column` field in the data config (e.g., `label_column: age`).
- **For k-fold cross-validation**, include one row per `(image, fold)` combination, with the `split` column indicating that image's role in that particular fold. With `cv.k=5` in the config, training is launched 5 times, each time filtering the CSV to one fold.

# Dataset preprocessing

Two preprocessing scripts are provided, one per modality: [`scripts/preprocess_ct.py`](scripts/preprocess_ct.py) for CT and [`scripts/preprocess_mri.py`](scripts/preprocess_mri.py) for MRI. Both take one or more directories of `.nii.gz` images, resample to a target spacing (defaults to the per-axis **median** across the input dataset; override with `--target-spacing Z Y X`), crop to the non-zero bounding box, normalize, and save as Blosc2 (`.b2nd`) — preserving the full resampled volume. Center 160³ patches are extracted at training time via `batchgenerators`, not during preprocessing.

Both scripts accept `--in-dir` multiple times in one invocation (e.g. `--in-dir imagesTr imagesVal`), so train/val image folders end up in the same output directory at a consistent spacing.

See the per-modality sections below for the normalization details, which differ between CT and MRI.

## Suggested directory layout

The data module doesn't enforce a layout — it just needs `img_dir` (a folder of preprocessed `.b2nd` files) and `csv_file` (the splits/labels CSV). That said, a tidy convention that keeps everything for one dataset under one folder:

```
dataset/
└── <data_name>/
    ├── raw/                <- original .nii.gz files (kept for re-preprocessing)
    ├── preprocessed/       <- .b2nd output from scripts/preprocess_ct.py / scripts/preprocess_mri.py
    └── split_labels.csv    <- splits/labels/folds CSV
```

Then in the data config:

```yaml
img_dir: dataset/<data_name>/preprocessed
csv_file: dataset/<data_name>/split_labels.csv
```

## CT preprocessing

For CT datasets, use [`scripts/preprocess_ct.py`](scripts/preprocess_ct.py). The script is dataset-agnostic and works on any CT dataset given one or more directories of `.nii.gz` images.

The pipeline runs in two passes:

**Pass 1 — Compute dataset-wide intensity statistics.** Scans all `.nii.gz` files once, sampling up to 10,000 foreground voxels per case (HU > -500). Aggregates global mean, std, and 0.5 / 99.5 percentiles. You can skip this pass by supplying stats directly via `--stats-mean`, `--stats-std`, `--stats-pct-00-5`, and `--stats-pct-99-5`.

**Pass 2 — Per-case processing.**
1. Resample to a target spacing. By default the script reads headers across every input image, takes the per-axis median spacing, and uses that. Override with `--target-spacing Z Y X` to force a specific spacing (e.g. `1 1 1` for 1mm isotropic). Cases already at the target spacing skip resampling automatically.
2. Crop to the non-zero bounding box (trims zero-padded edges; CT air at -1000 HU is preserved).
3. CT-normalize: clip to the dataset-wide `[percentile_00_5, percentile_99_5]` range, then z-score using the dataset-wide mean and std.
4. Save as Blosc2 at `<out-root>/<dataset-name>/<image_id>.b2nd`.

### Example usage

Auto-median spacing across train + val splits:
```bash
python scripts/preprocess_ct.py \
    --in-dir /path/to/raw/CT/imagesTr /path/to/raw/CT/imagesVal \
    --out-root /path/to/preprocessed_data \
    --dataset-name Dataset001_LiverROI \
    --num-workers 8
```

Force 1mm isotropic spacing:
```bash
python scripts/preprocess_ct.py \
    --in-dir /path/to/raw/CT/images \
    --out-root /path/to/preprocessed_data \
    --dataset-name Dataset001_LiverROI \
    --target-spacing 1 1 1 \
    --num-workers 8
```

### Arguments

| Flag | Description |
|---|---|
| `--in-dir` | One or more directories of raw `.nii.gz` CT images. Stats and median spacing span all of them. |
| `--out-root` | Output root. The script writes to `<out-root>/<dataset-name>/<image_id>.b2nd`. |
| `--dataset-name` | Name of the dataset folder (e.g. `Dataset001_LiverROI`). |
| `--target-spacing Z Y X` | Target voxel spacing in mm. **Optional.** If omitted, defaults to the per-axis median spacing across all input images. |
| `--skip-resample` | Skip the resampling step entirely (use native spacing). |
| `--num-workers` | Parallel processes for the stats / spacing / per-case passes. Default `8`. |
| `--stats-mean / --stats-std / --stats-pct-00-5 / --stats-pct-99-5` | Optional pre-supplied stats; bypasses Pass 1. All four must be set together. |

## MRI preprocessing

For MRI datasets, use [`scripts/preprocess_mri.py`](scripts/preprocess_mri.py). The structure mirrors `preprocess_ct.py` but with two key differences:

- **No dataset-wide stats pass.** MRI has no absolute intensity reference (unlike CT's HU scale), so intensities are not comparable across scanners or sequences. Each case is z-scored independently on its own foreground.
- **Foreground = voxels > 0.** This assumes the input MRIs are skull-stripped or otherwise have a zero background. For data with non-zero air background, mask it out first.

The per-case pipeline:
1. Resample to a target spacing. By default the script reads headers across every input image, takes the per-axis median spacing, and uses that. Override with `--target-spacing Z Y X`.
2. Crop to the non-zero bounding box (typically trims a sizeable margin on skull-stripped MRI).
3. Per-case z-score normalization on the foreground mask.
4. Save as Blosc2 at `<out-root>/<dataset-name>/<image_id>.b2nd`.

### Example usage

Auto-median spacing across train + val splits:
```bash
python scripts/preprocess_mri.py \
    --in-dir /path/to/raw/MRI/imagesTr /path/to/raw/MRI/imagesVal \
    --out-root /path/to/preprocessed_data \
    --dataset-name Dataset017_OpenNeuro \
    --num-workers 8
```

Force 1mm isotropic spacing:
```bash
python scripts/preprocess_mri.py \
    --in-dir /path/to/raw/MRI/images \
    --out-root /path/to/preprocessed_data \
    --dataset-name Dataset017_OpenNeuro \
    --target-spacing 1 1 1 \
    --num-workers 8
```

### Arguments

| Flag | Description |
|---|---|
| `--in-dir` | One or more directories of raw `.nii.gz` MR images. Median spacing spans all of them. |
| `--out-root` | Output root. The script writes to `<out-root>/<dataset-name>/<image_id>.b2nd`. |
| `--dataset-name` | Name of the dataset folder (e.g. `Dataset017_OpenNeuro`). |
| `--target-spacing Z Y X` | Target voxel spacing in mm. **Optional.** If omitted, defaults to the per-axis median spacing across all input images. |
| `--skip-resample` | Skip the resampling step entirely. |
| `--num-workers` | Parallel processes. Default `8`. |

# Including other datasets

If your dataset can be described as "a directory of preprocessed `.b2nd` files plus a CSV of splits/labels/folds," you don't need to write any new Python — just reuse the existing `AgeReg_DataModule` and point it at your data via the config.

## The common case: reuse `AgeReg_DataModule`

1. **Preprocess your data** with `scripts/preprocess_ct.py` or `scripts/preprocess_mri.py` to produce `<out-root>/<dataset-name>/<image_id>.b2nd` files.

2. **Build a CSV** with `image_name`, `split`, `fold`, and a label column. See the [Data CSV format](#data-csv-format) section for the schema.

3. **Add a `<your-data>.yaml`** to `cli_configs/data/`. The data config takes explicit paths to the image directory and CSV:

```yaml
# @package _global_
data:
  module:
    _target_: datasets.AgeReg.AgeReg_DataModule
    name: YourDatasetName
    img_dir: /path/to/preprocessed/<dataset-name>
    csv_file: /path/to/splits_labels.csv
    label_column: label   # or "age", etc.
    batch_size: 4
    train_transforms:
      _target_: augmentation.policies.batchgenerators.get_training_transforms
      patch_size: ${data.patch_size}
      rotation_for_DA: 0.523599
      mirror_axes: [0,1,2]
      do_dummy_2d_data_aug: False
    test_transforms:
      _target_: augmentation.policies.batchgenerators.get_test_transforms
      patch_size: ${data.patch_size}
      do_dummy_2d_data_aug: False
  cv:
    k: 1
  num_classes: 100
  patch_size: [160, 160, 160]

model:
  task: 'Ordinal_Regression'
  input_channels: 1
  input_dim: 3
  input_shape: ${data.patch_size}
  optimizer: AdamW
  lr: 0.001
  weight_decay: 1e-2

trainer:
  logger:
    project: YourDatasetName
  accumulate_grad_batches: 48
  max_epochs: 200
  sync_batchnorm: True

metrics:
  - 'mae'
  - 'mse'
```

The first line **must** be `# @package _global_` for Hydra to merge the config correctly.

4. **Run training** as usual:

```bash
python main.py env=cluster model=resenc_ord_reg data=your_data trainer.devices=1
```

You can also override paths from the command line for quick experiments:

```bash
python main.py data=your_data \
    data.module.img_dir=/some/other/path \
    data.module.csv_file=/path/to/other.csv
```

## When you actually need a custom `DataModule`

Write a new `DataModule` only if your data doesn't fit the AgeReg pattern — for example:

- **Multiple images per case** (e.g., paired modalities, multi-channel inputs).
- **Per-fold splits** where the same image is `train` in one fold and `val` in another.
- **Non-standard label structure** (e.g., multi-label classification, segmentation targets, censored survival times).
- **A different file format** than `.b2nd`.

In that case, mirror [`AgeReg.py`](/datasets/AgeReg.py) as a starting point: subclass `BaseDataModule`, accept your paths via `__init__`, and instantiate your `Dataset` in `setup()`. Custom transforms can go in `augmentation/policies/<your-data>.py`, inheriting from `BaseTransform`.

# Tasks and losses

The model config takes two related fields:

- `task`: one of `'Regression'`, `'Ordinal_Regression'`.
- `loss_fn`: name of the loss to use. When `null`, a sensible default is selected per task (see table below).

| `task`              | `loss_fn: null` (default)       | Other valid `loss_fn` values                                                                                  |
|---------------------|----------------------------------|----------------------------------------------------------------------------------------------------------------|
| `Regression`        | `MSELoss`                        | *(none)*                                                                                                       |
| `Ordinal_Regression`| `coral_loss`                     | `focal`, `topk10`, `topk20`, `bce_focal`, `bce_topk10`, `bce_topk20`, `weighted_bce`, `bce_mae`                |

Example ordinal regression-task config block:

```yaml
model:
  task: 'Ordinal_Regression'
  loss_fn: null   # uses CORAL loss by default
  ...
```

For ordinal regression, set `num_classes` in the data config to the number of ordinal levels (e.g. `100` for ages 0–99). The CORAL head emits `num_classes - 1` logits.

# Output layout and run naming

A training run writes everything for one experiment to:

```
<output_dir>/<dataset_name>/<trainer.logger.name>/
├── Configs/                  <- Hydra configs for the run
└── folds/
    ├── 0/                    <- ModelCheckpoint files for fold 0
    ├── 1/                    <- fold 1, if running CV
    └── ...
```

Two fields drive this:

- **`output_dir`** — the output root (e.g. `/home/jma/outputs`). Set via the env config (see below).
- **`trainer.logger.name`** — the experiment identifier. Used as both the W&B run name and the on-disk folder name. Defaults to a `YYYY-MM-DD_HH-MM-SS` timestamp if you don't set one. Override it on the CLI (`trainer.logger.name=MyExperiment1`) or in a data config to get a stable, human-readable name.

For multi-fold runs, `cli.py` appends `_fold{k}` to the W&B run name per fold while keeping the on-disk folder name unchanged — so all folds share one parent directory but each appears as its own run in W&B.

# Environment configs (`env=local` vs `env=cluster`)

Every training command picks an environment config via `env=...`. The env config sets `output_dir` and tweaks Lightning trainer settings to match the runtime.

| | `env=local` | `env=cluster` |
|---|---|---|
| `output_dir` source | Hard-coded path in [`configs/env/local.yaml`](configs/env/local.yaml) — edit `<path_to_output>` to your local output directory. | Reads the `EXPERIMENT_LOCATION` environment variable. Falls back to the `<path_to_output>` placeholder in [`configs/env/cluster.yaml`](configs/env/cluster.yaml) if you'd rather hard-code it. |
| Progress bar | Rich progress bar (inherited from `train.yaml` default). | TQDM progress bar (better behaved in non-TTY job logs). |
| Typical use | Interactive runs on your workstation. | Batch / SLURM jobs where the output root is set per-job via env var. |

For `env=cluster`, the recommended workflow is to `export EXPERIMENT_LOCATION=/path/to/exp` in your job script before invoking `python main.py …`. If you don't want to use the env var, just replace `<path_to_output>` in `cluster.yaml` with a hard path.

# Training

### ResEnc-L — Ordinal Regression
Uses the `ResEncoder_OrdinalRegressor` model with a CORAL-style head. The data config should set `task: 'Ordinal_Regression'` and `num_classes` to the number of ordinal levels.

Fine-tuning:

`python main.py env=cluster model=resenc_ord_reg data=Datasetname  trainer.devices=1 model.pretrained=True  model.chpt_path=<path/to/checkpoint>`

Training from scratch:

`python main.py env=cluster model=resenc_ord_reg data=Datasetname trainer.devices=1  model.pretrained=False`

An end-to-end age-regression example using OpenNeuro is provided as `data=age_ord_reg`:

`python main.py env=cluster model=resenc_ord_reg data=age_ord_reg trainer.devices=1 model.pretrained=False`

To use a non-default ordinal loss (e.g. focal):

`python main.py env=cluster model=resenc_ord_reg data=age_ord_reg trainer.devices=1 model.pretrained=False model.loss_fn=focal`

An MLP-head variant (`ResEncoder_OrdinalRegressor_MLP`) is also available via `model=resenc_ord_reg_MLP`. It replaces the single-linear CORAL projection with a small MLP and can help when the encoder's pooled features need extra capacity before the ordinal thresholds.

### ResEnc-L — Regression
Uses the `ResEncoder_Regressor` model, which pairs the ResEnc-L backbone with a plain `RegressionHead` (pool → dropout → linear). The head emits a single scalar per sample by default; set `num_outputs` in the model config for multi-output regression. In the data config, set `task: 'Regression'`:

```yaml
model:
  task: 'Regression'
  loss_fn: null   # MSELoss
```

Fine-tuning:

`python main.py env=cluster model=resenc_reg data=Datasetname trainer.devices=1 model.pretrained=True model.chpt_path=<path/to/checkpoint>`

Training from scratch:

`python main.py env=cluster model=resenc_reg data=Datasetname trainer.devices=1 model.pretrained=False`

# Inference

Inference is driven by [`scripts/predict.py`](scripts/predict.py) and configured via [`configs/infer.yaml`](configs/infer.yaml). It takes a single training-run directory as input and derives everything else (training config, checkpoints, transforms) from there.

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


**If you use this codebase, please cite:**
```
   @misc{Openmind,
   title={An OpenMind for 3D medical vision self-supervised learning},
   author={Tassilo Wald and Constantin Ulrich and Jonathan Suprijadi and Sebastian Ziegler and Michal Nohel and Robin Peretzke and Gregor Köhler and Klaus H. Maier-Hein},
   year={2025},
   eprint={2412.17041},
   archivePrefix={arXiv},
   primaryClass={cs.CV},
   url={https://arxiv.org/abs/2412.17041},
   }
```