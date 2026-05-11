# 3D medical image classification, regression, and ordinal regression repository
<sub>Copyright German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its license.<sub>

This repository extends and builds on [constantinulrich/SSL3D_classification](https://github.com/constantinulrich/SSL3D_classification), extended to support **regression** and **ordinal regression** tasks in addition to the original classification setup. The upstream repository in turn builds on the [IMAGE CLASSIFICATION FRAMEWORK BY HELMHOLTZ IMAGING](https://github.com/MIC-DKFZ/image_classification) and supports fine-tuning checkpoints from [nnssl](https://github.com/MIC-DKFZ/nnssl).

The main additions over upstream:
- A new `Regression` task with MSE loss.
- A new `Ordinal_Regression` task using the [CORAL](https://arxiv.org/abs/1901.07884) formulation, with several alternative ordinal losses (focal, top-k, weighted BCE, BCE+MAE, etc.) selectable via a `loss_fn` config field.
- New model variants: `ResEncoder_Regressor` (plain regression head) and `ResEncoder_OrdinalRegressor` (CORAL-style head).
- Inference script `inference_ord_reg_last_ckpt.py` for ordinal regression checkpoints.

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

Two preprocessing scripts are provided, one per modality: [`CT_preprocessing.py`](/datasets/preprocess_3D_data/datasets/CT_preprocessing.py) for CT and [`MRI_preprocessing.py`](/datasets/preprocess_3D_data/datasets/MRI_preprocessing.py) for MRI. Both take a directory of `.nii.gz` images, resample to a target spacing (default 1mm isotropic), crop to the non-zero bounding box, normalize, and save as Blosc2 (`.b2nd`) — preserving the full resampled volume. Random 160³ patch extraction happens at training time via `batchgenerators`, not during preprocessing.

See the per-modality sections below for the normalization details, which differ between CT and MRI.

## CT preprocessing

For CT datasets, use [`CT_preprocessing.py`](/datasets/preprocess_3D_data/datasets/CT_preprocessing.py). The script is dataset-agnostic and works on any CT dataset given a directory of `.nii.gz` images.

The pipeline runs in two passes:

**Pass 1 — Compute dataset-wide intensity statistics.** Scans all `.nii.gz` files once, sampling up to 10,000 foreground voxels per case (HU > -500). Aggregates global mean, std, and 0.5 / 99.5 percentiles. You can skip this pass by supplying stats directly via `--stats-mean`, `--stats-std`, `--stats-pct-00-5`, and `--stats-pct-99-5`.

**Pass 2 — Per-case processing.**
1. Resample to a target spacing (default 1×1×1 mm isotropic). Cases already at the target spacing skip this step automatically.
2. Crop to the non-zero bounding box (trims zero-padded edges; CT air at -1000 HU is preserved).
3. CT-normalize: clip to the dataset-wide `[percentile_00_5, percentile_99_5]` range, then z-score using the dataset-wide mean and std.
4. Save as Blosc2 at `<out-root>/<dataset-name>/<image_id>.b2nd`.

### Example usage

```bash
python datasets/preprocess_3D_data/datasets/CT_preprocessing.py \
    --in-dir /path/to/raw/CT/images \
    --out-root /path/to/nnssl_preprocessed \
    --dataset-name Dataset001_LiverROI \
    --target-spacing 1 1 1 \
    --num-workers 8
```

### Arguments

| Flag | Description |
|---|---|
| `--in-dir` | Directory of raw `.nii.gz` CT images. |
| `--out-root` | Output root. The script writes to `<out-root>/<dataset-name>/<image_id>.b2nd`. |
| `--dataset-name` | Name of the dataset folder (e.g. `Dataset001_LiverROI`). |
| `--target-spacing Z Y X` | Target voxel spacing in mm. Default `1 1 1`. |
| `--skip-resample` | Skip the resampling step entirely (use native spacing). |
| `--num-workers` | Parallel processes for both passes. Default `8`. |
| `--stats-mean / --stats-std / --stats-pct-00-5 / --stats-pct-99-5` | Optional pre-supplied stats; bypasses Pass 1. All four must be set together. |

## MRI preprocessing

For MRI datasets, use [`MRI_preprocessing.py`](/datasets/preprocess_3D_data/datasets/MRI_preprocessing.py). The structure mirrors `CT_preprocessing.py` but with two key differences:

- **No dataset-wide stats pass.** MRI has no absolute intensity reference (unlike CT's HU scale), so intensities are not comparable across scanners or sequences. Each case is z-scored independently on its own foreground.
- **Foreground = voxels > 0.** This assumes the input MRIs are skull-stripped or otherwise have a zero background. For data with non-zero air background, mask it out first.

The per-case pipeline:
1. Resample to a target spacing (default 1×1×1 mm isotropic).
2. Crop to the non-zero bounding box (typically trims a sizeable margin on skull-stripped MRI).
3. Per-case z-score normalization on the foreground mask.
4. Save as Blosc2 at `<out-root>/<dataset-name>/<image_id>.b2nd`.

### Example usage

```bash
python datasets/preprocess_3D_data/datasets/MRI_preprocessing.py \
    --in-dir /path/to/raw/MRI/images \
    --out-root /path/to/nnssl_preprocessed \
    --dataset-name Dataset017_OpenNeuro \
    --target-spacing 1 1 1 \
    --num-workers 8
```

### Arguments

| Flag | Description |
|---|---|
| `--in-dir` | Directory of raw `.nii.gz` MR images. |
| `--out-root` | Output root. The script writes to `<out-root>/<dataset-name>/<image_id>.b2nd`. |
| `--dataset-name` | Name of the dataset folder (e.g. `Dataset017_OpenNeuro`). |
| `--target-spacing Z Y X` | Target voxel spacing in mm. Default `1 1 1`. |
| `--skip-resample` | Skip the resampling step entirely. |
| `--num-workers` | Parallel processes. Default `8`. |

# Including other datasets

If your dataset can be described as "a directory of preprocessed `.b2nd` files plus a CSV of splits/labels/folds," you don't need to write any new Python — just reuse the existing `AgeReg_DataModule` and point it at your data via the config.

## The common case: reuse `AgeReg_DataModule`

1. **Preprocess your data** with `CT_preprocessing.py` or `MRI_preprocessing.py` to produce `<out-root>/<dataset-name>/<image_id>.b2nd` files.

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

- `task`: one of `'Classification'`, `'Regression'`, `'Ordinal_Regression'`.
- `loss_fn`: name of the loss to use. When `null`, a sensible default is selected per task (see table below).

| `task`              | `loss_fn: null` (default)       | Other valid `loss_fn` values                                                                                  |
|---------------------|----------------------------------|----------------------------------------------------------------------------------------------------------------|
| `Classification`    | `CrossEntropyLoss` (multiclass) / `BCEWithLogitsLoss` (multilabel) | `focal`, `topk10`                                                                                              |
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

# Training
### Primus-M
Fine-tuning:

`python main.py env=cluster model=primus data=Datasetname  trainer.devices=1 model.pretrained=True model.chpt_path=<path/to/checkpoint>`

Training from scratch:

`python main.py env=cluster model=primus data=Datasetname  trainer.devices=1 model.pretrained=False`

### ResEnc-L (classification)
Fine-tuning:

`python main.py env=cluster model=resenc data=Datasetname  trainer.devices=1 model.pretrained=True  model.chpt_path=<path/to/checkpoint>`

Training from scratch:

`python main.py env=cluster model=resenc data=Datasetname trainer.devices=1  model.pretrained=False`

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

### ResEnc-L — Regression
Uses the `ResEncoder_Regressor` model, which pairs the ResEnc-L backbone with a plain `RegressionHead` (pool → dropout → linear). The head emits a single scalar per sample by default; set `num_outputs` in the model config for multi-output regression. In the data config, set `task: 'Regression'`:

```yaml
model:
  task: 'Regression'
  loss_fn: null   # MSELoss
```

> **Note:** the matching model config `cli_configs/model/resenc_reg.yaml` has not been written yet. It can mirror `resenc_ord_reg.yaml` with `_target_: models.resenc.ResEncoder_Regressor`. Once added, training is launched the same way as the other variants:

Fine-tuning:

`python main.py env=cluster model=resenc_reg data=Datasetname trainer.devices=1 model.pretrained=True model.chpt_path=<path/to/checkpoint>`

Training from scratch:

`python main.py env=cluster model=resenc_reg data=Datasetname trainer.devices=1 model.pretrained=False`

# Inference

For ordinal regression checkpoints, use:

`python inference_ord_reg_last_ckpt.py <args>`

(See the script for the exact CLI surface.)


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