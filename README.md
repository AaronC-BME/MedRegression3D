# 3D medical image regression and ordinal regression repository
<sub>Copyright German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its license.<sub>

This repository extends and builds on [constantinulrich/SSL3D_classification](https://github.com/constantinulrich/SSL3D_classification), refocused on **regression** and **ordinal regression** tasks. The upstream repository in turn builds on the [IMAGE CLASSIFICATION FRAMEWORK BY HELMHOLTZ IMAGING](https://github.com/MIC-DKFZ/image_classification) and supports fine-tuning checkpoints from [nnssl](https://github.com/MIC-DKFZ/nnssl).

The main differences from upstream:
- A `Regression` task with MSE loss.
- An `Ordinal_Regression` task using the [CORAL](https://arxiv.org/abs/1901.07884) formulation, with several alternative ordinal losses (focal, top-k, weighted BCE, BCE+MAE, etc.) selectable via a `loss_fn` config field.
- Model variants: `ResEncoder_Regressor` (plain regression head), `ResEncoder_OrdinalRegressor` (CORAL-style head), and `ResEncoder_OrdinalRegressor_MLP` (CORAL head with an MLP projection).
- Inference script `scripts/predict.py` that either re-runs val + test from the training CSV (with metrics + age-bin reports) or predicts on a directory of unlabeled `.b2nd` images.
- The original `Classification` task and its associated models/losses/metrics have been removed.

## Documentation

| Topic | File |
|---|---|
| CSV schema for splits / labels / folds | [docs/data-csv-format.md](docs/data-csv-format.md) |
| CT preprocessing — pipeline details | [docs/preprocessing-ct.md](docs/preprocessing-ct.md) |
| MRI preprocessing — pipeline details | [docs/preprocessing-mri.md](docs/preprocessing-mri.md) |
| Training configs — what each `train_*.yaml` does | [docs/training-configs.md](docs/training-configs.md) |
| Adding your own dataset | [docs/custom-datasets.md](docs/custom-datasets.md) |
| Task choice and loss function options | [docs/tasks-and-losses.md](docs/tasks-and-losses.md) |
| Output directory layout and run naming | [docs/output-layout.md](docs/output-layout.md) |
| Inference (`predict.py`) | [docs/inference.md](docs/inference.md) |

# Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for environment and package management.

```shell
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# From inside the cloned repo
uv venv --python 3.11
source .venv/bin/activate

# Install the project (deps from pyproject.toml + medregression3d itself, editable).
# PyTorch and torchvision are pulled in transitively.
uv pip install -e .
```

To deactivate the env later, run `deactivate`. To resume work, `cd` into the repo and `source .venv/bin/activate` again.

## Verifying the install

```shell
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
```

You should see your torch version, `cuda: True`, and a non-zero device count if you have a GPU available.

# Dataset preprocessing

Two preprocessing scripts, one per modality: [`scripts/preprocess_ct.py`](scripts/preprocess_ct.py) for CT and [`scripts/preprocess_mri.py`](scripts/preprocess_mri.py) for MRI. Both take one or more directories of `.nii.gz` images, resample to a target spacing (defaults to the per-axis median across the input dataset), crop to the non-zero bounding box, normalize, and save as Blosc2 (`.b2nd`). Center 160³ patches are extracted at training time, not during preprocessing.

For the per-modality pipeline details (resampling, intensity statistics, normalization differences), see [docs/preprocessing-ct.md](docs/preprocessing-ct.md) and [docs/preprocessing-mri.md](docs/preprocessing-mri.md).

## Suggested directory layout

The data module doesn't enforce a layout — it just needs `img_dir` (a folder of preprocessed `.b2nd` files) and `csv_file` (the splits/labels CSV). A tidy convention:

```
dataset/
└── <data_name>/
    ├── raw/                <- original .nii.gz files (kept for re-preprocessing)
    ├── preprocessed/       <- .b2nd output from scripts/preprocess_ct.py / scripts/preprocess_mri.py
    └── split_labels.csv    <- splits/labels/folds CSV
```

## CT

```bash
python scripts/preprocess_ct.py \
    --in-dir /path/to/raw/CT/imagesTr /path/to/raw/CT/imagesVal \
    --out-root /path/to/preprocessed_data \
    --dataset-name Dataset001_LiverROI \
    --num-workers 8
```

| Flag | Description |
|---|---|
| `--in-dir` | One or more directories of raw `.nii.gz` CT images. Stats and median spacing span all of them. |
| `--out-root` | Output root. The script writes to `<out-root>/<dataset-name>/<image_id>.b2nd`. |
| `--dataset-name` | Name of the dataset folder (e.g. `Dataset001_LiverROI`). |
| `--target-spacing Z Y X` | Target voxel spacing in mm. **Optional.** If omitted, defaults to the per-axis median spacing across all input images. |
| `--skip-resample` | Skip the resampling step entirely (use native spacing). |
| `--num-workers` | Parallel processes for the stats / spacing / per-case passes. Default `8`. |
| `--stats-mean / --stats-std / --stats-pct-00-5 / --stats-pct-99-5` | Optional pre-supplied stats; bypasses the dataset-wide stats pass. All four must be set together. |

## MRI

```bash
python scripts/preprocess_mri.py \
    --in-dir /path/to/raw/MRI/imagesTr /path/to/raw/MRI/imagesVal \
    --out-root /path/to/preprocessed_data \
    --dataset-name Dataset017_OpenNeuro \
    --num-workers 8
```

| Flag | Description |
|---|---|
| `--in-dir` | One or more directories of raw `.nii.gz` MR images. Median spacing spans all of them. |
| `--out-root` | Output root. The script writes to `<out-root>/<dataset-name>/<image_id>.b2nd`. |
| `--dataset-name` | Name of the dataset folder (e.g. `Dataset017_OpenNeuro`). |
| `--target-spacing Z Y X` | Target voxel spacing in mm. **Optional.** If omitted, defaults to the per-axis median spacing across all input images. |
| `--skip-resample` | Skip the resampling step entirely. |
| `--num-workers` | Parallel processes. Default `8`. |

# Training

Every training run is driven by one **self-contained** config file in `configs/train_*.yaml`. Each config has env + data + model + trainer settings inline — there is no `defaults:` composition. Launch with:

```bash
python scripts/train.py --config-name=<config_name>
```

(or `medreg-train --config-name=<config_name>` — the installed console script does the same thing).

If you forget the flag, `train.py` prints a friendly error listing the available configs.

## Provided configs

| Config | Task | Notes |
|---|---|---|
| `train_age_ord_reg` | `Ordinal_Regression` (CORAL) | Template. Has `<placeholder>` paths to fill in — copy this for a new ordinal-regression experiment. |
| `train_age_reg` | `Regression` (MSE) | Template for plain regression. |

See [docs/training-configs.md](docs/training-configs.md) for what each config sets up and when to pick which.

## Examples

Fill in the placeholders in `train_age_ord_reg.yaml` (or copy it first), then:

```bash
python scripts/train.py --config-name=train_age_ord_reg
```

Override any key on the CLI without editing the file (single GPU, smaller LR, a different ordinal loss):

```bash
python scripts/train.py --config-name=train_age_ord_reg \
    trainer.devices=1 \
    model.lr=5e-4 \
    model.loss_fn=focal
```

Fine-tune from a checkpoint:

```bash
python scripts/train.py --config-name=train_age_ord_reg \
    model.pretrained=True \
    model.chpt_path=/path/to/checkpoint.ckpt
```

## Adding a new experiment

Copy an existing `configs/train_*.yaml` to `configs/train_<your_name>.yaml`, edit the placeholders (img_dir, csv_file, project, name), then `python scripts/train.py --config-name=train_<your_name>`. See [docs/custom-datasets.md](docs/custom-datasets.md) for the full workflow, and [docs/tasks-and-losses.md](docs/tasks-and-losses.md) for the `task` / `loss_fn` options. [docs/output-layout.md](docs/output-layout.md) explains where runs land on disk.

For running inference on a trained model, see [docs/inference.md](docs/inference.md).

---

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
