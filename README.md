# 3D medical image classification, regression, and ordinal regression repository
<sub>Copyright German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its license.<sub>

This repository is a fork of [constantinulrich/SSL3D_classification](https://github.com/constantinulrich/SSL3D_classification), extended to support **regression** and **ordinal regression** tasks in addition to the original classification setup. The upstream repository in turn builds on the [IMAGE CLASSIFICATION FRAMEWORK BY HELMHOLTZ IMAGING](https://github.com/MIC-DKFZ/image_classification) and supports fine-tuning checkpoints from [nnssl](https://github.com/MIC-DKFZ/nnssl).

The main additions over upstream:
- A new `Regression` task with MSE loss.
- A new `Ordinal_Regression` task using the [CORAL](https://arxiv.org/abs/1901.07884) formulation, with several alternative ordinal losses (focal, top-k, weighted BCE, BCE+MAE, etc.) selectable via a `loss_fn` config field.
- New model variants: `ResEncoder_Regressor` (plain regression head) and `ResEncoder_OrdinalRegressor` (CORAL-style head).
- Inference script `inference_ord_reg_last_ckpt.py` for ordinal regression checkpoints.

# Installation

This project can be installed with either `pip` (in a virtualenv or conda env) or [`uv`](https://docs.astral.sh/uv/) — a faster Python package manager. Both result in the same set of installed packages.

## Option 1: uv (recommended)

[uv](https://docs.astral.sh/uv/) is significantly faster than pip and handles the Python interpreter for you.

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

You can also pin a specific CUDA build explicitly if `--torch-backend=auto` doesn't pick what you want:

```shell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

To deactivate the env later, run `deactivate`. To resume work, `cd` into the repo and `source .venv/bin/activate` again.

## Option 2: pip

Install the requirements in a [virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```shell
pip install -r requirements.txt
```

You might need to adapt the cuda versions for torch and torchvision.
Find a torch installation guide for your system [here](https://pytorch.org/get-started/locally/).

## Verifying the install

```shell
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
```

You should see your torch version, `cuda: True`, and a non-zero device count if you have a GPU available.


# Dataset preprocessing

Currently, preprocessing is highly dataset- and user-dependent.
However in [this file](/datasets/preprocess_3D_data/datasets/template_brain_preprocessing.py) you can find examples of how a dataset can be preprocessed.

For the SSL3D challenge we will resample all images towards a 1mm target spacing and then crop the center of the image with a 160 cubic block.

## CT preprocessing

For CT datasets, use [`CT_preprocessing.py`](/datasets/preprocess_3D_data/datasets/CT_preprocessing.py). The script is dataset-agnostic and works on any CT dataset given a directory of `.nii.gz` images.

The pipeline per case is:

1. **Compute dataset-wide CT intensity statistics** in a first pass over all `.nii.gz` files in the input directory. Foreground voxels (HU > -500) are subsampled per case (10,000 voxels each) and aggregated to compute global mean, std, and the 0.5 / 99.5 percentiles. Stats are optionally cached to JSON via `--stats-cache` so subsequent runs skip this pass.
2. **Resample to a target spacing** (default 1×1×1 mm isotropic). Cases that are already at the target spacing skip resampling.
3. **Crop to the non-zero bounding box** (matches nnssl behavior — trims explicitly zero-padded edges while preserving all anatomical voxels, since CT air is at -1000 HU rather than 0).
4. **CT normalization**: clip voxel values to the dataset-wide [percentile_00_5, percentile_99_5] range, then z-score using the dataset-wide mean and std. Unlike per-image z-score, this preserves the absolute meaning of HU values across cases.
5. **Save as Blosc2** in the directory layout the dataloader expects:

### Example usage

```bash
python datasets/preprocess_3D_data/datasets/CT_preprocessing.py \
    --in-dir /path/to/raw/CT/images \
    --out-root $nnssl_preprocessed/Dataset001_LiverROI/nnsslPlans_onemmiso \
    --dataset-name Dataset001_LiverROI \
    --target-spacing 1 1 1 \
    --num-workers 8 \
    --stats-cache $nnssl_preprocessed/Dataset001_LiverROI/ct_intensity_stats.json
```

### Arguments

| Flag | Description |
|---|---|
| `--in-dir` | Directory of raw `.nii.gz` CT images. |
| `--out-root` | Output root, e.g. `.../nnsslPlans_onemmiso`. The script writes to `<out-root>/<dataset-name>/<dataset-name>/<id>/ses-DEFAULT/<id>.b2nd`. |
| `--dataset-name` | Name of the dataset folder (e.g. `Dataset001_LiverROI`). |
| `--target-spacing Z Y X` | Target voxel spacing in mm. Default `1 1 1`. |
| `--skip-resample` | Skip the resampling step entirely (use native spacing). |
| `--num-workers` | Parallel processes for both passes. Default `8`. |
| `--stats-cache PATH` | JSON file to cache intensity statistics between runs. |
| `--stats-mean / --stats-std / --stats-pct-00-5 / --stats-pct-99-5` | Optional pre-supplied stats, bypasses the first pass. |

# Including other datasets

For including your own dataset follow these steps:
1. In the ```dataset``` directory create a new file that implements the [torch dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files) class for your data. See [example](/datasets/RECvsT_1mm_cropped_160.py).
2. Additionally, create the [DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) for your dataset by writing a class that inherits from `BaseDataModule`. Write the `init` and `setup` functions for your dataset. The dataloaders are already defined by the `BaseDataModule`. An example could look like this:
    ```python
    from .base_datamodule import BaseDataModule

    class CustomDataModule(BaseDataModule):
      def __init__(self, **params):
          super(CustomDataModule, self).__init__(**params)

      def setup(self, stage: str):
          self.train_dataset = YourCustomPytorchDataset(
              data_path=self.data_path,
              split="train",
              transform=self.train_transforms,
          )
          self.val_dataset = YourCustomPytorchDataset(
              data_path=self.data_path,
              split="val",
              transform=self.test_transforms,
          )
    ```
   Note that the `__init__` function takes `**params` and passes them to the super init. By doing so the attributes `self.data_path`, `self.train_transforms` and `self.test_transforms` are already set automatically and can be used in the `setup` function. The `self.data_path` is a joined path consisting of the configs `data.module.data_root_dir` and `data.module.name`.
   Custom transforms can be added in `./augmentation/policies/<your-data>.py`. They need to inherit from the `BaseTransform` class. See the existing transforms for examples!
3. Add a `<your-data>.yaml` file to the data config group, defining some data-specific variables.
    ```yaml
    # @package _global_
    data:
      module:
        _target_: datasets.RECvsT_1mm_cropped_160.RECvsT_1mm_cropped_160_DataModule
        name: RECvsT_1mm_cropped_160
        data_root_dir: ${data_dir}
        batch_size: 1
        train_transforms:
        _target_: augmentation.policies.batchgenerators.get_training_transforms
        patch_size: ${data.patch_size}
        rotation_for_DA: 0.523599
        mirror_axes: [0,1,2]
        do_dummy_2d_data_aug: False
        test_transforms: null
      cv:
        k:3

      num_classes: 2
      patch_size: [160, 160, 160]

    model:
      task: 'Classification'
      cifar_size: False
      input_channels: 2
      input_dim: 3
      input_shape: ${data.patch_size}
      optimizer: AdamW
      lr: 0.0001
      warmstart: 20
      weight_decay: 1e-2
      label_smoothing: 0.2

    trainer:
      logger:
        project: RECvsT_1mm_cropped_160
      accumulate_grad_batches: 48
      max_epochs: 400
      sync_batchnorm: True

    metrics:
      - 'f1'
      - 'balanced_acc'
      - 'ap'
      - 'auroc'
    ```
   The `data.module._target_` defines the path to your `DataModule`. Note that the first line of the file needs to be `# @package _global_` in order for Hydra to read the config properly.

# Tasks and losses

The model config takes two related fields:

- `task`: one of `'Classification'`, `'Regression'`, `'Ordinal_Regression'`.
- `loss_fn`: name of the loss to use. When `null`, a sensible default is selected per task (see table below).

| `task`              | `loss_fn: null` (default)       | Other valid `loss_fn` values                                                                                  |
|---------------------|----------------------------------|----------------------------------------------------------------------------------------------------------------|
| `Classification`    | `CrossEntropyLoss` (multiclass) / `BCEWithLogitsLoss` (multilabel) | `focal`, `topk10`                                                                                              |
| `Regression`        | `MSELoss`                        | *(none)*                                                                                                       |
| `Ordinal_Regression`| `coral_loss`                     | `focal`, `topk10`, `topk20`, `bce_focal`, `bce_topk10`, `bce_topk20`, `weighted_bce`, `bce_mae`                |

Example regression-task config block:

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