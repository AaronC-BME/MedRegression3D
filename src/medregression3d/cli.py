import os
from contextlib import suppress
from pathlib import Path

import hydra
import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from medregression3d.utils.parsing import make_omegaconf_resolvers


def _prepare_cfg(cfg):
    """Top-level cfg mutations done once before the CV loop."""
    if cfg.seed:
        seed_everything(cfg.seed)
        cfg.trainer.benchmark = False
        cfg.trainer.deterministic = True

    # Hydra auto-creates main.log; remove it (W&B already captures everything).
    with suppress(FileNotFoundError):
        Path("./main.log").unlink()

    log_path = Path(cfg.trainer.logger.save_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    if cfg.trainer.devices > 1 and cfg.trainer.accelerator == "gpu":
        cfg.trainer.sync_batchnorm = True

    cfg.trainer.callbacks = [c for c in cfg.trainer.callbacks.values() if c]
    if not cfg.trainer["enable_checkpointing"]:
        cfg.trainer.callbacks = [
            c for c in cfg.trainer.callbacks
            if c["_target_"] != "lightning.pytorch.callbacks.ModelCheckpoint"
        ]


def _set_checkpoint_dir(cfg, base_name):
    """Point ModelCheckpoint at <output_dir>/<dataset>/<base_name>/folds/<fold>."""
    if not cfg.trainer["enable_checkpointing"]:
        return
    for cb in cfg.trainer.callbacks:
        if cb["_target_"] == "lightning.pytorch.callbacks.ModelCheckpoint":
            cb["dirpath"] = os.path.join(
                str(cfg.output_dir),
                str(cfg.data.module.name),
                str(base_name),
                "folds",
                str(cfg.data.module.fold),
            )


def _log_hyperparams(trainer, cfg):
    """Strip non-loggable fields and forward the rest to the logger."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    cfg_dict["model"].pop("_target_")
    cfg_dict["model"]["model"] = cfg_dict["model"].pop("name")
    trainer.logger.log_hyperparams(cfg_dict["model"])

    data_module = cfg_dict["data"]["module"]
    data_module.pop("_target_")
    for key in ("train_transforms", "test_transforms"):
        if data_module.get(key) is not None:
            data_module[key] = ".".join(
                data_module[key]["_target_"].split(".")[-2:]
            )
    data_module.pop("name")
    trainer.logger.log_hyperparams(data_module)

    trainer_cfg = cfg_dict["trainer"]
    for key in (
        "_target_", "callbacks", "enable_checkpointing",
        "enable_progress_bar", "logger", "num_sanity_val_steps",
    ):
        trainer_cfg.pop(key, None)
    trainer.logger.log_hyperparams(trainer_cfg)


CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train")
def main(cfg):
    _prepare_cfg(cfg)
    print(OmegaConf.to_yaml(cfg))

    # `trainer.logger.name` is the source of truth for both the W&B run name
    # and the on-disk run folder. Capture the base before the fold loop so
    # all folds share one parent dir even though their W&B names differ.
    base_name = cfg.trainer.logger.name

    for k in range(cfg.data.cv.k):
        if cfg.data.cv.k > 1:
            cfg.data.module.fold = k
        elif cfg.data.module.fold is None:
            cfg.data.module.fold = "0"

        if cfg.data.cv.k > 1:
            cfg.trainer.logger.name = f"{base_name}_fold{cfg.data.module.fold}"

        _set_checkpoint_dir(cfg, base_name)

        trainer = instantiate(cfg.trainer)
        model = instantiate(cfg.model)

        if cfg.model.compile:
            model = torch.compile(model, mode="default")
        dataset = instantiate(cfg.data).module

        _log_hyperparams(trainer, cfg)

        if cfg.val_only:
            trainer.validate(model, dataset)
        else:
            trainer.fit(model, dataset, ckpt_path=cfg.get("ckpt_path", None))

        wandb.finish()


if __name__ == "__main__":
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    make_omegaconf_resolvers()
    main()
