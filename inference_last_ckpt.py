import os
from pathlib import Path
from uuid import uuid4

import hydra
import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf
import pandas as pd

from parsing_utils import make_omegaconf_resolvers
import glob

from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    Precision,
    Recall,
)


@hydra.main(version_base=None, config_path="./cli_configs", config_name="infer")
def inference(cfg):
    # delete automatically created hydra logger
    try:
        Path(
            "./main.log"
        ).unlink()
    except:
        pass

    # check if a fold was given, if yes scan only the fold dir for the checkpoint path
    if cfg.fold:
        ckp_paths = list(Path(Path(cfg.ckpt_dir) / str(cfg.fold)).glob("*.ckpt"))
    else:
        ckp_paths = list(Path(cfg.ckpt_dir).glob("*/*.ckpt"))
    
    logits = []
    ckp_paths = [x for x in ckp_paths if 'epoch18-Val_loss=0.50-Val_mae=5.51.ckpt' in str(x)]
    print(f'checkpoint paths: {ckp_paths}')
    for ckp_path in ckp_paths:

        # load the config that was used during training
        #used_training_cfg = OmegaConf.load(os.path.join(cfg.exp_dir, "config.yaml"))
        used_training_config_path = glob.glob(os.path.join(cfg.exp_dir, "*/config.yaml"))[0]
        print(f"Using config: {used_training_config_path}")
        used_training_cfg = OmegaConf.load(used_training_config_path)
        used_training_cfg.trainer.pop("logger")
        used_training_cfg.trainer.pop("callbacks")
        used_training_cfg.model.metrics = cfg.metrics  # overwrite metrics

        # instantiate the model using this config
        model = instantiate(used_training_cfg.model)
        # load the weights
        model.load_state_dict(torch.load(ckp_path)["state_dict"])
        model.eval()

        # instantiate the dataset from the config if not some other dataset is specified in the infer.yaml
        dataset = instantiate(used_training_cfg.data).module
        # instantiate the trainer and also pass the new metrics
        trainer = instantiate(used_training_cfg.trainer)
       
        # run trainer.predict
        y, y_hat = zip(*trainer.predict(model, dataset))
        y = torch.cat(y)
        y_hat = torch.cat(y_hat)
        mae_metric = MeanAbsoluteError()
        mae = mae_metric(y_hat, y)
        print(f"Test MAE: {mae.item():.4f}")

        # Patient IDs
        try:
            patient_ids = dataset.test_dataset.img_files  # for datamodule-style
        except AttributeError:
            patient_ids = dataset.img_files  # direct dataset

        if len(patient_ids) != len(y_hat):
            patient_ids = patient_ids[:len(y_hat)]

        abs_error = torch.abs(y_hat - y).cpu().numpy()
        raw_error = (y_hat - y).cpu().numpy()  # <-- add this
        mean_error = raw_error.mean()          # <-- compute mean error

        df = pd.DataFrame({
            "PatientID": patient_ids,
            "GroundTruth": y.cpu().numpy(),
            "Prediction": y_hat.cpu().numpy(),
            "Error": raw_error,                # <-- new column for raw error
            "MAE": abs_error,
        })

        out_path = os.path.join(cfg.exp_dir, "predictions.xlsx")
        df.to_excel(out_path, index=False)
        print(f"✅ Saved predictions to {out_path}")

        print(f"Test MAE: {mae.item():.4f}")
        print(f"Test Mean Error: {mean_error:.4f}")  # <-- print mean error

if __name__ == "__main__":
    make_omegaconf_resolvers()
    inference()
