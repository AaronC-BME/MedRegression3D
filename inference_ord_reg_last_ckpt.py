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
    ckp_paths = [x for x in ckp_paths if 'last.ckpt' in str(x)] # THis is where to select the checkpoint
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

        # Unpack probas from prediction tuples
        probas_list = [probas for (_, probas) in y_hat]  # assuming y_hat is a list of (logits, probas)
        probas_tensor = torch.cat(probas_list, dim=0)  # shape: [N, num_classes - 1]

        # Decode ordinal predictions (CORAL-style)
        # Each row in probas_tensor has sigmoid outputs for thresholds
        preds = torch.sum(probas_tensor > 0.5, dim=1)  # shape: [N]

        # Convert targets to tensor
        targets_tensor = torch.cat(y) if isinstance(y, (list, tuple)) else y

        # Compute errors
        abs_error = torch.abs(preds - targets_tensor).cpu().numpy()
        error = (preds - targets_tensor).cpu().numpy()

        # Compute MAE and ME
        mae_metric = MeanAbsoluteError()
        mae = mae_metric(preds, targets_tensor)
        mean_error = error.mean()
    
        print(f"Test MAE: {mae.item():.4f}")
        print(f"Test ME (signed): {mean_error:.4f}")

        # Extract patient IDs from dataset
        # Assumes dataset returns a dictionary with a "name" or "id" field per sample
        # You may need to adjust this depending on your actual dataset structure

        patient_ids = dataset.test_dataset.img_files

        # Match number of patient IDs to predictions (in case of batching)
        if len(patient_ids) != len(preds):
            patient_ids = patient_ids[:len(preds)]


        # Create a DataFrame
        df = pd.DataFrame({
            "PatientID": patient_ids,
            "GroundTruth": targets_tensor.cpu().numpy(),
            "Prediction": preds.cpu().numpy(),
            "MAE": abs_error,
            "Error": error,
        })

        # Save to Excel
        out_path = os.path.join(cfg.exp_dir, "predictions.xlsx")
        df.to_excel(out_path, index=False)
        print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    make_omegaconf_resolvers()
    inference()
