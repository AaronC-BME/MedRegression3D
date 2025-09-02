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
from metrics.balanced_accuracy import BalancedAccuracy

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


@hydra.main(version_base=None, config_path="./cli_configs", config_name="infer_cls")
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
    ckp_paths = [x for x in ckp_paths if 'epoch151-Val_loss=0.73-Val_bal_acc=0.75.ckpt' in str(x)]
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

        y_pred = torch.argmax(y_hat, dim=1)

        # Define classification metrics
        acc_metric = Accuracy(task='multiclass', num_classes=used_training_cfg.model.num_classes)
        f1_metric = F1Score(task='multiclass', num_classes=used_training_cfg.model.num_classes, average='macro')
        auroc_metric = AUROC(task='multiclass', num_classes=used_training_cfg.model.num_classes)
        ap_metric = AveragePrecision(task='multiclass', num_classes=used_training_cfg.model.num_classes)
        balanced_acc_metric = BalancedAccuracy(num_classes=used_training_cfg.model.num_classes)

        balanced_acc = balanced_acc_metric(y_pred, y)
        acc = acc_metric(y_pred, y)
        f1 = f1_metric(y_pred, y)
        auroc = auroc_metric(y_hat, y)  # use logits for AUROC
        ap = ap_metric(y_hat, y)        # use logits for Average Precision

        print(f"Test Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Test AUROC: {auroc:.4f}")
        print(f"Test Average Precision: {ap:.4f}")

        # Patient IDs
        try:
            patient_ids = dataset.test_dataset.img_files  # for datamodule-style
        except AttributeError:
            patient_ids = dataset.img_files  # direct dataset

        if len(patient_ids) != len(y_hat):
            patient_ids = patient_ids[:len(y_hat)]

        df = pd.DataFrame({
            "PatientID": patient_ids,
            "GroundTruth": y.cpu().numpy(),
            "Prediction": y_pred.cpu().numpy(),
            "Logits": [log.tolist() for log in y_hat.cpu()],  # optional: for debugging
        })

        out_path = os.path.join(cfg.exp_dir, "classification_predictions.xlsx")
        df.to_excel(out_path, index=False)
        print(f"✅ Saved predictions to {out_path}")

        metrics_df = pd.DataFrame({
            "Metric": ["Balanced Accuracy", "Accuracy", "F1 Score", "AUROC", "Average Precision"],
            "Value": [balanced_acc.item(), acc.item(), f1.item(), auroc.item(), ap.item()]
        })

        metrics_out_path = os.path.join(cfg.exp_dir, "classification_metrics.xlsx")
        metrics_df.to_excel(metrics_out_path, index=False)
        print(f"✅ Saved metrics to {metrics_out_path}")

if __name__ == "__main__":
    make_omegaconf_resolvers()
    inference()
