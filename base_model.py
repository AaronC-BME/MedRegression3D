import math
import warnings
from functools import partial

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from madgrad import MADGRAD
from timm.optim import RMSpropTF
from torch.optim.lr_scheduler import _LRScheduler
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
from torchmetrics.aggregation import CatMetric

from augmentation.mixup import mixup_criterion, mixup_data
from losses.coral_loss import coral_loss, label_to_levels
from metrics.balanced_accuracy import BalancedAccuracy
from metrics.conf_mat import ConfusionMatrix
from regularization.sam import SAM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_per_class_f1(metrics_res, prefix):
    """Replace `<prefix>F1_per_class` tensor entry with one scalar per class."""
    key = f"{prefix}F1_per_class"
    if key in metrics_res:
        for i, value in enumerate(metrics_res[key]):
            metrics_res[f"{prefix}F1_class_{i}"] = (
                value if not torch.isnan(value) else 0.0
            )
        del metrics_res[key]
    return metrics_res


def _build_classification_metrics(metrics_list, metric_task, num_classes):
    """Build the dict of torchmetrics for classification tasks."""
    common = dict(task=metric_task, num_classes=num_classes, num_labels=num_classes)
    out = {}
    if "acc" in metrics_list:
        out["Accuracy"] = Accuracy(**common)
    if "balanced_acc" in metrics_list:
        out["Balanced_Accuracy"] = BalancedAccuracy(task=metric_task, num_classes=num_classes)
    if "f1" in metrics_list:
        out["F1"] = F1Score(average="macro", **common)
    if "f1_per_class" in metrics_list:
        out["F1_per_class"] = F1Score(average=None, **common)
    if "pr" in metrics_list:
        out["Precision"] = Precision(average="macro", **common)
        out["Recall"] = Recall(average="macro", **common)
    if "top5acc" in metrics_list:
        out["Accuracy_top5"] = Accuracy(top_k=5, **common)
    if "auroc" in metrics_list:
        out["AUROC"] = AUROC(average="macro", **common)
    if "ap" in metrics_list:
        out["AP"] = AveragePrecision(**common)
    return out


def _build_regression_metrics(metrics_list):
    out = {}
    if "mse" in metrics_list:
        out["MSE"] = MeanSquaredError()
    if "mae" in metrics_list:
        out["MAE"] = MeanAbsoluteError()
    return out


VALID_TASKS = ("Classification", "Regression", "Ordinal_Regression")

# Loss-function names valid for each task. None => task default.
_VALID_LOSS_FNS = {
    "Classification": (None, "focal", "topk10"),
    "Regression": (None,),
    "Ordinal_Regression": (
        None, "focal", "topk10", "topk20",
        "bce_focal", "bce_topk10", "bce_topk20",
        "weighted_bce", "bce_mae",
    ),
}


def _build_criterion(task, loss_fn, label_smoothing, subtask):
    """Return the loss callable for a given (task, loss_fn) pair.

    If ``loss_fn`` is ``None`` the task-specific default is used.
    """
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task!r}. Expected one of {VALID_TASKS}.")

    valid = _VALID_LOSS_FNS[task]
    if loss_fn not in valid:
        raise ValueError(
            f"Unknown loss_fn={loss_fn!r} for task={task!r}. "
            f"Valid options are: {valid}."
        )

    # ---------- Classification ----------
    if task == "Classification":
        if loss_fn is None:
            if subtask == "multiclass":
                return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            if subtask == "multilabel":
                return nn.BCEWithLogitsLoss()
            raise ValueError(f"Unknown subtask: {subtask!r}")
        if loss_fn == "focal":
            from losses.cls_loss import FocalLoss
            print("Using Focal loss for Classification")
            return FocalLoss()
        if loss_fn == "topk10":
            from losses.cls_loss import TopKLoss
            print("Using Topk10 loss for Classification")
            return TopKLoss()

    # ---------- Regression ----------
    if task == "Regression":
        # Only None is valid (checked above).
        return nn.MSELoss()

    # ---------- Ordinal Regression ----------
    if task == "Ordinal_Regression":
        if loss_fn is None:
            return coral_loss
        if loss_fn == "focal":
            from losses.coral_loss import coral_focal_loss
            print("Using Coral Focal Loss (Gamma=3.0) for Ordinal Regression")
            return partial(coral_focal_loss, gamma=3.0)
        if loss_fn == "topk10":
            from losses.coral_loss import coral_topk_loss
            print("Using Coral TopK10 Loss for Ordinal Regression")
            return coral_topk_loss
        if loss_fn == "topk20":
            from losses.coral_loss import coral_topk_loss
            print("Using Coral TopK20 Loss for Ordinal Regression")
            return partial(coral_topk_loss, k=20)
        if loss_fn == "bce_focal":
            from losses.coral_loss import combined_bce_focal_loss
            print("Using Combined BCE and Focal Loss (Gamma=3.0) for Ordinal Regression")
            return partial(combined_bce_focal_loss, gamma=3.0)
        if loss_fn == "bce_topk10":
            from losses.coral_loss import combined_bce_topk_loss
            print("Using Combined BCE and TopK10 Loss for Ordinal Regression")
            return combined_bce_topk_loss
        if loss_fn == "bce_topk20":
            from losses.coral_loss import combined_bce_topk_loss
            print("Using Combined BCE and TopK20 Loss for Ordinal Regression")
            return partial(combined_bce_topk_loss, topk=20)
        if loss_fn == "weighted_bce":
            # Weights are populated in setup() and passed in training/val step.
            print("Using Weighted BCE Loss for Ordinal Regression")
            return coral_loss
        if loss_fn == "bce_mae":
            from losses.coral_loss import combined_coral_mae_loss
            print("Using BCE Loss and MAE (L1) Loss for Ordinal Regression")
            return combined_coral_mae_loss

    # Unreachable: guarded by the validity check above.
    raise ValueError(f"Unhandled (task, loss_fn) pair: ({task!r}, {loss_fn!r}).")


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class BaseModel(L.LightningModule):
    def __init__(
        self,
        task,
        loss_fn,
        metric_computation_mode,
        result_plot,
        metrics,
        num_classes,
        name,
        lr,
        weight_decay,
        optimizer,
        nesterov,
        sam,
        adaptive_sam,
        scheduler,
        T_max,
        warmstart,
        epochs,
        mixup,
        mixup_alpha,
        label_smoothing,
        stochastic_depth,
        resnet_dropout,
        squeeze_excitation,
        apply_shakedrop,
        undecay_norm,
        zero_init_residual,
        input_dim,
        input_channels,
        pretrained,
        *args,
        **kwargs,
    ):
        super().__init__()

        # --- Task / subtask / loss ------------------------------------------
        if task not in VALID_TASKS:
            raise ValueError(
                f"Unknown task: {task!r}. Expected one of {VALID_TASKS}."
            )
        self.task = task
        self.loss_fn = loss_fn  # None => task default
        self.subtask = kwargs["subtask"]

        # --- Metrics ----------------------------------------------------------
        self.metric_computation_mode = metric_computation_mode
        self.result_plot_setting = result_plot

        if self.task == "Classification":
            metric_task = self.subtask  # "multiclass" or "multilabel"
            metrics_dict = _build_classification_metrics(metrics, metric_task, num_classes)
        elif self.task in ("Ordinal_Regression", "Regression"):
            metrics_dict = _build_regression_metrics(metrics)
        else:
            metrics_dict = {}

        # Result-plotting bookkeeping
        if self.result_plot_setting in ("val", "all"):
            if self.task == "Classification":
                self.val_conf_mat = ConfusionMatrix(num_classes=num_classes)
            elif self.task in ("Ordinal_Regression", "Regression"):
                self.val_pred_list = []
                self.val_label_list = []
        if self.result_plot_setting == "all":
            if self.task == "Classification":
                self.train_conf_mat = ConfusionMatrix(num_classes=num_classes)
            elif self.task in ("Ordinal_Regression", "Regression"):
                self.train_pred_list = []
                self.train_label_list = []

        self.save_preds = bool(kwargs["save_preds"])
        if self.save_preds:
            self.val_preds = CatMetric(dist_sync_on_step=False)
            self.val_labels = CatMetric(dist_sync_on_step=False)
            self.val_indices = CatMetric(dist_sync_on_step=False)

        metric_collection = MetricCollection(metrics_dict)
        self.train_metrics = metric_collection.clone(prefix="Train/")
        self.val_metrics = metric_collection.clone(prefix="Val/")

        # --- Training args ----------------------------------------------------
        self.name = name
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.nesterov = nesterov
        self.sam = sam
        self.adaptive_sam = adaptive_sam
        self.scheduler = scheduler
        self.T_max = T_max
        self.warmstart = warmstart
        self.warmstart2 = kwargs["warmstart2"]
        self.epochs = epochs
        self.pretrained = pretrained

        # --- Regularization ---------------------------------------------------
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.stochastic_depth = stochastic_depth
        self.resnet_dropout = resnet_dropout
        self.se = squeeze_excitation
        self.apply_shakedrop = apply_shakedrop
        self.undecay_norm = undecay_norm
        self.zero_init_residual = zero_init_residual

        # --- Finetuning -------------------------------------------------------
        self.finetuning_method = kwargs["finetune_method"]

        # --- Data -------------------------------------------------------------
        self.input_dim = input_dim
        self.input_channels = input_channels
        self.num_classes = num_classes

        # SAM uses manual optimization
        if self.sam:
            self.automatic_optimization = False

        # --- Loss -------------------------------------------------------------
        self.criterion = _build_criterion(
            self.task, self.loss_fn, self.label_smoothing, self.subtask,
        )

    # -----------------------------------------------------------------------
    # Forward / setup
    # -----------------------------------------------------------------------

    def forward(self, x):
        pass

    def setup(self, stage=None):
        self.level_weights = None
        if self.loss_fn == "weighted_bce":
            print("Setting up level weights for Ordinal Regression Weighted BCE")
            self.level_weights = self.trainer.datamodule.level_weights.to(self.device)

    # -----------------------------------------------------------------------
    # Step helpers
    # -----------------------------------------------------------------------

    @property
    def _is_ordinal(self):
        return self.task == "Ordinal_Regression"

    def _forward_logits(self, x):
        """Run forward and return logits only (handles tuple-returning ordinal heads)."""
        out = self(x)
        if self._is_ordinal and isinstance(out, tuple):
            return out[0]
        return out

    def _compute_loss(self, y_hat, y):
        """Standard (non-mixup, non-SAM) loss given logits and labels."""
        if self._is_ordinal:
            levels = label_to_levels(y, self.num_classes)
            if self.loss_fn == "bce_mae":
                return self.criterion(
                    y_hat, levels, y, importance_weights=self.level_weights
                )
            if self.loss_fn == "weighted_bce":
                return self.criterion(
                    y_hat, levels, importance_weights=self.level_weights
                )
            return self.criterion(y_hat, levels)

        target = y.float() if self.subtask == "multilabel" else y.long()
        return self.criterion(y_hat, target)

    def _update_metrics(self, metrics_obj, y_hat, y):
        """Update an epochwise MetricCollection with predictions in the right form."""
        if self.task == "Classification":
            if self.subtask == "multilabel":
                metrics_obj.update(torch.sigmoid(y_hat.detach()), y)
            else:  # multiclass
                metrics_obj.update(F.softmax(y_hat.detach(), dim=-1), y)
        elif self._is_ordinal:
            pred_classes = (torch.sigmoid(y_hat.detach()) > 0.5).int().sum(dim=1)
            metrics_obj.update(pred_classes, y)
        else:
            metrics_obj.update(y_hat.view(-1).detach(), y.view(-1))

    def _log_metrics(self, metrics_res, prefix):
        _flatten_per_class_f1(metrics_res, prefix)
        self.log_dict(
            metrics_res,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Forward + (mixup) target prep
        if self.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(x, y, alpha=self.mixup_alpha)
            y_hat = self(inputs)
        else:
            y_hat = self._forward_logits(x)
            if (not self._is_ordinal) and self.num_classes == 1:
                y_hat = y_hat.view(-1)

        # Edge case: batch size 1 with squeezed batch dim
        if x.shape[0] == 1 and len(y_hat.shape) == 1:
            y_hat = y_hat.unsqueeze(0)

        # SAM uses manual optimization with two forward/backward passes
        if self.sam:
            opt = self.optimizers()

            if self.mixup:
                loss = mixup_criterion(self.criterion, y_hat, targets_a, targets_b, lam)
            else:
                loss = self.criterion(y_hat, y)
            self.manual_backward(loss)
            opt.first_step(zero_grad=True)

            if self.mixup:
                self.manual_backward(
                    mixup_criterion(
                        self.criterion, self(inputs), targets_a, targets_b, lam
                    )
                )
            else:
                second = self(x)
                if self.num_classes == 1:
                    second = second.view(-1)
                self.manual_backward(self.criterion(second, y))
            opt.second_step(zero_grad=True)
        else:
            if self.mixup:
                loss = mixup_criterion(self.criterion, y_hat, targets_a, targets_b, lam)
            else:
                loss = self._compute_loss(y_hat, y)

        self.log(
            "Train/loss", loss,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
        )

        if torch.isnan(y_hat).any():
            print("######################################### Model predicts NaNs!")

        # Metrics
        if self.metric_computation_mode == "stepwise":
            metrics_res = self.train_metrics(y_hat, y)
            self._log_metrics(metrics_res, "Train/")
        elif self.metric_computation_mode == "epochwise":
            self._update_metrics(self.train_metrics, y_hat, y)

        # Optional plot bookkeeping
        if hasattr(self, "train_conf_mat"):
            self.train_conf_mat.update(y_hat, y)
        if hasattr(self, "train_pred_list"):
            self.train_pred_list.extend(y_hat)
            self.train_label_list.extend(y)

        return loss

    # -----------------------------------------------------------------------
    # Validation step
    # -----------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self._forward_logits(x)
        if self.num_classes == 1:
            y_hat = y_hat.view(-1)

        val_loss = self._compute_loss(y_hat, y)

        self.log(
            "Val/loss", val_loss,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
        )

        if self.metric_computation_mode == "stepwise":
            metrics_res = self.val_metrics(y_hat, y)
            self._log_metrics(metrics_res, "Val/")
        elif self.metric_computation_mode == "epochwise":
            self._update_metrics(self.val_metrics, y_hat, y)

        if hasattr(self, "val_conf_mat"):
            self.val_conf_mat.update(y_hat, y)
        if hasattr(self, "val_preds"):
            actual_batch_size = x.size(0)
            start_idx = batch_idx * self.trainer.val_dataloaders.batch_size
            idx = torch.arange(
                start_idx, start_idx + actual_batch_size, device=self.device
            )
            self.val_preds.update(y_hat.detach())
            self.val_labels.update(y.detach())
            self.val_indices.update(idx)

    # -----------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.num_classes == 1:
            y_hat = y_hat.view(-1)
        return y, y_hat

    # -----------------------------------------------------------------------
    # Epoch ends
    # -----------------------------------------------------------------------

    def _log_val_predictions_table(self, preds_all, labels_all):
        """Log per-sample prediction table to W&B based on task type."""
        if self.task == "Classification":
            columns = (
                ["GT_" + str(i) for i in range(len(labels_all[0]))]
                if self.subtask == "multilabel"
                else ["GT"]
            ) + ["Pred_" + str(i) for i in range(len(preds_all[0]))]
            data = [
                (
                    (x.tolist() if self.subtask == "multilabel" else [x])
                    + (
                        F.softmax(y, dim=-1)
                        if self.subtask == "multiclass"
                        else torch.sigmoid(y)
                    ).tolist()
                )
                for x, y in zip(labels_all, preds_all)
            ]
            table = wandb.Table(data=data, columns=columns)
            wandb.log({"Val Predictions": table})

        elif self.task == "Regression":
            data = [[x.item(), y.item()] for x, y in zip(labels_all, preds_all)]
            table = wandb.Table(data=data, columns=["GT", "Pred"])
            wandb.log({"Val Predictions": table})

        elif self._is_ordinal:
            binary_preds = (torch.sigmoid(preds_all) > 0.5).int()
            expected_ages = binary_preds.sum(dim=1)
            data = [[x.item(), y.item()] for x, y in zip(labels_all, expected_ages)]
            table = wandb.Table(data=data, columns=["GT", "Pred"])
            wandb.log({"Val Predictions": table})

        else:
            raise NotImplementedError

    def _log_val_scatterplot(self, preds_all, labels_all):
        """Scatterplot of GT vs prediction for Regression / Ordinal Regression."""
        if self.task == "Regression":
            data = [[x, y] for (x, y) in zip(labels_all, preds_all)]
        elif self._is_ordinal:
            binary_preds = (torch.sigmoid(preds_all) > 0.5).int()
            expected_ages = binary_preds.sum(dim=1)
            data = [[x, y] for (x, y) in zip(labels_all, expected_ages)]
        else:
            return

        table = wandb.Table(data=data, columns=["Ground Truth", "Prediction"])
        wandb.log({
            "Val Scatterplot": wandb.plot.scatter(
                table, "Ground Truth", "Prediction", "Validation Scatterplot",
            )
        })

    def on_validation_epoch_end(self) -> None:
        if self.metric_computation_mode == "epochwise":
            metrics_res = self.val_metrics.compute()
            self._log_metrics(metrics_res, "Val/")
            self.val_metrics.reset()

        if hasattr(self, "val_conf_mat"):
            self.val_conf_mat.save_state(self, "val")
            self.val_conf_mat.reset()

        if hasattr(self, "val_preds"):
            preds_all = self.val_preds.compute()
            labels_all = self.val_labels.compute()
            indices = self.val_indices.compute()

            if self.trainer.is_global_zero:
                # Sort by original index to preserve dataset order
                sorted_idx = torch.argsort(indices)
                preds_all = preds_all[sorted_idx]
                labels_all = labels_all[sorted_idx]

                self._log_val_scatterplot(preds_all, labels_all)

                if self.save_preds:
                    self._log_val_predictions_table(preds_all, labels_all)

            self.val_preds.reset()
            self.val_labels.reset()
            self.val_indices.reset()

    def on_train_epoch_end(self) -> None:
        if self.metric_computation_mode == "epochwise":
            metrics_res = self.train_metrics.compute()
            self._log_metrics(metrics_res, "Train/")
            self.train_metrics.reset()

        if hasattr(self, "train_conf_mat"):
            self.train_conf_mat.save_state(self, "train")
            self.train_conf_mat.reset()

        if hasattr(self, "train_pred_list"):
            data = [
                [x, y] for (x, y) in zip(self.train_label_list, self.train_pred_list)
            ]
            table = wandb.Table(data=data, columns=["Ground Truth", "Prediction"])
            wandb.log({
                "Train Scatterplot": wandb.plot.scatter(
                    table, "Ground Truth", "Prediction", "Train Scatterplot",
                )
            })
            self.train_pred_list = []
            self.train_label_list = []

    # -----------------------------------------------------------------------
    # Init from scratch (when not pretrained)
    # -----------------------------------------------------------------------

    def on_train_start(self):
        if self.pretrained:
            return

        print("Initializing weights")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # -----------------------------------------------------------------------
    # Optimizer / scheduler
    # -----------------------------------------------------------------------

    def _split_params_for_sawtooth(self):
        """Group named params into encoder vs head (cls_head or reg_head)."""
        encoder_params, cls_head_params, reg_head_params = [], [], []
        for name, param in self.named_parameters():
            if "encoder" in name:
                encoder_params.append(param)
            elif "cls_head" in name:
                cls_head_params.append(param)
            elif "reg_head" in name:
                reg_head_params.append(param)

        if cls_head_params:
            return encoder_params, cls_head_params, "cls_head"
        return encoder_params, reg_head_params, "reg_head"

    def _build_param_groups(self):
        """Param groups for optimizers. Returns either an iterable of params or
        a list of param-group dicts (when sawtooth fine-tuning is on)."""
        # Optionally split norm/bias params off from weight decay.
        if self.undecay_norm:
            model_params, norm_params = [], []
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if "norm" in name or "bias" in name or "bn" in name:
                    norm_params.append(p)
                else:
                    model_params.append(p)
            base_params = [
                {"params": model_params},
                {"params": norm_params, "weight_decay": 0},
            ]
        else:
            base_params = self.parameters()

        if self.finetuning_method != "full_sawtooth":
            return base_params

        # Sawtooth: separate head from encoder so the scheduler can warm them
        # up independently. We always rebuild the groups directly from named
        # params (the undecay_norm split is not combined with sawtooth).
        encoder_params, head_params, head_name = self._split_params_for_sawtooth()

        common_head = {
            "params": head_params,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "name": head_name,
        }
        common_enc = {
            "params": encoder_params,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "name": "encoder",
        }
        # Some optimizers want momentum in the group dict
        if self.optimizer in ("SGD", "Madgrad"):
            common_head["momentum"] = 0.9
            common_enc["momentum"] = 0.9
            if self.optimizer == "SGD":
                common_head["nesterov"] = self.nesterov
                common_enc["nesterov"] = self.nesterov

        return [common_head, common_enc]

    def _build_optimizer(self, params):
        """Construct optimizer (non-SAM path)."""
        if self.optimizer == "SGD":
            return torch.optim.SGD(
                params, lr=self.lr, momentum=0.9,
                weight_decay=self.weight_decay, nesterov=self.nesterov,
            )
        if self.optimizer == "Adam":
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer == "AdamW":
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer == "Rmsprop":
            return RMSpropTF(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer == "Madgrad":
            return MADGRAD(
                params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def _build_sam_optimizer(self, params):
        # ASAM paper suggests 10x larger rho for adaptive SAM than normal SAM
        rho = 0.5 if self.adaptive_sam else 0.05
        common = dict(
            adaptive=self.adaptive_sam, lr=self.lr,
            weight_decay=self.weight_decay, rho=rho,
        )

        if self.optimizer == "SGD":
            return SAM(
                params, torch.optim.SGD, momentum=0.9, nesterov=self.nesterov,
                **common,
            )
        if self.optimizer == "Madgrad":
            return SAM(params, MADGRAD, momentum=0.9, **common)
        if self.optimizer == "Adam":
            return SAM(params, torch.optim.Adam, **common)
        if self.optimizer == "AdamW":
            return SAM(params, torch.optim.AdamW, **common)
        if self.optimizer == "Rmsprop":
            return SAM(params, RMSpropTF, **common)
        raise ValueError(f"Unknown optimizer for SAM: {self.optimizer}")

    def _build_scheduler(self, optimizer):
        if not self.scheduler:
            return None

        if self.scheduler == "CosineAnneal":
            if self.warmstart == 0:
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.T_max,
                )
            if self.finetuning_method == "full_sawtooth":
                print(
                    f"[INFO] Using CosineAnnealingLR_DoubleWarmstart: "
                    f"warmstart1={self.warmstart}, warmstart2={self.warmstart2}, "
                    f"T_max={self.T_max}"
                )
                return CosineAnnealingLR_DoubleWarmstart(
                    optimizer, T_max=self.T_max,
                    warmstart1=self.warmstart, warmstart2=self.warmstart2,
                )
            print(
                f"[INFO] Using CosineAnnealingLR_Warmstart: "
                f"warmstart1={self.warmstart}, T_max={self.T_max}"
            )
            return CosineAnnealingLR_Warmstart(
                optimizer, T_max=self.T_max, warmstart=self.warmstart,
            )

        if self.scheduler == "Step":
            # Decay every quarter of total epochs
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.epochs // 4, gamma=0.1,
            )
        if self.scheduler == "MultiStep":
            # Decay at half and three-quarters of training
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer, [self.epochs // 2, self.epochs * 3 // 4],
            )

        raise ValueError(f"Unknown scheduler: {self.scheduler}")

    def configure_optimizers(self):
        # Note from original: leave bias and BN params undecayed
        # (https://arxiv.org/pdf/1812.01187.pdf, Bag of Tricks).
        params = self._build_param_groups()

        if self.sam:
            optimizer = self._build_sam_optimizer(params)
        else:
            optimizer = self._build_optimizer(params)

        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return [optimizer]
        return [optimizer], [scheduler]


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------

class CosineAnnealingLR_Warmstart(_LRScheduler):
    """
    CosineAnnealingLR with a linear warmup phase. See
    https://arxiv.org/pdf/1706.02677.pdf.
    """

    def __init__(
        self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False, warmstart=0,
    ):
        self.T_max = T_max - warmstart  # warmup epochs not part of cosine period
        self.eta_min = eta_min
        self.warmstart = warmstart
        self.T = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        # Warmup
        if self.last_epoch < self.warmstart:
            addrates = [lr / (self.warmstart + 1) for lr in self.base_lrs]
            return [
                addrates[i] * (self.last_epoch + 1)
                for i, _ in enumerate(self.optimizer.param_groups)
            ]

        # Cosine annealing
        if self.T == 0:
            self.T += 1
            return self.base_lrs

        if (self.T - 1 - self.T_max) % (2 * self.T_max) == 0:
            updated_lr = [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / self.T_max))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
            self.T += 1
            return updated_lr

        updated_lr = [
            (1 + math.cos(math.pi * self.T / self.T_max))
            / (1 + math.cos(math.pi * (self.T - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
        self.T += 1
        return updated_lr


class CosineAnnealingLR_DoubleWarmstart(_LRScheduler):
    """
    Two consecutive linear warmup phases followed by cosine annealing.

    - Phase 1 (warmstart1 epochs): only the head warms up; encoder LR stays at 0.
    - Phase 2 (warmstart2 epochs): both head and encoder warm up.
    - Cosine annealing decays both groups.
    """

    def __init__(
        self,
        optimizer,
        T_max,
        eta_min=0,
        last_epoch=-1,
        verbose=False,
        warmstart1=0,
        warmstart2=0,
    ):
        self.warmstart1 = warmstart1
        self.warmstart2 = warmstart2
        self.eta_min = eta_min
        self.T_max = T_max - (warmstart1 + warmstart2)  # cosine decay period
        self.T = 0  # internal counter (unused, kept for parity)

        # Locate parameter groups by name; either "cls_head" or "reg_head" works.
        self.head_group = None
        self.encoder_group = None
        for param_group in optimizer.param_groups:
            name = param_group.get("name")
            if name in ("cls_head", "reg_head"):
                self.head_group = param_group
            elif name == "encoder":
                self.encoder_group = param_group

        if self.head_group is None:
            raise ValueError(
                "Optimizer must have a parameter group named 'cls_head' or 'reg_head'."
            )
        if self.encoder_group is None:
            raise ValueError("Optimizer must have a parameter group named 'encoder'.")

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        warmup_total = self.warmstart1 + self.warmstart2

        # Phase 1: warm up the head only.
        if self.last_epoch < self.warmstart1:
            # Bug fix: guard against warmstart1 == 0 (would div-by-zero, though
            # the branch wouldn't be taken with warmstart1 == 0 anyway).
            denom = max(self.warmstart1, 1)
            warmup_factor = (self.last_epoch + 1) / denom
            return [
                group["initial_lr"] * warmup_factor if group is self.head_group else 0
                for group in self.optimizer.param_groups
            ]

        # Phase 2: warm up both head and encoder.
        if self.last_epoch < warmup_total:
            # Bug fix: guard against warmstart2 == 0.
            denom = max(self.warmstart2, 1)
            warmup_factor = (self.last_epoch - self.warmstart1 + 1) / denom
            return [
                group["initial_lr"] * warmup_factor
                for group in self.optimizer.param_groups
            ]

        # Cosine annealing for both groups.
        epoch_cosine = self.last_epoch - warmup_total
        return [
            self.eta_min
            + (group["initial_lr"] - self.eta_min)
            * 0.5
            * (1 + math.cos(math.pi * epoch_cosine / self.T_max))
            for group in self.optimizer.param_groups
        ]


# ---------------------------------------------------------------------------
# Generic wrapper
# ---------------------------------------------------------------------------

class ModelConstructor(BaseModel):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def forward(self, x):
        return self.model(x)