import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torchmetrics import Metric
from torchmetrics.functional.classification import stat_scores
from torchmetrics.utilities.data import _bincount


class BalancedAccuracy(Metric):
    def __init__(
            self,
            num_classes: int,
            task: str = "multiclass",
            threshold: float = 0.5,
            dist_sync_on_step=False,
    ):
        assert task in {
            "multiclass",
            "multilabel",
        }, "Only 'multiclass' and 'multilabel' tasks are supported."
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes
        self.task = task
        self.threshold = threshold

        self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        target = target.to(torch.long)
        if self.task == "multilabel":

            # Auto-detect logits vs probs
            if preds.max() > 1.0 or preds.min() < 0.0:
                preds = torch.sigmoid(preds)

            preds = (preds >= self.threshold).long()

            stats = stat_scores(
                preds=preds,
                target=target,
                task="multilabel",
                num_labels=self.num_classes,
                average=None,
            )
        elif self.task == "multiclass":
            if preds.ndim == 2 and preds.size(1) == self.num_classes:
                preds = torch.argmax(preds, dim=1)

            stats = stat_scores(
                preds=preds,
                target=target,
                task="multiclass",
                num_classes=self.num_classes,
                average=None,
            )

        if stats.ndim == 1:
            stats = stats.unsqueeze(0)  # make it 2D to unbind along dim=1

        tp, fp, tn, fn, _ = stats.unbind(dim=1)
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self):
        recall = self.tp / (self.tp + self.fn + 1e-8)
        specificity = self.tn / (self.tn + self.fp + 1e-8)
        balanced_acc = (recall + specificity) / 2
        return balanced_acc.mean()


class ConfusionMatrix(Metric):
    full_state_update = False

    def __init__(self, num_classes: int, labels: list = None) -> None:
        super().__init__(dist_sync_on_step=False)
        self.num_classes = num_classes
        if labels is not None:
            self.labels = labels
        else:
            self.labels = np.arange(self.num_classes).astype(str)
        self.add_state(
            "mat",
            default=torch.zeros((num_classes, num_classes), dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def compute(self):
        pass

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        pred = pred.argmax(1).flatten()
        gt = gt.flatten()
        n = self.num_classes

        with torch.no_grad():
            k = (gt >= 0) & (gt < n)
            inds = n * gt[k].to(torch.int64) + pred[k]
            confmat = _bincount(inds, minlength=n**2).reshape(n, n)

        self.mat += confmat

    def save_state(self, trainer: pl.Trainer, split: str) -> None:
        def mat_to_figure(mat: np.ndarray, name: str = "Confusion matrix", norm_colorbar=False) -> Figure:
            figure = plt.figure(figsize=(8, 8))
            plt.imshow(mat, interpolation="nearest", cmap=plt.cm.viridis)
            plt.title(name)
            if norm_colorbar:
                plt.clim(0, 1)
            plt.colorbar()
            if hasattr(self, "class_names"):
                labels = self.class_names
            else:
                labels = np.arange(self.num_classes)

            tick_marks = np.arange(len(labels))

            plt.xticks(tick_marks, labels, rotation=0)
            plt.yticks(tick_marks, labels)
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.close(figure)

            return figure

        confmat = self.mat.detach().cpu().numpy()
        figure = mat_to_figure(confmat, "Confusion Matrix")

        confmat_norm = np.around(confmat.astype("float") / confmat.sum(axis=1)[:, np.newaxis], decimals=2)
        figure_norm = mat_to_figure(confmat_norm, "Confusion Matrix (normalized)", norm_colorbar=True)

        for logger in trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]:
            if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
                logger.experiment.add_figure(
                    "{}_ConfusionMatrix_normalized/ConfusionMatrix".format(split),
                    figure_norm,
                    trainer.current_epoch,
                )
                logger.experiment.add_figure(
                    "{}_ConfusionMatrix_absolute/ConfusionMatrix".format(split),
                    figure,
                    trainer.current_epoch,
                )
            elif isinstance(logger, pl.loggers.mlflow.MLFlowLogger):
                logger.experiment.log_figure(
                    run_id=logger.run_id,
                    figure=figure_norm,
                    artifact_file="{}_ConfusionMatrix_normalized.png".format(split),
                )
                logger.experiment.log_figure(
                    run_id=logger.run_id,
                    figure=figure,
                    artifact_file="{}_ConfusionMatrix_absolute.png".format(split),
                )
            elif isinstance(logger, pl.loggers.wandb.WandbLogger):
                logger.log_image(
                    key="{}_ConfusionMatrix_normalized".format(split),
                    images=[figure_norm],
                    step=trainer.current_epoch,
                )
                logger.log_image(
                    key="{}_ConfusionMatrix_absolute".format(split),
                    images=[figure],
                    step=trainer.current_epoch,
                )
