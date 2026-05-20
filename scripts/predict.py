import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from medregression3d.data.datamodules import AgeReg_Data
from medregression3d.utils.parsing import make_omegaconf_resolvers


CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs")
)


def _select_best_ckpt(ckp_paths, prefer_best=True):
    """
    From a list of checkpoint paths, pick one.
    If prefer_best is True, return the checkpoint with the lowest 'Val_mae'
    parsed from its filename. Falls back to last.ckpt if no parseable filename
    is found. If prefer_best is False, return last.ckpt.
    """
    ckp_paths = [Path(p) for p in ckp_paths]
    last = [p for p in ckp_paths if p.name == "last.ckpt"]
    not_last = [p for p in ckp_paths if p.name != "last.ckpt"]

    if not prefer_best:
        return last[0] if last else (ckp_paths[0] if ckp_paths else None)

    def _parse_mae(p):
        try:
            tag = str(p).split("Val_mae=")[1]
            return float(tag.split(".ckpt")[0])
        except (IndexError, ValueError):
            return float("inf")

    if not_last:
        not_last.sort(key=_parse_mae)
        if _parse_mae(not_last[0]) != float("inf"):
            return not_last[0]

    return last[0] if last else (ckp_paths[0] if ckp_paths else None)


def _compute_metrics(preds, targets):
    """Return a dict of regression metrics given two 1-D float tensors."""
    preds = preds.float()
    targets = targets.float()
    err = preds - targets
    mae = err.abs().mean().item()
    rmse = torch.sqrt((err ** 2).mean()).item()
    me = err.mean().item()
    n = preds.numel()
    if n > 1 and preds.std().item() > 0 and targets.std().item() > 0:
        pearson = torch.corrcoef(torch.stack([preds, targets]))[0, 1].item()
    else:
        pearson = float("nan")
    return {"N": n, "MAE": mae, "RMSE": rmse, "MeanError": me, "Pearson_r": pearson}


def _bin_errors_by_age(targets, preds, bin_width=10, max_age=100):
    """
    Group prediction errors into age bins of `bin_width` years based on the
    ground-truth age. Returns a DataFrame with columns:
        Bin, N, MAE, RMSE, MeanError, StdError
    """
    targets_np = targets.detach().cpu().numpy().astype(float)
    preds_np = preds.detach().cpu().numpy().astype(float)
    err_np = preds_np - targets_np
    abs_err_np = np.abs(err_np)

    edges = np.arange(0, max_age + bin_width, bin_width)
    labels = [f"{int(edges[i])}-{int(edges[i + 1]) - 1}" for i in range(len(edges) - 1)]

    rows = []
    for i, lab in enumerate(labels):
        lo, hi = edges[i], edges[i + 1]
        mask = (targets_np >= lo) & (targets_np < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"Bin": lab, "N": 0, "MAE": np.nan, "RMSE": np.nan,
                         "MeanError": np.nan, "StdError": np.nan})
        else:
            rows.append({
                "Bin": lab,
                "N": n,
                "MAE": float(abs_err_np[mask].mean()),
                "RMSE": float(np.sqrt((err_np[mask] ** 2).mean())),
                "MeanError": float(err_np[mask].mean()),
                "StdError": float(err_np[mask].std(ddof=1)) if n > 1 else 0.0,
            })

    overflow_mask = targets_np >= max_age
    n_over = int(overflow_mask.sum())
    if n_over > 0:
        rows.append({
            "Bin": f">={int(max_age)}",
            "N": n_over,
            "MAE": float(abs_err_np[overflow_mask].mean()),
            "RMSE": float(np.sqrt((err_np[overflow_mask] ** 2).mean())),
            "MeanError": float(err_np[overflow_mask].mean()),
            "StdError": float(err_np[overflow_mask].std(ddof=1)) if n_over > 1 else 0.0,
        })

    return pd.DataFrame(rows)


def _plot_error_bars(bin_df, title, out_path, metric="MAE"):
    """Bar chart of `metric` per age bin. Empty bins drawn as gaps."""
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(bin_df)), 5))

    bins = bin_df["Bin"].tolist()
    values = bin_df[metric].to_numpy(dtype=float)
    counts = bin_df["N"].to_numpy(dtype=int)

    plot_values = np.where(np.isnan(values), 0.0, values)

    bars = ax.bar(bins, plot_values, color="steelblue", edgecolor="black")

    for bar, n, v in zip(bars, counts, values):
        height = bar.get_height()
        if n == 0 or np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, 0,
                    "n=0", ha="center", va="bottom", fontsize=8, color="gray")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f"n={n}\n{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Age bin (years)")
    ax.set_ylabel(metric + (" (years)" if metric in ("MAE", "RMSE", "MeanError") else ""))
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if metric == "MeanError":
        ax.axhline(0, color="black", linewidth=0.8)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved plot: {out_path}")


def _save_bin_report(targets, preds, tag, pred_dir, bin_width=10, max_age=100):
    """
    Compute age-binned errors, save them as CSV, and produce MAE / MeanError
    bar charts. `tag` is used in filenames.
    """
    bin_df = _bin_errors_by_age(targets, preds, bin_width=bin_width, max_age=max_age)

    csv_path = os.path.join(pred_dir, f"error_by_age_bin_{tag}.csv")
    bin_df.to_csv(csv_path, index=False)
    print(f"[{tag}] saved per-bin error stats to {csv_path}")

    _plot_error_bars(
        bin_df,
        title=f"MAE per age bin ({tag})",
        out_path=os.path.join(pred_dir, f"error_by_age_bin_{tag}_MAE.png"),
        metric="MAE",
    )
    _plot_error_bars(
        bin_df,
        title=f"Mean signed error per age bin ({tag})",
        out_path=os.path.join(pred_dir, f"error_by_age_bin_{tag}_MeanError.png"),
        metric="MeanError",
    )

    return bin_df


def _build_predict_manifest(data_dir, label_column, manifest_path):
    """List .b2nd files under data_dir and write a synthetic CSV that AgeReg_Data accepts.

    All rows are labeled split='test', fold=0, with a dummy label value of 0. Ground-truth
    labels aren't available in predict-mode, so metrics are skipped downstream.
    """
    images = sorted(p.stem for p in Path(data_dir).glob("*.b2nd"))
    if not images:
        raise FileNotFoundError(f"No .b2nd files found under {data_dir}")
    df = pd.DataFrame({
        "image_name": images,
        "split": "test",
        "fold": 0,
        label_column: 0.0,
    })
    df.to_csv(manifest_path, index=False)
    return images


def _run_data_dir_predict(model, training_cfg, trainer, data_dir, pred_dir, fold_id):
    """Predict on every .b2nd in data_dir; save predictions only (no metrics)."""
    label_column = training_cfg.data.module.get("label_column", "label")
    batch_size = int(training_cfg.data.module.batch_size)
    num_workers = int(training_cfg.data.module.get("num_workers", 4))

    manifest_path = os.path.join(pred_dir, "_predict_manifest.csv")
    image_ids = _build_predict_manifest(data_dir, label_column, manifest_path)
    print(f"[data_dir] manifest with {len(image_ids)} images written to {manifest_path}")

    test_transforms = instantiate(training_cfg.data.module.test_transforms)

    predict_ds = AgeReg_Data(
        img_dir=str(data_dir),
        csv_file=manifest_path,
        split="test",
        fold=0,
        label_column=label_column,
        transform=test_transforms,
        train=False,
    )

    loader = DataLoader(
        predict_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    predictions = trainer.predict(model, dataloaders=loader)
    _, y_hats = zip(*predictions)
    probas = torch.cat([p.detach().cpu() for (_, p) in y_hats], dim=0)
    preds = (probas > 0.5).sum(dim=1).float()

    if len(image_ids) != len(preds):
        raise RuntimeError(
            f"[data_dir] Length mismatch: {len(image_ids)} ids vs {len(preds)} preds."
        )

    df = pd.DataFrame({
        "PatientID": image_ids,
        "Prediction": preds.numpy(),
    })
    out_path = os.path.join(pred_dir, f"predictions_data_dir_fold{fold_id}.xlsx")
    df.to_excel(out_path, index=False)
    print(f"[data_dir] Saved {len(df)} predictions to {out_path}")


def _run_split(split, model, dataset, trainer, fold_id, pred_dir, bin_width, max_age):
    """
    Run inference on one split ('val' or 'test'), save per-case predictions,
    age-bin reports, and a per-split summary CSV. Returns the metrics dict.
    """
    dataset.setup(split)
    if split == "val":
        loader = dataset.val_dataloader()
        ids_source = dataset.val_dataset.img_files
    elif split == "test":
        loader = dataset.test_dataloader()
        ids_source = dataset.test_dataset.img_files
    else:
        raise ValueError(f"Unknown split: {split!r}")

    predictions = trainer.predict(model, dataloaders=loader)

    ys, y_hats = zip(*predictions)
    targets = torch.cat([y.detach().cpu() for y in ys]).float()
    probas = torch.cat([p.detach().cpu() for (_, p) in y_hats], dim=0)
    preds = (probas > 0.5).sum(dim=1).float()

    patient_ids = list(ids_source)
    if len(patient_ids) != len(preds):
        raise RuntimeError(
            f"[fold {fold_id} / {split}] Length mismatch between patient IDs "
            f"({len(patient_ids)}) and predictions ({len(preds)})."
        )

    metrics = _compute_metrics(preds, targets)
    print(
        f"[fold {fold_id} / {split}] N={metrics['N']}  MAE={metrics['MAE']:.4f}  "
        f"RMSE={metrics['RMSE']:.4f}  ME={metrics['MeanError']:.4f}  "
        f"Pearson={metrics['Pearson_r']:.4f}"
    )

    # Per-case predictions xlsx
    df = pd.DataFrame({
        "PatientID": patient_ids,
        "GroundTruth": targets.numpy(),
        "Prediction": preds.numpy(),
        "AbsError": (preds - targets).abs().numpy(),
        "Error": (preds - targets).numpy(),
    })
    pred_path = os.path.join(pred_dir, f"predictions_{split}_fold{fold_id}.xlsx")
    df.to_excel(pred_path, index=False)
    print(f"[fold {fold_id} / {split}] Saved predictions to {pred_path}")

    # Age-bin report + plots
    _save_bin_report(
        targets, preds,
        tag=f"{split}_fold{fold_id}",
        pred_dir=pred_dir,
        bin_width=bin_width,
        max_age=max_age,
    )

    # Per-split summary CSV
    summary_df = pd.DataFrame([{"Split": split, "Fold": fold_id, **metrics}])
    for col in ["MAE", "RMSE", "MeanError", "Pearson_r"]:
        summary_df[col] = summary_df[col].apply(
            lambda v: round(v, 4) if isinstance(v, (int, float)) and not pd.isna(v) else v
        )
    summary_path = os.path.join(pred_dir, f"summary_{split}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[fold {fold_id} / {split}] Saved summary to {summary_path}")

    return metrics


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="infer")
def inference(cfg):
    try:
        Path("./main.log").unlink()
    except Exception:
        pass

    prefer_best = cfg.get("prefer_best", True)
    bin_width = int(cfg.get("age_bin_width", 10))
    max_age = int(cfg.get("age_bin_max", 100))

    # Resolve directories from the single run_dir input.
    # Layout expected: <run_dir>/{Configs/config.yaml, folds/<k>/*.ckpt}
    run_dir = Path(cfg.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    fold_id = str(cfg.fold) if cfg.get("fold") is not None else "0"

    ckp_dir = run_dir / "folds" / fold_id
    ckp_list = list(ckp_dir.glob("*.ckpt"))
    if not ckp_list:
        raise FileNotFoundError(f"No checkpoints found under {ckp_dir}")

    ckp_path = _select_best_ckpt(ckp_list, prefer_best=prefer_best)
    if ckp_path is None:
        raise FileNotFoundError(f"No usable checkpoint selected from {ckp_dir}")
    print(f"[fold {fold_id}] using checkpoint: {ckp_path}")

    # Locate training config snapshot
    training_config_path = run_dir / "Configs" / "config.yaml"
    if not training_config_path.is_file():
        raise FileNotFoundError(
            f"Training config not found at {training_config_path}"
        )
    print(f"Using training config: {training_config_path}")

    # Predictions / reports output directory
    pred_dir = Path(cfg.pred_dir) if cfg.get("pred_dir") else run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Build model + datamodule + trainer from the saved training config
    used_training_cfg = OmegaConf.load(training_config_path)
    used_training_cfg.trainer.pop("logger", None)
    used_training_cfg.trainer.pop("callbacks", None)
    used_training_cfg.model.metrics = cfg.metrics

    # Force single-device prediction so neither val nor test gets sharded/padded
    used_training_cfg.trainer.devices = 1
    used_training_cfg.trainer.strategy = "auto"
    used_training_cfg.trainer.sync_batchnorm = False

    used_training_cfg.data.module.fold = int(fold_id)

    model = instantiate(used_training_cfg.model)
    state = torch.load(ckp_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()

    trainer = instantiate(used_training_cfg.trainer)

    # Branch on whether a custom data_dir was supplied.
    data_dir = cfg.get("data_dir")
    if data_dir:
        data_dir_path = Path(data_dir)
        if not data_dir_path.is_dir():
            raise FileNotFoundError(f"data_dir does not exist: {data_dir_path}")
        _run_data_dir_predict(
            model=model,
            training_cfg=used_training_cfg,
            trainer=trainer,
            data_dir=data_dir_path,
            pred_dir=str(pred_dir),
            fold_id=fold_id,
        )
    else:
        dataset = instantiate(used_training_cfg.data).module
        for split in ("val", "test"):
            _run_split(
                split=split,
                model=model,
                dataset=dataset,
                trainer=trainer,
                fold_id=fold_id,
                pred_dir=str(pred_dir),
                bin_width=bin_width,
                max_age=max_age,
            )


if __name__ == "__main__":
    make_omegaconf_resolvers()
    inference()