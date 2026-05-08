import glob
import glob
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from omegaconf import OmegaConf

from parsing_utils import make_omegaconf_resolvers


def _select_best_ckpt(ckp_paths, prefer_best=True):
    """
    From a list of checkpoint paths, pick one.
    If prefer_best is True, return the checkpoint with the lowest 'Val_mae' parsed
    from its filename. Falls back to last.ckpt if no parseable filename is found.
    If prefer_best is False, return last.ckpt.
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
    ground-truth age. Returns a DataFrame indexed by bin label with columns:
        N, MAE, RMSE, MeanError, StdError
    Bins are [0, 10), [10, 20), ..., up to max_age. Anything >= max_age is
    grouped into a final ">=max_age" bin (kept for safety; usually empty).
    """
    targets_np = targets.detach().cpu().numpy().astype(float)
    preds_np = preds.detach().cpu().numpy().astype(float)
    err_np = preds_np - targets_np
    abs_err_np = np.abs(err_np)

    edges = np.arange(0, max_age + bin_width, bin_width)  # [0, 10, 20, ..., max_age]
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

    # Catch anything at or beyond max_age
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

    df = pd.DataFrame(rows)
    return df


def _plot_error_bars(bin_df, title, out_path, metric="MAE"):
    """
    Plot a bar chart of the chosen error metric per age bin and save to disk.
    Empty bins (N=0) are still drawn as gaps. Sample counts (N) are annotated
    on top of each bar.
    """
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(bin_df)), 5))

    bins = bin_df["Bin"].tolist()
    values = bin_df[metric].to_numpy(dtype=float)
    counts = bin_df["N"].to_numpy(dtype=int)

    # Replace NaN with 0 for plotting but keep track to label them as "n=0"
    plot_values = np.where(np.isnan(values), 0.0, values)

    bars = ax.bar(bins, plot_values, color="steelblue", edgecolor="black")

    # Annotate each bar with sample count
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

    # A horizontal reference line at 0 helps for signed-error plots
    if metric == "MeanError":
        ax.axhline(0, color="black", linewidth=0.8)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved plot: {out_path}")


def _save_bin_report(targets, preds, tag, exp_dir, bin_width=10, max_age=100):
    """
    Compute age-binned errors, save them as CSV, and produce MAE / MeanError
    bar charts. `tag` is used in filenames (e.g. 'fold0', 'ensemble_avg_probas').
    """
    bin_df = _bin_errors_by_age(targets, preds, bin_width=bin_width, max_age=max_age)

    csv_path = os.path.join(exp_dir, f"error_by_age_bin_{tag}.csv")
    bin_df.to_csv(csv_path, index=False)
    print(f"[{tag}] saved per-bin error stats to {csv_path}")

    mae_plot_path = os.path.join(exp_dir, f"error_by_age_bin_{tag}_MAE.png")
    _plot_error_bars(
        bin_df,
        title=f"MAE per age bin ({tag})",
        out_path=mae_plot_path,
        metric="MAE",
    )

    me_plot_path = os.path.join(exp_dir, f"error_by_age_bin_{tag}_MeanError.png")
    _plot_error_bars(
        bin_df,
        title=f"Mean signed error per age bin ({tag})",
        out_path=me_plot_path,
        metric="MeanError",
    )

    return bin_df


@hydra.main(version_base=None, config_path="./cli_configs", config_name="infer")
def inference(cfg):
    try:
        Path("./main.log").unlink()
    except Exception:
        pass

    prefer_best = cfg.get("prefer_best", True)

    # Allow overriding the bin width / max age from the config if desired.
    bin_width = int(cfg.get("age_bin_width", 10))
    max_age = int(cfg.get("age_bin_max", 100))

    # Gather checkpoints per fold
    if cfg.fold:
        candidate_paths = list(Path(Path(cfg.ckpt_dir) / str(cfg.fold)).glob("*.ckpt"))
        fold_groups = {str(cfg.fold): candidate_paths}
        candidate_paths = list(Path(Path(cfg.ckpt_dir) / str(cfg.fold)).glob("*.ckpt"))
        fold_groups = {str(cfg.fold): candidate_paths}
    else:
        all_paths = list(Path(cfg.ckpt_dir).glob("*/*.ckpt"))
        fold_groups = {}
        for p in all_paths:
            fold_groups.setdefault(p.parent.name, []).append(p)

    if not fold_groups:
        raise FileNotFoundError(f"No checkpoints found under {cfg.ckpt_dir}")

    # Locate training config
    matches = glob.glob(os.path.join(cfg.exp_dir, "*/config.yaml"))
    if not matches:
        raise FileNotFoundError(
            f"No config.yaml found under {cfg.exp_dir}/*/config.yaml"
        )
    used_training_config_path = matches[0]
    print(f"Using training config: {used_training_config_path}")

    fold_ids_sorted = sorted(fold_groups.keys(), key=lambda x: (len(x), x))

    summary_rows = []
    all_probas = []      # one entry per fold: tensor [N, K-1]
    all_preds = []       # one entry per fold: tensor [N]
    targets_ref = None
    patient_ids_ref = None

    for fold_id in fold_ids_sorted:
        ckp_list = fold_groups[fold_id]
        ckp_path = _select_best_ckpt(ckp_list, prefer_best=prefer_best)
        if ckp_path is None:
            print(f"[fold {fold_id}] no checkpoint found, skipping")
            continue
        print(f"[fold {fold_id}] using checkpoint: {ckp_path}")

        # Load training config fresh (we mutate it)
        all_paths = list(Path(cfg.ckpt_dir).glob("*/*.ckpt"))
        fold_groups = {}
        for p in all_paths:
            fold_groups.setdefault(p.parent.name, []).append(p)

    if not fold_groups:
        raise FileNotFoundError(f"No checkpoints found under {cfg.ckpt_dir}")

    # Locate training config
    matches = glob.glob(os.path.join(cfg.exp_dir, "*/config.yaml"))
    if not matches:
        raise FileNotFoundError(
            f"No config.yaml found under {cfg.exp_dir}/*/config.yaml"
        )
    used_training_config_path = matches[0]
    print(f"Using training config: {used_training_config_path}")

    fold_ids_sorted = sorted(fold_groups.keys(), key=lambda x: (len(x), x))

    summary_rows = []
    all_probas = []      # one entry per fold: tensor [N, K-1]
    all_preds = []       # one entry per fold: tensor [N]
    targets_ref = None
    patient_ids_ref = None

    for fold_id in fold_ids_sorted:
        ckp_list = fold_groups[fold_id]
        ckp_path = _select_best_ckpt(ckp_list, prefer_best=prefer_best)
        if ckp_path is None:
            print(f"[fold {fold_id}] no checkpoint found, skipping")
            continue
        print(f"[fold {fold_id}] using checkpoint: {ckp_path}")

        # Load training config fresh (we mutate it)
        used_training_cfg = OmegaConf.load(used_training_config_path)
        used_training_cfg.trainer.pop("logger", None)
        used_training_cfg.trainer.pop("callbacks", None)
        used_training_cfg.model.metrics = cfg.metrics

        # Force single-device prediction so the test set isn't sharded/padded
        used_training_cfg.trainer.devices = 1
        used_training_cfg.trainer.strategy = "auto"
        used_training_cfg.trainer.sync_batchnorm = False

        used_training_cfg.data.module.fold = int(fold_id)

        used_training_cfg.trainer.pop("logger", None)
        used_training_cfg.trainer.pop("callbacks", None)
        used_training_cfg.model.metrics = cfg.metrics

        # Force single-device prediction so the test set isn't sharded/padded
        used_training_cfg.trainer.devices = 1
        used_training_cfg.trainer.strategy = "auto"
        used_training_cfg.trainer.sync_batchnorm = False

        used_training_cfg.data.module.fold = int(fold_id)

        model = instantiate(used_training_cfg.model)
        state = torch.load(ckp_path, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        state = torch.load(ckp_path, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        model.eval()

        dataset = instantiate(used_training_cfg.data).module
        dataset.setup("test")

        dataset.setup("test")

        trainer = instantiate(used_training_cfg.trainer)

        predictions = trainer.predict(
            model, dataloaders=dataset.test_dataloader()
        )

        predictions = trainer.predict(
            model, dataloaders=dataset.test_dataloader()
        )

        ys, y_hats = zip(*predictions)
        targets = torch.cat([y.detach().cpu() for y in ys]).float()
        probas = torch.cat([p.detach().cpu() for (_, p) in y_hats], dim=0)
        preds = (probas > 0.5).sum(dim=1).float()

        # Sanity: verify all folds see the same test set in the same order.
        # If they don't, ensemble averaging across folds would silently misalign.
        patient_ids = list(dataset.test_dataset.img_files)
        if targets_ref is None:
            targets_ref = targets
            patient_ids_ref = patient_ids
        else:
            if patient_ids != patient_ids_ref:
                raise RuntimeError(
                    f"[fold {fold_id}] Test patient IDs differ from fold "
                    f"{fold_ids_sorted[0]}. Cannot ensemble: each fold sees a "
                    f"different test set. Inspect splits.json."
                )
            if not torch.allclose(targets, targets_ref):
                raise RuntimeError(
                    f"[fold {fold_id}] Test targets differ from fold "
                    f"{fold_ids_sorted[0]} (same IDs, different labels?)."
                )

        ys, y_hats = zip(*predictions)
        targets = torch.cat([y.detach().cpu() for y in ys]).float()
        probas = torch.cat([p.detach().cpu() for (_, p) in y_hats], dim=0)
        preds = (probas > 0.5).sum(dim=1).float()

        # Sanity: verify all folds see the same test set in the same order.
        # If they don't, ensemble averaging across folds would silently misalign.
        patient_ids = list(dataset.test_dataset.img_files)
        if targets_ref is None:
            targets_ref = targets
            patient_ids_ref = patient_ids
        else:
            if patient_ids != patient_ids_ref:
                raise RuntimeError(
                    f"[fold {fold_id}] Test patient IDs differ from fold "
                    f"{fold_ids_sorted[0]}. Cannot ensemble: each fold sees a "
                    f"different test set. Inspect splits.json."
                )
            if not torch.allclose(targets, targets_ref):
                raise RuntimeError(
                    f"[fold {fold_id}] Test targets differ from fold "
                    f"{fold_ids_sorted[0]} (same IDs, different labels?)."
                )

        if len(patient_ids) != len(preds):
            raise RuntimeError(
                f"[fold {fold_id}] Length mismatch between patient IDs "
                f"({len(patient_ids)}) and predictions ({len(preds)})."
            )

        # Per-fold metrics
        metrics = _compute_metrics(preds, targets)
        print(
            f"[fold {fold_id}] N={metrics['N']}  MAE={metrics['MAE']:.4f}  "
            f"RMSE={metrics['RMSE']:.4f}  ME={metrics['MeanError']:.4f}  "
            f"Pearson={metrics['Pearson_r']:.4f}"
        )

        # Per-fold predictions xlsx
        df = pd.DataFrame(
            {
                "PatientID": patient_ids,
                "GroundTruth": targets.numpy(),
                "Prediction": preds.numpy(),
                "AbsError": (preds - targets).abs().numpy(),
                "Error": (preds - targets).numpy(),
            }
        )
        out_path = os.path.join(cfg.exp_dir, f"predictions_fold{fold_id}.xlsx")
            raise RuntimeError(
                f"[fold {fold_id}] Length mismatch between patient IDs "
                f"({len(patient_ids)}) and predictions ({len(preds)})."
            )

        # Per-fold metrics
        metrics = _compute_metrics(preds, targets)
        print(
            f"[fold {fold_id}] N={metrics['N']}  MAE={metrics['MAE']:.4f}  "
            f"RMSE={metrics['RMSE']:.4f}  ME={metrics['MeanError']:.4f}  "
            f"Pearson={metrics['Pearson_r']:.4f}"
        )

        # Per-fold predictions xlsx
        df = pd.DataFrame(
            {
                "PatientID": patient_ids,
                "GroundTruth": targets.numpy(),
                "Prediction": preds.numpy(),
                "AbsError": (preds - targets).abs().numpy(),
                "Error": (preds - targets).numpy(),
            }
        )
        out_path = os.path.join(cfg.exp_dir, f"predictions_fold{fold_id}.xlsx")
        df.to_excel(out_path, index=False)
        print(f"[fold {fold_id}] Saved predictions to {out_path}")

        # Per-fold age-bin error report + bar charts
        _save_bin_report(
            targets, preds,
            tag=f"fold{fold_id}",
            exp_dir=cfg.exp_dir,
            bin_width=bin_width,
            max_age=max_age,
        )

        summary_rows.append({"Fold": fold_id, **metrics})
        all_probas.append(probas)
        all_preds.append(preds)

    if not summary_rows:
        print("No folds were processed; nothing to summarize.")
        return

    summary_df = pd.DataFrame(summary_rows)
    per_fold_metrics = ["MAE", "RMSE", "MeanError", "Pearson_r"]

    # ----- Cross-fold mean ± std (CV-style report) -----
    means = summary_df[per_fold_metrics].mean()
    stds = (
        summary_df[per_fold_metrics].std(ddof=1)
        if len(summary_df) > 1
        else pd.Series({m: float("nan") for m in per_fold_metrics})
    )
    mean_row = {
        "Fold": "MEAN_OF_FOLDS",
        "N": int(summary_df["N"].iloc[0]),  # all folds share the same test set
        **{m: means[m] for m in per_fold_metrics},
    }
    std_row = {
        "Fold": "STD_OF_FOLDS",
        "N": "",
        **{m: stds[m] for m in per_fold_metrics},
    }

    # ----- Ensemble: average the predictions across folds, then decode -----
    # Two reasonable ways to ensemble for CORAL ordinal regression:
    #   (a) Average probas across folds, then threshold at 0.5 ("ensemble probas")
    #   (b) Average per-fold age predictions ("ensemble preds")
    # Both are reported below; (a) is usually preferred since it averages the
    # underlying threshold probabilities before decoding.
    if len(all_probas) > 1:
        stacked_probas = torch.stack(all_probas, dim=0)        # [folds, N, K-1]
        ensemble_probas = stacked_probas.mean(dim=0)           # [N, K-1]
        ens_preds_from_probas = (ensemble_probas > 0.5).sum(dim=1).float()
        ens_metrics_a = _compute_metrics(ens_preds_from_probas, targets_ref)

        stacked_preds = torch.stack(all_preds, dim=0).float()  # [folds, N]
        ens_preds_from_preds = stacked_preds.mean(dim=0)       # [N]
        ens_metrics_b = _compute_metrics(ens_preds_from_preds, targets_ref)

        ens_a_row = {"Fold": "ENSEMBLE_avg_probas", **ens_metrics_a}
        ens_b_row = {"Fold": "ENSEMBLE_avg_preds", **ens_metrics_b}
        extra_rows = [mean_row, std_row, ens_a_row, ens_b_row]

        # Age-bin reports for the two ensemble variants
        _save_bin_report(
            targets_ref, ens_preds_from_probas,
            tag="ensemble_avg_probas",
            exp_dir=cfg.exp_dir,
            bin_width=bin_width,
            max_age=max_age,
        )
        _save_bin_report(
            targets_ref, ens_preds_from_preds,
            tag="ensemble_avg_preds",
            exp_dir=cfg.exp_dir,
            bin_width=bin_width,
            max_age=max_age,
        )
    else:
        ens_preds_from_probas = None
        ens_preds_from_preds = None
        extra_rows = [mean_row, std_row]

    summary_df_full = pd.concat(
        [summary_df, pd.DataFrame(extra_rows)], ignore_index=True
    )

    # Round for readability
    for col in per_fold_metrics:
        summary_df_full[col] = summary_df_full[col].apply(
            lambda v: round(v, 4) if isinstance(v, (int, float)) and not pd.isna(v) else v
        )

    summary_path = os.path.join(cfg.exp_dir, "summary.csv")
    summary_df_full.to_csv(summary_path, index=False)
    print("\n========== Hold-out test summary ==========")
    print(summary_df_full.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")

    # ----- Save ensemble predictions for inspection -----
    if ens_preds_from_probas is not None:
        ens_df = pd.DataFrame(
            {
                "PatientID": patient_ids_ref,
                "GroundTruth": targets_ref.numpy(),
                "Pred_AvgProbas": ens_preds_from_probas.numpy(),
                "Pred_AvgPreds": ens_preds_from_preds.numpy(),
                "AbsError_AvgProbas": (ens_preds_from_probas - targets_ref).abs().numpy(),
                "AbsError_AvgPreds": (ens_preds_from_preds - targets_ref).abs().numpy(),
            }
        )
        ens_path = os.path.join(cfg.exp_dir, "predictions_ensemble.xlsx")
        ens_df.to_excel(ens_path, index=False)
        print(f"Saved ensemble predictions to {ens_path}")


if __name__ == "__main__":
    make_omegaconf_resolvers()
    inference()