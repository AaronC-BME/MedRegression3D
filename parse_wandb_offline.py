"""
Parse an offline wandb run (.wandb file) without internet, dump all logged
metrics to CSV, and plot train/val loss + train/val MAE + train/val MSE.

This version reads BOTH:
  - history records (for things logged via wandb.log, with nested_key support)
  - summary records (for per-epoch Lightning metrics which only show up
    as summary updates in newer wandb versions)

Usage:
    python parse_wandb_offline.py \
        --run-dir /path/to/offline-run-<timestamp>-<id> \
        --out-dir ./wandb_offline_plots
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore


DEFAULT_RUN_DIR = (
    "/home/jma/projects/aaron/Dataset/nnssl_dataset/nnssl_results/"
    "Dataset003_LiverAge/ResNetL-VOCO/liver/wandb/"
    "offline-run-20260508_175949-7dgue773"
)


def _decode_json(s: str):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


def _item_key(item) -> str:
    """Flat key string, handling both legacy `key` and newer `nested_key`."""
    nested = list(item.nested_key)
    if nested:
        return "/".join(nested)
    return item.key


def find_wandb_file(run_dir: Path) -> Path:
    candidates = list(run_dir.glob("run-*.wandb"))
    if not candidates:
        raise FileNotFoundError(f"No run-*.wandb file found in {run_dir}")
    if len(candidates) > 1:
        print(f"[warn] multiple .wandb files found, using {candidates[0]}")
    return candidates[0]


def parse_records(wandb_file: Path):
    """Walk every record, returning a history DataFrame and a
    summary-snapshot DataFrame (one row per summary update event, with
    the full running state at that moment)."""
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))

    history_rows = []
    summary_snapshots = []
    summary_state = {}

    n_records = 0
    while True:
        data = ds.scan_data()
        if data is None:
            break
        n_records += 1

        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)
        rtype = rec.WhichOneof("record_type")

        if rtype == "history":
            row = {_item_key(it): _decode_json(it.value_json)
                   for it in rec.history.item}
            history_rows.append(row)

        elif rtype == "summary":
            for item in rec.summary.update:
                summary_state[_item_key(item)] = _decode_json(item.value_json)
            if any(k in summary_state for k in ("_step", "epoch", "_runtime")):
                summary_snapshots.append(dict(summary_state))

    print(f"[info] scanned {n_records} records "
          f"({len(history_rows)} history, {len(summary_snapshots)} summary snapshots)")

    history_df = pd.DataFrame(history_rows)
    summary_df = pd.DataFrame(summary_snapshots)

    for df in (history_df, summary_df):
        if "_step" in df.columns:
            df.sort_values("_step", inplace=True)
            df.reset_index(drop=True, inplace=True)

    return history_df, summary_df


def extract_epoch_curves(summary_df: pd.DataFrame, metric_keys) -> pd.DataFrame:
    """Collapse the summary timeline into one row per epoch (or step)."""
    if summary_df.empty:
        return pd.DataFrame()

    have = [k for k in metric_keys if k in summary_df.columns]
    if not have:
        return pd.DataFrame()

    x_key = "epoch" if "epoch" in summary_df.columns else "_step"
    cols = [x_key] + have
    sub = summary_df[cols].dropna(subset=have, how="all").copy()
    sub = sub.groupby(x_key, as_index=False).last()
    return sub


def plot_pair(curves: pd.DataFrame, train_key: str, val_key: str,
              ylabel: str, out_path: Path):
    have_train = train_key in curves.columns
    have_val = val_key in curves.columns
    if not (have_train or have_val):
        print(f"[skip] neither {train_key} nor {val_key} present")
        return

    x_key = "epoch" if "epoch" in curves.columns else "_step"
    fig, ax = plt.subplots(figsize=(8, 5))

    if have_train:
        sub = curves[[x_key, train_key]].dropna()
        ax.plot(sub[x_key], sub[train_key], label="Train", linewidth=1.8)
    if have_val:
        sub = curves[[x_key, val_key]].dropna()
        ax.plot(sub[x_key], sub[val_key], label="Val", linewidth=1.8)

    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} over training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok]   wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=str, default="./wandb_offline_plots")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_file = find_wandb_file(run_dir)
    print(f"[info] parsing {wandb_file}")

    history_df, summary_df = parse_records(wandb_file)

    if not history_df.empty:
        p = out_dir / "history.csv"
        history_df.to_csv(p, index=False)
        print(f"[ok]   wrote {p}  ({len(history_df)} rows, {len(history_df.columns)} cols)")

    if not summary_df.empty:
        p = out_dir / "summary_timeline.csv"
        summary_df.to_csv(p, index=False)
        print(f"[ok]   wrote {p}  ({len(summary_df)} rows, {len(summary_df.columns)} cols)")

    metric_keys = [
        "Train/loss", "Val/loss",
        "Train/MAE", "Val/MAE",
        "Train/MSE", "Val/MSE",
    ]
    if not summary_df.empty:
        print("[info] sample of summary keys: "
              f"{sorted(summary_df.columns)[:40]}")

    curves = extract_epoch_curves(summary_df, metric_keys)
    if curves.empty:
        print("[warn] none of the target metric keys were found in summary.")
        print("       Inspect summary_timeline.csv to see what's actually logged.")
        return

    p = out_dir / "metrics_per_epoch.csv"
    curves.to_csv(p, index=False)
    print(f"[ok]   wrote {p}  ({len(curves)} epochs)")

    plot_pair(curves, "Train/loss", "Val/loss",
              ylabel="Loss", out_path=out_dir / "loss.png")
    plot_pair(curves, "Train/MAE", "Val/MAE",
              ylabel="MAE", out_path=out_dir / "mae.png")
    plot_pair(curves, "Train/MSE", "Val/MSE",
              ylabel="MSE", out_path=out_dir / "mse.png")


if __name__ == "__main__":
    main()