"""
Inspect Blosc2 (.b2nd) image files in a directory (recursively).

Reports per-file shape, dtype, and (optionally) min/max/mean/std for each
.b2nd file found anywhere under --in-dir. Also prints summary statistics:
  - total number of files
  - distribution of shapes (how many files have each unique shape)
  - per-axis size statistics
  - aggregate intensity range across the dataset

Useful for sanity-checking a preprocessed dataset before training.

Examples:
    # Just see how many files and what shapes they have (fast):
    python check_b2nd_shapes.py --in-dir /path/to/Dataset001_LiverROI

    # Print the shape of every single file:
    python check_b2nd_shapes.py --in-dir /path/to/Dataset001_LiverROI --per-file

    # Same, but also compute intensity stats (slower, loads each volume):
    python check_b2nd_shapes.py --in-dir /path/to/Dataset001_LiverROI --per-file --stats

    # Spot-check the first 10 files only:
    python check_b2nd_shapes.py --in-dir /path/to/Dataset001_LiverROI --per-file --limit 10
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import blosc2
import numpy as np


def inspect_one(path: Path, with_stats: bool = False) -> Optional[dict]:
    """
    Open one .b2nd file and return a dict of {shape, dtype, [min, max, mean, std]}.
    If `with_stats` is False (default), only shape + dtype are returned (fast).
    Returns None on error.
    """
    try:
        arr = blosc2.open(str(path), mode="r")
        shape = tuple(arr.shape)
        dtype = str(arr.dtype)
        info = {"shape": shape, "dtype": dtype}
        if with_stats:
            # Materialize values for stats. b2nd is chunked so this can be slow
            # on huge volumes; omit --stats if you only care about geometry.
            data = arr[...]
            info.update(
                min=float(data.min()),
                max=float(data.max()),
                mean=float(data.mean()),
                std=float(data.std()),
            )
        return info
    except Exception as e:
        print(f"[error] {path}: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--in-dir", required=True, type=Path,
                        help="Directory to search recursively for .b2nd files. "
                             "E.g. /path/to/nnssl_preprocessed/Dataset001_LiverROI")
    parser.add_argument("--per-file", action="store_true",
                        help="Print one line per file (otherwise just the summary).")
    parser.add_argument("--stats", action="store_true",
                        help="Also compute intensity statistics (min/max/mean/std) "
                             "per file. Slower, since it must load each volume.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only inspect the first N files (useful for spot-checking).")
    args = parser.parse_args()

    if not args.in_dir.is_dir():
        raise SystemExit(f"--in-dir does not exist: {args.in_dir}")

    files = sorted(args.in_dir.rglob("*.b2nd"))
    if not files:
        raise SystemExit(f"No .b2nd files found under {args.in_dir}")

    if args.limit is not None:
        files = files[: args.limit]

    print(f"Found {len(files)} .b2nd file(s) under {args.in_dir}")
    if not args.stats:
        print("(skipping intensity statistics; pass --stats to compute them)")
    print()

    shape_counter: Counter = Counter()
    dtype_counter: Counter = Counter()
    axis_sizes: list[list[int]] = [[], [], [], [], []]  # supports up to 5D
    all_mins, all_maxs, all_means, all_stds = [], [], [], []
    n_ok = 0
    n_fail = 0

    if args.per_file:
        if args.stats:
            header = (f"{'file':<70s} {'shape':<28s} {'dtype':<10s} "
                      f"{'min':>10s} {'max':>10s} {'mean':>10s} {'std':>10s}")
        else:
            header = f"{'file':<70s} {'shape':<28s} {'dtype':<10s}"
        print(header)
        print("-" * len(header))

    for path in files:
        info = inspect_one(path, with_stats=args.stats)
        if info is None:
            n_fail += 1
            continue
        n_ok += 1

        shape = info["shape"]
        dtype = info["dtype"]
        shape_counter[shape] += 1
        dtype_counter[dtype] += 1
        for i, dim in enumerate(shape):
            if i < len(axis_sizes):
                axis_sizes[i].append(dim)

        if args.stats:
            all_mins.append(info["min"])
            all_maxs.append(info["max"])
            all_means.append(info["mean"])
            all_stds.append(info["std"])

        if args.per_file:
            display_path = str(path.relative_to(args.in_dir))
            if len(display_path) > 68:
                display_path = "..." + display_path[-65:]
            shape_str = str(shape)
            if args.stats:
                print(f"{display_path:<70s} {shape_str:<28s} {dtype:<10s} "
                      f"{info['min']:>10.3f} {info['max']:>10.3f} "
                      f"{info['mean']:>10.3f} {info['std']:>10.3f}")
            else:
                print(f"{display_path:<70s} {shape_str:<28s} {dtype:<10s}")

    if args.per_file:
        print()

    # ----- Summary section -----
    print("=" * 70)
    print(f"Summary of {n_ok} file(s)" + (f" ({n_fail} failed)" if n_fail else ""))
    print("=" * 70)

    if dtype_counter:
        print("\nDtypes:")
        for dt, n in dtype_counter.most_common():
            print(f"  {dt}: {n}")

    if shape_counter:
        print(f"\nUnique shapes: {len(shape_counter)}")
        if len(shape_counter) <= 20:
            print("Shape histogram:")
            for shp, n in shape_counter.most_common():
                print(f"  {shp}: {n}")
        else:
            print("(top 10 shown; total >20 unique shapes)")
            for shp, n in shape_counter.most_common(10):
                print(f"  {shp}: {n}")

        print("\nPer-axis size statistics (across all files):")
        for i, sizes in enumerate(axis_sizes):
            if not sizes:
                continue
            arr = np.array(sizes)
            print(f"  axis {i}: min={arr.min():>5d}  max={arr.max():>5d}  "
                  f"mean={arr.mean():>7.1f}  median={int(np.median(arr)):>5d}")

    if args.stats and all_means:
        print("\nIntensity statistics (across all files):")
        print(f"  mins:  range [{min(all_mins):.3f}, {max(all_mins):.3f}]")
        print(f"  maxs:  range [{min(all_maxs):.3f}, {max(all_maxs):.3f}]")
        print(f"  means: range [{min(all_means):.3f}, {max(all_means):.3f}]  "
              f"avg of means: {np.mean(all_means):.3f}")
        print(f"  stds:  range [{min(all_stds):.3f}, {max(all_stds):.3f}]  "
              f"avg of stds:  {np.mean(all_stds):.3f}")
        print()
        print("If preprocessing applied normalization correctly:")
        print("  - ZScore (per-image):  per-file means~0, per-file stds~1")
        print("  - CTNormalization:     per-file means~0, per-file stds~1, range bounded")
        print("  - Raw HU values:       per-file mins<-500, maxs>500, means in HU range")


if __name__ == "__main__":
    main()