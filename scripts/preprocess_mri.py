"""
General-purpose MRI preprocessing for SSL3D_regression.

Reads raw NIfTI MR images, then for each case:
  1. Resamples to a target spacing (default 1mm isotropic).
  2. Crops to the non-zero bounding box (matches nnssl behavior).
  3. Applies per-case z-score normalization on the foreground (voxels > 0).
  4. Saves as Blosc2 in the directory layout the AgeReg dataloader expects:
        <out_root>/<dataset-name>/<id>.b2nd

Notes:
  - This script is dataset-agnostic. Use --dataset-name to set the folder.
  - Unlike CT, MRI has no absolute intensity reference (HU). Intensities vary
    by scanner, sequence, and acquisition, so dataset-wide stats are not
    meaningful. Each case is z-scored independently on its own foreground.
  - Foreground is defined as voxels > 0. This assumes the input MRIs have
    been brain-extracted/skull-stripped or otherwise have a zero background.
    If your data has a non-zero air background, mask it out first or modify
    the normalizer.
  - Resampling order is 3 (cubic) for image data, matching the SSL3D template.
  - If your data is already at the target spacing, resampling is a near-no-op
    (one short shape check) and adds negligible time. Pass --skip-resample to
    skip it explicitly.
  - Random patch extraction to 160^3 happens at training time via
    batchgenerators, not here. We preserve full resampled volumes.

Usage:
    python MRI_preprocessing.py \\
        --in-dir <path_to_input_directory> \\
        --out-root <path_to_output_root> \\
        --dataset-name <dataset_name> \\
        --target-spacing 1 1 1 \\
        --num-workers 8
"""
import sys
import os
import argparse
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

# Make the parent package importable when running this script directly,
# matching the pattern used by template_brain_preprocessing.py.
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
)

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from datasets.preprocess_3D_data.cropping import crop_to_nonzero
from datasets.preprocess_3D_data.normalization import ZScoreNormalization
from datasets.preprocess_3D_data.blosc_helper import save_case, comp_blosc2_params
from datasets.preprocess_3D_data.default_resampling import resample_data_or_seg_to_spacing

# Cubic interpolation for MR image data. Matches SSL3D template default.
RESAMPLING_ORDER = 3


# --------------------------------------------------------------------------- #
# Per-case processing
# --------------------------------------------------------------------------- #
def process_one_case(args: tuple) -> Optional[str]:
    """
    Worker function. Takes a tuple so it works with Pool.imap.
    Returns the case ID on success, None on failure.

    Pipeline order matters here:
        1. Load
        2. Resample to target spacing (changes voxel grid, preserves anatomy)
        3. Crop to nonzero (safe to do after resampling - bbox tracks anatomy)
        4. Per-case z-score on the foreground mask (voxels > 0)
        5. Save
    """
    image_path, out_path_truncated, target_spacing, skip_resample = args
    case_id = Path(out_path_truncated).name

    try:
        # ---- 1. Load ----
        img = sitk.ReadImage(image_path)
        # SimpleITK GetSpacing returns (x, y, z); array is (z, y, x) — invert.
        original_spacing = img.GetSpacing()[::-1]
        data = sitk.GetArrayFromImage(img)  # (Z, Y, X)

        # Add channel dim for compatibility with cropping/resampling utilities
        data = data[np.newaxis, ...].astype(np.float32, copy=False)  # (1, Z, Y, X)

        # Sanity check
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print(f"[skip] {case_id}: NaN/Inf in input")
            return None

        # ---- 2. Resample to target spacing ----
        if not skip_resample:
            target = list(target_spacing)
            # Skip the resample if already at target spacing (saves time, avoids
            # unnecessary interpolation noise).
            already_target = all(
                abs(o - t) < 1e-3 for o, t in zip(original_spacing, target)
            )
            if not already_target:
                data = resample_data_or_seg_to_spacing(
                    data,
                    original_spacing,
                    target,
                    is_seg=False,
                    order=RESAMPLING_ORDER,
                )

        # ---- 3. Crop to non-zero bounding box ----
        # For skull-stripped MRI this often trims a sizeable margin.
        data, _seg, _bbox = crop_to_nonzero(data, seg=None)

        # ---- 4. Per-case z-score on foreground (voxels > 0) ----
        foreground_mask = data[0] > 0
        if not foreground_mask.any():
            print(f"[skip] {case_id}: empty foreground after cropping")
            return None

        normalizer = ZScoreNormalization()
        # Pass the foreground mask as a seg-shaped tensor so the normalizer
        # restricts mean/std to true tissue voxels.
        data = normalizer.run(data, seg=foreground_mask[np.newaxis, ...])

        # ---- 5. Save as Blosc2 ----
        # save_case typically appends ".b2nd" itself; we pass the path without
        # the extension to match the existing helper's convention.
        os.makedirs(os.path.dirname(out_path_truncated), exist_ok=True)
        block_size, chunk_size = comp_blosc2_params(
            data.shape, (160, 160, 160), data.itemsize
        )
        save_case(data, out_path_truncated, chunks=chunk_size, blocks=block_size)

        return case_id

    except Exception as e:
        print(f"[error] {case_id}: {e}")
        return None


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--in-dir", required=True, type=Path,
                        help="Directory containing raw .nii.gz MR images.")
    parser.add_argument("--out-root", required=True, type=Path,
                        help="Output root. The script will write to "
                             "<out-root>/<dataset-name>/<id>.b2nd")
    parser.add_argument("--dataset-name", required=True, type=str,
                        help="Name of the dataset folder, e.g. Dataset017_OpenNeuro.")
    parser.add_argument("--target-spacing", type=float, nargs=3,
                        default=[1.0, 1.0, 1.0], metavar=("Z", "Y", "X"),
                        help="Target spacing in mm as three floats: Z Y X. Default: 1 1 1.")
    parser.add_argument("--skip-resample", action="store_true",
                        help="Skip resampling entirely (use the data as-is).")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel processes. Default: 8")

    args = parser.parse_args()

    if not args.in_dir.is_dir():
        raise SystemExit(f"--in-dir does not exist: {args.in_dir}")

    image_paths = sorted(str(p) for p in args.in_dir.glob("*.nii.gz"))
    if not image_paths:
        raise SystemExit(f"No .nii.gz files found in {args.in_dir}")

    print(f"Found {len(image_paths)} MR images.")
    if args.skip_resample:
        print("[note] --skip-resample set. Volumes will be saved at their native spacing.")
    else:
        print(f"[note] target spacing (Z Y X): {tuple(args.target_spacing)} mm")

    # ---- Build per-case output paths and dispatch ---- #
    out_root = args.out_root / args.dataset_name
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting outputs to {out_root}/<id>.b2nd")

    job_args = []
    for img_path in image_paths:
        case_id = Path(img_path).name
        if case_id.endswith(".nii.gz"):
            case_id = case_id[:-len(".nii.gz")]
        elif case_id.endswith(".nii"):
            case_id = case_id[:-len(".nii")]
        out_path_truncated = str(out_root / case_id)
        job_args.append((
            img_path,
            out_path_truncated,
            tuple(args.target_spacing),
            args.skip_resample,
        ))

    print(f"\nProcessing {len(job_args)} cases with {args.num_workers} workers...")
    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_one_case, job_args),
                                total=len(job_args)))
    else:
        results = [process_one_case(j) for j in tqdm(job_args)]

    n_ok = sum(1 for r in results if r is not None)
    n_fail = len(results) - n_ok
    print(f"\nDone. {n_ok} succeeded, {n_fail} failed.")
    if n_fail:
        print("Failed cases were printed above with [error] or [skip] tags.")


if __name__ == "__main__":
    main()