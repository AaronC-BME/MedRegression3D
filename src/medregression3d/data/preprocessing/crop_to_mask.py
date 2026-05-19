import os
import glob
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool
from functools import partial

from datasets.preprocess_3D_data.blosc_helper import save_case, comp_blosc2_params


def load_image_np(path):
    """Load NIfTI image and return NumPy array and reference image for metadata."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def get_mask_center(mask_arr):
    """Return (z, y, x) center of non-zero mask voxels."""
    coords = np.argwhere(mask_arr == 1)
    if coords.size == 0:
        raise ValueError("Mask is empty (no voxels with value 1).")
    return tuple(coords.mean(axis=0).astype(int))


def crop_center_with_padding_np(arr, center, size=(160, 160, 160)):
    """Crop a 3D NumPy array with zero padding around a center (z, y, x)."""
    zc, yc, xc = center
    dz, dy, dx = size[0] // 2, size[1] // 2, size[2] // 2

    z_start, z_end = zc - dz, zc + dz
    y_start, y_end = yc - dy, yc + dy
    x_start, x_end = xc - dx, xc + dx

    cropped = np.zeros(size, dtype=arr.dtype)

    z_start_valid = max(z_start, 0)
    y_start_valid = max(y_start, 0)
    x_start_valid = max(x_start, 0)

    z_end_valid = min(z_end, arr.shape[0])
    y_end_valid = min(y_end, arr.shape[1])
    x_end_valid = min(x_end, arr.shape[2])

    z_off = z_start_valid - z_start
    y_off = y_start_valid - y_start
    x_off = x_start_valid - x_start

    cropped[
    z_off:z_off + (z_end_valid - z_start_valid),
    y_off:y_off + (y_end_valid - y_start_valid),
    x_off:x_off + (x_end_valid - x_start_valid)
    ] = arr[
        z_start_valid:z_end_valid,
        y_start_valid:y_end_valid,
        x_start_valid:x_end_valid
        ]

    return cropped
