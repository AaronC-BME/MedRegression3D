import os
import numpy as np
import pandas as pd
import json
from datasets.preprocess_3D_data.utils_preprocessing import get_median_spacing_and_size_distributed, run_all_cases
from datasets.preprocess_3D_data.cross_validation import generate_crossval_split

def preprocess_dataset_toshape(
        nii_files: list,
        unique_identifiers: list,
        label_dict: dict,
        target_img_size,  # Union[str, Tuple[int, int, int]] – left untyped for flexibility
        num_worker: int,
        out_folder: str
):
    """
    Preprocess a list of .nii.gz medical image files by computing median spacing/size
    and optionally resampling to a target size. Also saves the label dictionary.

    Args:
        nii_files (List[str]):
            A list of paths to .nii.gz image files.

        unique_identifiers (List[str]):
            A list of unique identifiers corresponding to each case.

        label_dict (Dict[str, int]):
            A mapping from class labels (as strings) to integers.

        target_img_size (Union[str, Tuple[int, int, int]]):
            Target image shape. Can be a tuple (e.g., (128, 128, 128))
            or the string 'median' to use median shape.

        num_worker (int):
            Number of worker processes for parallel computation.

        out_folder (str):
            Output directory to save processed data and the labels.json file.
    """
    print(f"Total .nii.gz files found: {len(nii_files)}")

    median_spacing, median_size, all_spacings, all_shapes = get_median_spacing_and_size_distributed(
        nii_files, num_workers=num_worker
    )

    print('median shape:', median_size)
    print('median spacing:', median_spacing)

    if type(target_img_size) == str:
        if target_img_size == 'median':
            run_all_cases(nii_files, median_size[::-1], out_folder, mode='shape', num_workers=num_worker)
    else:
        run_all_cases(nii_files, target_img_size, out_folder, mode='shape',num_workers=num_worker)

    with open(os.path.join(out_folder, 'labels.json'), "w") as f:
        json.dump(label_dict, f, indent=2)


    split_file = generate_crossval_split(unique_identifiers, n_splits=3)

    with open(os.path.join(out_folder, 'splits.json'), "w") as f:
        json.dump(split_file, f, indent=2)




def preprocess_dataset_tospacing(
        nii_files: list,
        unique_identifiers: list,
        label_dict: dict,
        target_spacing,  # Union[str, Tuple[int, int, int]] – left untyped for flexibility
        num_worker: int,
        out_folder: str
):
    """
    Preprocess a list of .nii.gz medical image files by computing median spacing/size
    and optionally resampling to a target size. Also saves the label dictionary.

    Args:
        nii_files (List[str]):
            A list of paths to .nii.gz image files.

        unique_identifiers (List[str]):
            A list of unique identifiers corresponding to each case.

        label_dict (Dict[str, int]):
            A mapping from class labels (as strings) to integers.

        target_img_size (Union[str, Tuple[int, int, int]]):
            Target image shape. Can be a tuple (e.g., (128, 128, 128))
            or the string 'median' to use median shape.

        num_worker (int):
            Number of worker processes for parallel computation.

        out_folder (str):
            Output directory to save processed data and the labels.json file.
    """
    print(f"Total .nii.gz files found: {len(nii_files)}")

    # median_spacing, median_size, all_spacings, all_shapes = get_median_spacing_and_size_distributed(
    #     nii_files, num_workers=num_worker
    # )
    #
    # print('median shape:', median_size)
    # print('median spacing:', median_spacing)


    run_all_cases(nii_files, target_spacing, out_folder, mode='spacing', num_workers=num_worker)

    with open(os.path.join(out_folder, 'labels.json'), "w") as f:
        json.dump(label_dict, f, indent=2)


    split_file = generate_crossval_split(unique_identifiers, n_splits=3)

    with open(os.path.join(out_folder, 'splits.json'), "w") as f:
        json.dump(split_file, f, indent=2)





