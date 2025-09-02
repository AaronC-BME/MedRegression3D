import shutil

import pandas as pd
from datasets.preprocess_3D_data.crop_to_mask import load_image_np, crop_center_with_padding_np, get_mask_center
import glob
import numpy as np
from multiprocessing import Pool
from functools import partial
import os

from datasets.preprocess_3D_data.blosc_helper import save_case, comp_blosc2_params
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.utilities.file_and_folder_operations import load_json

from datasets.preprocess_3D_data.preprocess_dataset import preprocess_dataset_tospacing


def process_and_save_all_cases(
        image_dir,
        mask_dir,
        out_dir,
        target_shape=(160, 160, 160),
        num_workers=8
):
    os.makedirs(out_dir, exist_ok=True)

    t1_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    image_ids = [p.split('/')[-1][:8] for p in t1_paths]

    with Pool(processes=num_workers) as pool:
        pool.map(
            partial(
                process_single_case,
                image_dir=image_dir,
                mask_dir=mask_dir,
                out_dir=out_dir,
                crop_size=target_shape
            ),
            image_ids
        )

def process_single_case(image_id, image_dir, mask_dir, out_dir, crop_size):
    try:
        modalities = load_and_crop_modalities_by_mask_center(
            image_id=image_id,
            image_dir=image_dir,
            mask_dir=mask_dir,
            crop_size=crop_size
        )
        for modality_name, arr in modalities.items():
            expanded_arr = np.expand_dims(arr, axis=0)
            # print(arr.shape, crop_size, arr.itemsize, expanded_arr.shape)
            blocks, chunks = comp_blosc2_params(
                expanded_arr.shape, crop_size, arr.itemsize
            )

            out_path_truncated = os.path.join(
                out_dir, f"{image_id}_{modality_name}"
            )

            save_case(expanded_arr, out_path_truncated, chunks=chunks, blocks=blocks)

            print(f"✅ Saved cropped data for image {image_id}")
    except Exception as e:
        print(f"❌ Failed to process {image_id}: {e}")

def load_and_crop_modalities_by_mask_center(
        image_id,
        image_dir,
        mask_dir,
        crop_size=(160, 160, 160)
):
    """Load 4 modalities + mask, crop around mask center, return dict of arrays."""
    pseudo, _ = load_image_np(os.path.join(image_dir, f"{image_id}_MR_pseudo_ax_km.nii.gz"))
    postop, _ = load_image_np(os.path.join(image_dir, f"{image_id}_MR_postop_ax_km_reg.nii.gz"))
    mask, _ = load_image_np(os.path.join(mask_dir, f"{image_id}_MR_pseudo_ax_km_bet.nii.gz"))

    center = get_mask_center(mask)

    return {
        "MR_pseudo_ax_km": crop_center_with_padding_np(pseudo, center, crop_size),
        "MR_postop_ax_km_reg": crop_center_with_padding_np(postop, center, crop_size),
    }

def create_label_dict(path):
    label_dict = load_json(path)
    return label_dict

if __name__ == '__main__':
    # Base path and folders to search
    '''
    Steps to implement yourself: 
    1. get all nifti files (nii_files
    2. get all case identifier (unique ids)
    3. create a lable dict containing class labels 
    4. use preprocess_dataset_tospacing function to resample all images to 1mm spacing
    5. apply hdbet to all image to find brain center (https://github.com/MIC-DKFZ/HD-BET)
    6. copy labelsjson and splits.json file to preprocessed folder
    6. crop all images given the HDbet masks and save them as blosc  in preprocessed folder for training ( process_and_save_all_cases needs to be adapted for your dataset)
    
    
    '''
    dataset_base = '/bdm-das/ADSP_v1/nnssl_dataset/ukbb_dataset/nnUNetResEncUNetLPlans_3d_fullres'
    nii_files = glob.glob(os.path.join(dataset_base, '*.nii.gz'))
    unique_ids = [os.path.basename(f).split('_')[0] for f in nii_files]
    label_dict = load_json('/bdm-das/ADSP_v1/nnssl_dataset/ukbb_dataset/labelsTr.json') # {'unique_id1': 1, ...}



    ###############resampling to 1mm spacing - uncomment!######################
    preprocess_dataset_tospacing(nii_files, unique_ids, label_dict, [1.,1.,1.], out_folder='/bdm-das/ADSP_v1/nnssl_dataset/ukbb_dataset/preprocess_data/to_spacing',  num_worker=12)

    ###########use hdbet to find brain center ##########
    ###########hd-bet -i imagepath -o outpath --save_bet_mask --no_bet_image######

    ###########last step: copy files, crop and save images#####################
    image_dir = '/bdm-das/ADSP_v1/nnssl_dataset/ukbb_dataset/preprocess_data/to_spacing'
    mask_dir = "/bdm-das/ADSP_v1/nnssl_dataset/ukbb_dataset/preprocess_data/to_spacing/masks"
    out_dir = "/bdm-das/ADSP_v1/nnssl_dataset/ukbb_dataset/preprocess_data/to_spacing_cropped"
    maybe_mkdir_p(out_dir)
    shutil.copy(join(image_dir, 'labels.json'), join(out_dir, 'labels.json'))
    shutil.copy(join(image_dir, 'splits.json'), join(out_dir, 'splits.json'))
    
    process_and_save_all_cases(
        image_dir=image_dir,
        mask_dir=mask_dir,
        out_dir=out_dir,
        target_shape=(160, 160, 160), #we use a 160pix patchsize for the pre-training
        num_workers=12)  # Adjust for your system