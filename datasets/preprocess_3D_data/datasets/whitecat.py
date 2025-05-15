import os
import numpy as np
import pandas as pd
import json
from datasets.preprocess_3D_data.preprocess_dataset import preprocess_dataset


if __name__ == '__main__':
    # Base path and folders to search
    base_path = "/home/c306h/E132-Projekte/Projects/2022_Peretzke_Interactive_Fiber_Dissection/data/WhiteCat/WhiteCAT_FA_MD_consti/"  # Change this to your actual base directory

    # Collect matching files
    nii_files = []
    file_names = []

    for file in os.listdir(base_path):
        if file.endswith(".nii.gz"):
            nii_files.append(os.path.join(base_path, file))
            file_names.append(file[:-10])

    unique_ids = np.unique(file_names)
    print(f'Unique IDs: {unique_ids}', len(unique_ids))


    # generat label_json
    df = pd.read_excel("/home/c306h/E132-Projekte/Projects/2022_Peretzke_Interactive_Fiber_Dissection/data/WhiteCat/WhiteCAT_FA_MD_consti/whiteCAT.xlsx")  # Replace with your actual file name
    # Normalize Label_25 column: convert "TRUE" to 1, anything else to 0
    df['Label_25_normalized'] = df['Label_25'].apply(lambda x: 1 if str(x).strip().upper() == "TRUE" or x == 1 else 0)
    # Create dictionary with record_id as key and normalized Label_25 as value
    record_label_dict = dict(zip(df['record_id'], df['Label_25_normalized']))


    preprocess_dataset(nii_files, unique_ids, record_label_dict, 'median', out_folder='/home/c306h/cluster-data/classification/WhiteCat_median_shape',  num_worker=8,)
    preprocess_dataset(nii_files, unique_ids,record_label_dict, [128,256,256], out_folder='/home/c306h/cluster-data/classification/WhiteCat_1mmiso',  num_worker=8)
    preprocess_dataset(nii_files, unique_ids, record_label_dict, [160,160,160], out_folder='/home/c306h/cluster-data/classification/WhiteCat_fixed160patchsize',  num_worker=8)