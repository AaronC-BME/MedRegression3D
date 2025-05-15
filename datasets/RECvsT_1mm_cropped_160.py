import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO


class RECvsT_1mm_cropped_160_Data(Dataset):
    def __init__(self, root, split, fold, transform=None):
        super().__init__()
        """
        GLvsL_median_shape Dataset
        """
        self.img_dir = Path(root) / "RECvsT_1mm_cropped_160"
        label_file = Path(root) / "RECvsT_1mm_cropped_160/labels.json"
        split_file = Path(root) / "RECvsT_1mm_cropped_160/splits.json"

        with open(split_file) as f:
            self.img_files = json.load(f)[fold]["train" if split == "train" else "val"]

        with open(label_file) as f:
            labels = json.load(f)
        self.labels = [labels[i] for i in self.img_files]

        self.transform = transform

    def __getitem__(self, idx):

        img1, _ = Blosc2IO.load(self.img_dir / (self.img_files[idx] + "_MR_postop_ax_km_reg.b2nd"), mode="r")
        img2, _ = Blosc2IO.load(self.img_dir / (self.img_files[idx] + "_MR_pseudo_ax_km.b2nd"), mode="r")
        img =  np.stack([img1[0], img2[0]], axis=0)

        if self.transform:
            img = self.transform(**{"image": torch.from_numpy(img[...])})["image"]
        else:
            img = torch.from_numpy(img[...])

        return img, self.labels[idx]

    def __len__(self):
        return len(self.img_files)


class RECvsT_1mm_cropped_160_DataModule(BaseDataModule):
    def __init__(self, **params):
        super(RECvsT_1mm_cropped_160_DataModule, self).__init__(**params)

    def setup(self, stage: str):

        self.train_dataset = RECvsT_1mm_cropped_160_Data(
            self.data_path,
            split="train",
            transform=self.train_transforms,
            fold=self.fold,
        )
        self.val_dataset = RECvsT_1mm_cropped_160_Data(
            self.data_path,
            split="val",
            transform=self.test_transforms,
            fold=self.fold,
        )