import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO
import os


class AgeReg_Data(Dataset):
    def __init__(self, root, split, fold, transform=None, train=True):
        super().__init__()
        """
        GLvsL_median_shape Dataset
        """
        self.img_dir = Path(root) / "nnsslPlans_onemmiso/Dataset001_UKBB/Dataset001_UKBB"
        label_file = Path(root) / "labelsTr.json"
        split_file = Path(root) / "splits.json"

        with open(split_file) as f:
            splits = json.load(f)
            if split in splits:
                self.img_files = splits[split]
            else:
                raise ValueError(f"Unknown split name: {split}")

        with open(label_file) as f:
            labels = json.load(f)
        #self.img_files = [x for x in self.img_files if x in labels.keys()]
        #self.labels = np.array([labels[i] for i in self.img_files]).astype(np.int8)
        self.labels = torch.tensor([labels[i] for i in self.img_files], dtype=torch.float)

        self.transform = transform
        self.train = train


    def __getitem__(self, idx):

        img, _ = Blosc2IO.load(os.path.join(self.img_dir, self.img_files[idx], 'ses-DEFAULT', self.img_files[idx] + ".b2nd"), mode="r")

        if self.train: # self.transform:
            img = self.transform(**{"image": torch.from_numpy(img[...])})["image"]
            #print(f"train {self.img_files[idx]}: {img.shape}")
        else:
            img = self.transform.transforms[0](**{"image": torch.from_numpy(img[...])})["image"]
            #img = torch.from_numpy(img[...])
            #print(f"val {self.img_files[idx]}: {img.shape}")

        return img, self.labels[idx]

    def __len__(self):
        return len(self.img_files)


class AgeReg_DataModule(BaseDataModule):
    def __init__(self, **params):
        super(AgeReg_DataModule, self).__init__(**params)

    def setup(self, stage: str):

        self.train_dataset = AgeReg_Data(
            self.data_path,
            split="train",
            transform=self.train_transforms,
            fold=self.fold,
        )
        self.val_dataset = AgeReg_Data(
            self.data_path,
            split="val",
            transform=self.test_transforms,
            fold=self.fold,
            train=False,
        )
        self.test_dataset = AgeReg_Data(
            self.data_path,
            split="test",
            transform=self.test_transforms,
            fold=self.fold,
            train=False,
        )