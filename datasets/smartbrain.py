import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO


class SmartbrainData(Dataset):
    def __init__(self, root, split, fold, transform=None):
        super().__init__()
        """
        MRNet Dataset
        """
        try:
            self.img_dir = Path(root) / "nnsslPlans_onemmiso/Dataset002_SmartBRAINMRI/Dataset002_SmartBRAINMRI"
            label_file = Path(root) / "labelsTr.json"
            split_file = Path(root) / "splits_final.json"

            with open(split_file) as f:
                self.img_files = json.load(f)["train" if split == "train" else "val"]

            with open(label_file) as f:
                labels = json.load(f)
            self.img_files = [x for x in self.img_files if x in labels.keys()]
            #self.labels = np.array([labels[i] for i in self.img_files]).astype(np.int8)
            self.labels = torch.tensor([labels[i] for i in self.img_files], dtype=torch.long)


            self.transform = transform
        except:
            print(f'{self.img_dir=}')
            print(f'{label_file=}')
            print(f'{split_file=}')
            #print(f'{self.transform=}')

            raise ValueError(
                "Please check the dataset path and the split file. "
                "The dataset should be in the format of nnUNet."
            )

    def __getitem__(self, idx):
        pth = self.img_files[idx]

        img, _ = Blosc2IO.load(self.img_dir / (pth + ".b2nd"), mode="r")

        if self.transform:
            img = self.transform(**{"image": torch.from_numpy(img[...])})["image"]
        else:
            img = torch.from_numpy(img[...])

        return img, self.labels[idx]

    def __len__(self):
        return len(self.img_files)


class SmartbrainDataModule(BaseDataModule):
    def __init__(self, **params):
        super(SmartbrainDataModule, self).__init__(**params)

    def setup(self, stage: str):

        self.train_dataset = SmartbrainData(
            self.data_path,
            split="train",
            transform=self.train_transforms,
            fold=self.fold,
        )
        self.val_dataset = SmartbrainData(
            self.data_path,
            split="val",
            transform=self.test_transforms,
            fold=self.fold,
        )
