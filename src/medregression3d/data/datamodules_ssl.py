import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .base_datamodule import BaseDataModule
from medregression3d.utils.io import Blosc2IO


class AgeReg_Data(Dataset):
    def __init__(self, root, split, fold, transform=None, train=True):
        super().__init__()
        """
        Age regression dataset.
        Expects splits.json structured as {fold_name: {split_name: [ids]}}.
        """
        self.img_dir = Path(root) / "nnsslPlans_onemmiso/Dataset002_LiverROI_SSL3D_preprocessing/Dataset002_LiverROI_SSL3D_preprocessing"
        label_file = Path(root) / "labels.json"
        split_file = Path(root) / "splits.json"

        with open(split_file) as f:
            splits = json.load(f)

        # Resolve fold key — accept either "fold_0" or just 0
        # Resolve fold key — accept "fold_0", "0", or 0
        fold_str = str(fold)
        if fold_str in splits:
            fold_key = fold_str
        elif f"fold_{fold_str}" in splits:
            fold_key = f"fold_{fold_str}"
        else:
            fold_key = fold_str  # let the check below raise a clean error
            
        if fold_key not in splits:
            raise ValueError(
                f"Unknown fold: {fold_key}. Available folds: {list(splits.keys())}"
            )
        if split not in splits[fold_key]:
            raise ValueError(
                f"Unknown split '{split}' in fold {fold_key}. "
                f"Available splits: {list(splits[fold_key].keys())}"
            )

        self.img_files = splits[fold_key][split]

        with open(label_file) as f:
            labels = json.load(f)

        # Sanity check: every file in this split has a label
        missing = [f for f in self.img_files if f not in labels]
        if missing:
            raise ValueError(
                f"{len(missing)} entries in split '{split}'/fold '{fold_key}' "
                f"have no label. First few: {missing[:5]}"
            )

        self.labels = torch.tensor(
            [labels[i] for i in self.img_files], dtype=torch.float
        )
        self.transform = transform
        self.train = train

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir,
            self.img_files[idx],
            "ses-DEFAULT",
            self.img_files[idx] + ".b2nd",
        )
        img, _ = Blosc2IO.load(img_path, mode="r")

        if self.train:
            img = self.transform(**{"image": torch.from_numpy(img[...])})["image"]
        else:
            img = self.transform.transforms[0](
                **{"image": torch.from_numpy(img[...])}
            )["image"]

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