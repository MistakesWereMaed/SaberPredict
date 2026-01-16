import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

PATH_LABEL_MAP = "Dataset/Data/label_map.csv"
WINDOW_SIZE = 4
NUM_JOINTS = 17

class SkeletonDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        label_map = pd.read_csv(PATH_LABEL_MAP)

        self.label_to_id = {row["label"]: row["id"] for _, row in label_map.iterrows()}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        self.samples = []
        self._build_samples()

    def _build_samples(self):
        kpt_cols = [f"x{i}" for i in range(NUM_JOINTS)] + \
                   [f"y{i}" for i in range(NUM_JOINTS)]

        for wid, group in self.df.groupby("window_id"):
            group = group.sort_values("frame")

            # Enforce fixed window size
            if len(group) != WINDOW_SIZE:
                continue

            # --- Keypoints ---
            kpts = group[kpt_cols].values.astype(np.float32)
            kpts = np.nan_to_num(kpts, nan=0.0)
            kpts = kpts.reshape(WINDOW_SIZE, NUM_JOINTS, 2)

            # --- Confidence (frame-wise, scalar) ---
            conf = group["confidence"].values.astype(np.float32)
            conf = np.nan_to_num(conf, nan=0.0)
            conf = conf.reshape(WINDOW_SIZE, 1)

            # --- Label ---
            label_str = group["action"].iloc[0]
            label = self.label_to_id[label_str]

            self.samples.append((kpts, conf, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kpts, conf, label = self.samples[idx]

        return {
            "keypoints": torch.from_numpy(kpts),      # (T, 17, 2)
            "confidence": torch.from_numpy(conf),     # (T, 1)
            "label": torch.tensor(label, dtype=torch.long)
        }

class SkeletonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv,
        test_csv,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Build train first to define label space
        self.train_set = SkeletonDataset(self.train_csv)
        self.label_dict = self.train_set.id_to_label
        self.num_classes = len(self.label_dict)

        # Share label mapping across splits
        self.test_set = SkeletonDataset(self.test_csv)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
