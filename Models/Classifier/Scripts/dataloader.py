# dataset.py
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        self.unique_labels = sorted(df["action"].unique())

        self.label_to_id = {label: i for i, label in enumerate(self.unique_labels)}
        self.id_to_label = {i: l for l, i in self.label_to_id.items()}

        kpt_cols = [f"x{i}" for i in range(17)] + [f"y{i}" for i in range(17)]
        samples = []

        # Group 4 frames per window
        for wid, group in df.groupby("window_id"):
            group = group.sort_values("frame")

            # Convert keypoints to (4,17,2)
            kpts = group[kpt_cols].values      # (4,34)
            kpts = kpts.reshape(4, 17, 2)

            label_str = group["action"].iloc[0]
            label_id = self.label_to_id[label_str]

            samples.append((kpts.astype(np.float32), label_id))

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kpts, label = self.samples[idx]
        return (
            torch.from_numpy(kpts),                 # (4,17,2)
            torch.tensor(label, dtype=torch.long)
        )

class SkeletonDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, batch_size=32, num_workers=4):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = SkeletonDataset(self.csv_path)
        self.num_classes = len(dataset.unique_labels)
        self.label_dict = dataset.id_to_label

        N = len(dataset)
        val_size = int(0.1 * N)
        test_size = int(0.1 * N)
        train_size = N - val_size - test_size

        self.train_set, self.val_set, self.test_set = random_split(
            dataset, [train_size, val_size, test_size]
        )

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
            self.val_set,
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