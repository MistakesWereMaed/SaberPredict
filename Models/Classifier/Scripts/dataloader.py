import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, random_split

class SkeletonDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # All keypoint coordinate columns (x0..x16, y0..y16)
        keypoint_cols = [f"x{i}" for i in range(17)] + [f"y{i}" for i in range(17)]

        samples = []

        # Group by window_id → each sample is 4 frames
        for _, group in df.groupby("window_id"):
            group = group.sort_values("frame")

            # Extract 4 × 34 frame data
            keypoints = group[keypoint_cols].values  # shape (4, 34)

            # Reshape to (4, 17, 2)
            keypoints = keypoints.reshape(4, 17, 2)

            # Action label (all frames have same label)
            label = group["action"].iloc[0]

            samples.append((keypoints.astype(np.float32), int(label)))

        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data, label = self.samples[idx]
        # Convert to torch tensors
        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)
    
class SkeletonDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, batch_size=32, num_workers=4):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        full_dataset = SkeletonDataset(self.csv_path)

        N = len(full_dataset)
        val_size = int(0.1 * N)
        test_size = int(0.1 * N)
        train_size = N - val_size - test_size

        self.train_set, self.val_set, self.test_set = random_split(
            full_dataset,
            [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )