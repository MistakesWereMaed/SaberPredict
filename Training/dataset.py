import torch
import pandas as pd
import numpy as np

from collections import Counter
from torch.utils.data import Dataset

PATH_LABEL_MAP = "Dataset/Data/label_map.csv"

NUM_JOINTS = 17
WINDOW_SIZE = 4

class SkeletonDataset(Dataset):
    def __init__(self, df):
        self.df = df
        label_map = pd.read_csv(PATH_LABEL_MAP)

        self.label_to_id = {
            row["label"]: row["id"] for _, row in label_map.iterrows()
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        self.samples = []
        self._build_samples()

    def _build_samples(self):
        kpt_cols = (
            [f"x{i}" for i in range(NUM_JOINTS)] +
            [f"y{i}" for i in range(NUM_JOINTS)]
        )

        self.labels = []

        for wid, group in self.df.groupby("window_id"):
            group = group.sort_values("frame")

            # Enforce fixed window size
            if len(group) != WINDOW_SIZE:
                continue

            # --- Keypoints (T, 17, 2) ---
            kpts = group[kpt_cols].values.astype(np.float32)
            kpts = np.nan_to_num(kpts, nan=0.0)
            kpts = kpts.reshape(WINDOW_SIZE, NUM_JOINTS, 2)

            # --- Temporal Derivatives ---
            deltas = np.zeros_like(kpts, dtype=np.float32)
            deltas[1:] = kpts[1:] - kpts[:-1]

            # --- Concatenate (T, 17, 4) ---
            kpts_aug = np.concatenate([kpts, deltas], axis=-1)

            # --- Label ---
            label_str = group["action"].iloc[0]
            label = self.label_to_id[label_str]

            self.samples.append((kpts_aug, label))
            self.labels.append(label)

        counts = Counter(self.labels)
        total = sum(counts.values())

        self.class_weights = {
            cls: total / count
            for cls, count in counts.items()
        }

        weights = np.array([self.class_weights[label] for label in self.labels])
        self.sampler_weights = weights / weights.sum()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kpts, label = self.samples[idx]

        return {
            "keypoints": torch.from_numpy(kpts),   # (T, 17, 4)
            "label": torch.tensor(label, dtype=torch.long),
        }