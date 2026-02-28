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

        left_x_cols  = [f"xl{i}" for i in range(NUM_JOINTS)]
        left_y_cols  = [f"yl{i}" for i in range(NUM_JOINTS)]
        right_x_cols = [f"xr{i}" for i in range(NUM_JOINTS)]
        right_y_cols = [f"yr{i}" for i in range(NUM_JOINTS)]

        self.labels = []

        # If window_id exists use that, otherwise slide manually
        if "window_id" in self.df.columns:
            groups = self.df.groupby("window_id")
        else:
            # fallback: sliding windows per file
            groups = []
            for file_name, file_group in self.df.groupby("file"):
                file_group = file_group.sort_values("frame").reset_index(drop=True)
                for i in range(len(file_group) - WINDOW_SIZE + 1):
                    groups.append((None, file_group.iloc[i:i+WINDOW_SIZE]))

        for _, group in groups:

            group = group.sort_values("frame")

            if len(group) != WINDOW_SIZE:
                continue

            frames = []

            for _, row in group.iterrows():

                # ---- LEFT ----
                left_kpts = np.stack([
                    row[left_x_cols].values,
                    row[left_y_cols].values
                ], axis=-1).astype(np.float32)

                # ---- RIGHT ----
                right_kpts = np.stack([
                    row[right_x_cols].values,
                    row[right_y_cols].values
                ], axis=-1).astype(np.float32)

                left_kpts = np.nan_to_num(left_kpts, nan=0.0)
                right_kpts = np.nan_to_num(right_kpts, nan=0.0)

                # frame shape: (2, 17, 2)
                frame_kpts = np.stack([left_kpts, right_kpts], axis=0)

                frames.append(frame_kpts)

            # (T, 2, 17, 2)
            kpts = np.array(frames, dtype=np.float32)

            # ---- Temporal Derivatives ----
            deltas = np.zeros_like(kpts)
            deltas[1:] = kpts[1:] - kpts[:-1]

            # ---- Concatenate (T, 2, 17, 4) ----
            kpts_aug = np.concatenate([kpts, deltas], axis=-1)

            # ---- Labels (window-level) ----
            left_label_str  = group["left_action"].iloc[0]
            right_label_str = group["right_action"].iloc[0]

            left_label  = self.label_to_id[left_label_str]
            right_label = self.label_to_id[right_label_str]

            labels = np.array([left_label, right_label], dtype=np.int64)

            self.samples.append((kpts_aug, labels))
            self.labels.extend(labels.tolist())

        # ---- Class weights (across both fencers) ----
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
        kpts, labels = self.samples[idx]

        return {
            "keypoints": torch.from_numpy(kpts),   # (T, 2, 17, 4)
            "label": torch.tensor(labels, dtype=torch.long),  # (2,)
        }