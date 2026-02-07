import ast
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

PATH_LABEL_MAP = "Dataset/Data/label_map.csv"

SEQ_LEN = 10
NUM_JOINTS = 17

class SkeletonSequenceDataset(Dataset):
    def __init__(self, frame_csv, sequence_csv):
        self.df = pd.read_csv(frame_csv)
        self.seq_df = pd.read_csv(sequence_csv)

        self.label_map = pd.read_csv(PATH_LABEL_MAP)
        self.label_to_id = {
            row["label"]: row["id"] for _, row in self.label_map.iterrows()
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        expected_x = [f"x{i}" for i in range(NUM_JOINTS)]
        expected_y = [f"y{i}" for i in range(NUM_JOINTS)]
        expected_cols = set(expected_x + expected_y)

        if not expected_cols.issubset(self.df.columns):
            if "keypoints" not in self.df.columns:
                raise ValueError(
                    "CSV must contain either x*/y* columns or a 'keypoints' column"
                )

            self.df["keypoints"] = self.df["keypoints"].apply(
                lambda x: [tuple(p) for p in ast.literal_eval(x)] if isinstance(x, str) else x
            )
            self.df = self._expand_keypoints_column(self.df)

        self.kpt_cols = expected_x + expected_y

        # Pre-group frames for fast lookup
        self.groups = {
            k: g.sort_values("frame").reset_index(drop=True)
            for k, g in self.df.groupby(["file", "fencer"])
        }

        self.kpt_cols = (
            [f"x{i}" for i in range(NUM_JOINTS)] +
            [f"y{i}" for i in range(NUM_JOINTS)]
        )

    def __len__(self):
        return len(self.seq_df)

    def __getitem__(self, idx):
        row = self.seq_df.iloc[idx]

        g = self.groups[(row.file, row.fencer)]

        start = row.start_idx
        end = start + SEQ_LEN

        window = g.iloc[start:end]

        # --- Keypoints ---
        kpts = window[self.kpt_cols].values.astype(np.float32)
        kpts = np.nan_to_num(kpts, nan=0.0)
        kpts = kpts.reshape(SEQ_LEN, NUM_JOINTS, 2)

        # --- Labels (per-frame) ---
        labels = window["action"].map(self.label_to_id).values

        return {
            "keypoints": torch.from_numpy(kpts),          # (T, 17, 2)
            "labels": torch.tensor(labels, dtype=torch.long)  # (T,)
        }
    
    def _expand_keypoints_column(self, df):
        """
        Expands a serialized 'keypoints' column into x0..y16 columns.
        """

        def parse_kpts(val):
            arr = np.asarray(val, dtype=np.float32)

            # Shape handling
            if arr.shape == (0,):
                return np.zeros((NUM_JOINTS, 2), dtype=np.float32)
            if arr.shape == (NUM_JOINTS, 2):
                pass
            elif arr.shape == (NUM_JOINTS * 2,):
                arr = arr.reshape(NUM_JOINTS, 2)
            else:
                raise ValueError(f"Invalid keypoint shape: {arr.shape}")

            return arr

        kpts = df["keypoints"].apply(parse_kpts)

        xs = np.stack(kpts.apply(lambda a: a[:, 0]))
        ys = np.stack(kpts.apply(lambda a: a[:, 1]))

        for i in range(NUM_JOINTS):
            df[f"x{i}"] = xs[:, i]
            df[f"y{i}"] = ys[:, i]

        return df.drop(columns=["keypoints"])