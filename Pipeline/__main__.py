import cv2
import time
import torch
import pandas as pd
import numpy as np

from collections import deque
from Pipeline.Models.classifier import TCN
from Pipeline.Models.pose_estimator import PoseEstimator

CHECKPOINT_PATH = "Pipeline/Models/Checkpoints/TCN-best.ckpt"
MAP_PATH        = "Dataset/Data/label_map.csv"

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

NUM_JOINTS      = 17
WINDOW_SIZE     = 4

class SkeletonWindowBuffer:
    def __init__(self, fencer, window_size=WINDOW_SIZE, num_joints=NUM_JOINTS):
        self.fencer = fencer
        self.window_size = window_size
        self.num_joints = num_joints
        self.buffer = deque(maxlen=window_size)

    def add_frame(self, keypoints):
        """
        keypoints: np.ndarray (17, 2) or None
        """
        if keypoints is None:
            # Strictly match training-time shape
            keypoints = np.zeros((self.num_joints, 2), dtype=np.float32)

        self.buffer.append(keypoints.astype(np.float32))

    def is_ready(self):
        return len(self.buffer) == self.window_size

    def get_window(self):
        """
        Returns shape (1, T, 17, 2)
        """
        assert self.is_ready()
        window = np.stack(self.buffer, axis=0)  # (8, 17, 2)
        return torch.from_numpy(window).unsqueeze(0)

class Pipeline():
    def __init__(self):
        self.pose_estimator     = PoseEstimator()
        self.classifier         = TCN.load_from_checkpoint(CHECKPOINT_PATH, map_location=DEVICE)

        self.left_buffer        = SkeletonWindowBuffer("LEFT")
        self.right_buffer       = SkeletonWindowBuffer("RIGHT")
        
        label_map = pd.read_csv(MAP_PATH)
        
        self.id_to_label = {row["id"]: row["label"] for _, row in label_map.iterrows()}
        self.classifier.eval()

    def _process_pose(self, buffer, output):
        fencer  = buffer.fencer
        label   = "NO_ACTION"

        kpts    = output[fencer]["keypoints"]
        
        buffer.add_frame(kpts)
        if buffer.is_ready():
            window = buffer.get_window().to(DEVICE)
            with torch.no_grad():
                logits = self.classifier(window)[0]
                pred_id = torch.argmax(logits, dim=1).item()
                label = self.id_to_label[pred_id]

        return [label, kpts]

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_idx = 0
        records = []

        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            output = self.pose_estimator.process_frame(frame, frame_idx)
            left   = self._process_pose(self.left_buffer, output)
            right  = self._process_pose(self.right_buffer, output)

            t1 = time.perf_counter()
            time_ms = t1 - t0

            records.append((frame_idx, time_ms, *left, *right, output["roi"]))
            frame_idx += 1

        cap.release()
        return records

    def unpack(self, records):
        rows = []
        for record in records:
            frame_idx, time, left_label, left_kpts, right_label, right_kpts, roi = record

            left_kpts = left_kpts.tolist() if isinstance(left_kpts, np.ndarray) else []
            right_kpts = right_kpts.tolist() if isinstance(right_kpts, np.ndarray) else []
            roi = roi.tolist() if isinstance(roi, np.ndarray) else []

            rows.append({
                "frame_idx":        frame_idx,
                "time":             time,
                "roi":              roi,

                "left_label":       left_label,
                "left_keypoints":   left_kpts,

                "right_label":      right_label,
                "right_keypoints":  right_kpts,
            })

        return pd.DataFrame(rows)

def main():
    video_path      = "Dataset/Videos/Clips/5/1_Left.mp4"
    output_path     = "Dataset/Data/test_out.csv"

    pipeline = Pipeline()

    results = pipeline.run(video_path)
    df      = pipeline.unpack(results)

    df.to_csv(output_path, index=False)

    df_times = df[["time"]]

    summary = df_times.agg(["mean", "median", "max", lambda x: x.quantile(0.95)])
    summary.index = ["mean", "median", "max", "p95"]

    print("\n--- Timing Summary (ms) ---")
    print(summary * 1000)

if __name__ == "__main__":
    main()