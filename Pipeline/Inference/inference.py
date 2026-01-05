import cv2
import time
import numpy as np
import pandas as pd
import torch

from collections import deque


from pose_estimator import PoseEstimator
from TCN import model as TCN

# ----------------------------
# CONFIGURATION
# ----------------------------

VIDEO_PATH = "../Dataset/Videos/Clips/1/11_Left.mp4"
CHECKPOINT_PATH = "../Models/Checkpoints/TCN-v2.ckpt"
MAP_PATH = "label_map.csv"
OUTPUT_PATH = "test_out.csv"

ROI = (0, 600, 1900, 850)
IMG_SIZE = 1280
PAD = 20

WINDOW_SIZE = 8
NUM_JOINTS = 17

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# BUFFER CLASS
# ----------------------------

class SkeletonWindowBuffer:
    def __init__(self, fencer, window_size=8, num_joints=17):
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

# ----------------------------
# MAIN INFERENCE LOOP
# ----------------------------

def process_pose(buffer, classifier, id_to_label, output):
    fencer = buffer.fencer
    kpts = output[fencer]["keypoints"]
    conf = output[fencer]["confidence"]
    box = output[fencer]["box"]
    label = "NO_ACTION"
    
    buffer.add_frame(kpts)
    if buffer.is_ready():
        window = buffer.get_window().to(DEVICE)
        with torch.no_grad():
            logits = classifier(window)[0]
            pred_id = torch.argmax(logits, dim=1).item()
            label = id_to_label[pred_id]

    return kpts, conf, box, label

def main():
    label_map = pd.read_csv(MAP_PATH)
    id_to_label = {row["id"]: row["label"] for _, row in label_map.iterrows()}

    # Load models
    pose_estimator = PoseEstimator(roi=ROI, pad=PAD, imgsz=IMG_SIZE)
    classifier = TCN.load_from_checkpoint(CHECKPOINT_PATH, map_location=DEVICE)
    classifier.eval()

    # Buffers
    left_buffer = SkeletonWindowBuffer("LEFT")
    right_buffer = SkeletonWindowBuffer("RIGHT")

    # Video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    frame_idx = 0
    timings = []
    records = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # Process frame
        output = pose_estimator.process_frame(frame, frame_idx)
        left_kpts, left_conf, left_box, left_label     = process_pose(left_buffer, classifier, id_to_label, output)
        right_kpts, right_conf, right_box, right_label = process_pose(right_buffer, classifier, id_to_label, output)

        t1 = time.perf_counter()
        timings.append(t1 - t0)

        # Record results
        for fencer, kpts, conf, box, label in zip(
            ["LEFT","RIGHT"],
            [left_kpts, right_kpts],
            [left_conf, right_conf],
            [left_box, right_box],
            [left_label, right_label]
        ):
            records.append({
                "frame": frame_idx,
                "fencer": fencer,
                "pred_label": label,
                "inference_time_s": t1 - t0,
                "box": box if box is not None else None,
                "keypoints": kpts.tolist() if kpts is not None else None,
                "confidence": float(conf) if conf is not None else None
            })

        frame_idx += 1

    cap.release()

    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved results to {OUTPUT_PATH}")

    timings = np.array(timings)
    print("\n--- Timing Summary ---")
    print(f"Frames processed: {len(timings)}")
    print(f"Mean time/frame: {timings.mean()*1000:.2f} ms")
    print(f"Median time/frame: {np.median(timings)*1000:.2f} ms")
    print(f"95th percentile: {np.percentile(timings,95)*1000:.2f} ms")

if __name__ == "__main__":
    main()