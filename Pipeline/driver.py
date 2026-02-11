import os
import re
import cv2
import time
import torch
import pandas as pd
import numpy as np

from Pipeline.roi_detector import ROIDetector
from Pipeline.pose_estimator import PoseEstimator
from Pipeline.pose_filter import PoseFilter
from Pipeline.classifier import TCN
from Pipeline.buffer import SkeletonWindowBuffer

# ---------------- CONFIG ---------------- #
PATH_LABEL_MAP      = "Dataset/Data/label_map.csv"

PATH_ROI_MODEL      = "Training/Checkpoints/xlarge.pt"
PATH_POSE_MODEL     = "Training/Checkpoints/yolo11x-pose.pt"
PATH_TCN            = "Training/Checkpoints/"

IMG_SIZE_ROI        = 640
IMG_SIZE_POSE       = 1280

ROI_CONF_THRESHOLD  = 0.25
POSE_CONF_THRESHOLD = 0.25

MAX_OUTSIDE_RATIO   = 0.70
MIN_POSE_AREA       = 1200

ROI_PAD_Y           = 50

def get_best_checkpoint(checkpoints_dir):
    """
    Returns the checkpoint file with the highest val_acc in the given directory.

    Expects filenames like: 'TCN-epoch=25-val_acc=0.86.ckpt'
    """
    best_file = None
    best_acc = -1.0

    pattern = re.compile(r"val_acc=([0-9]+)")

    for fname in os.listdir(checkpoints_dir):
        if not fname.endswith(".ckpt"):
            continue
        match = pattern.search(fname)
        if match:
            acc = float(match.group(1))
            if acc > best_acc:
                best_acc = acc
                best_file = fname

    if best_file is None:
        raise FileNotFoundError(f"No valid checkpoint found in {checkpoints_dir}")

    return os.path.join(checkpoints_dir, best_file)

# ---------------- PIPELINE ---------------- #

class Pipeline:
    def __init__(self):
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"

        self.roi_detector   = ROIDetector(PATH_ROI_MODEL, imgsz=IMG_SIZE_ROI, conf=ROI_CONF_THRESHOLD)
        self.pose_estimator = PoseEstimator(PATH_POSE_MODEL, imgsz=IMG_SIZE_POSE, conf=POSE_CONF_THRESHOLD)
        self.pose_filter    = PoseFilter(max_outside_ratio=MAX_OUTSIDE_RATIO, min_area=MIN_POSE_AREA)

        checkpoint          = get_best_checkpoint(PATH_TCN)
        self.classifier     = TCN.load_from_checkpoint(checkpoint, map_location=self.device,).eval()

        self.left_buffer    = SkeletonWindowBuffer("LEFT")
        self.right_buffer   = SkeletonWindowBuffer("RIGHT")

        self.label_map = pd.read_csv(PATH_LABEL_MAP)
        self.id_to_label = {row["id"]: row["label"] for _, row in self.label_map.iterrows()}

        self.roi = None  # persistent ROI

    def _maybe_classify(self, buffer, assigned, run_classification=True):
        t0 = time.perf_counter()

        fencer = buffer.fencer
        label = "SKIPPED" if not run_classification else "OTHER_NO_ACTION"

        entry = assigned.get(fencer)
        kpts = entry["keypoints"] if entry else None
        conf = entry["confidence"] if entry else None

        if not run_classification:
            return label, kpts, conf, 0.0

        buffer.add_frame(kpts)

        if buffer.is_ready():
            window = buffer.get_window().to(self.device)
            with torch.no_grad():
                logits = self.classifier(window)[0]
                if logits.dim() == 3:
                    logits = logits[0]          # [T, C]

                pred_id = torch.argmax(logits[-1], dim=-1).item()
                label = self.id_to_label[pred_id]

        t_ms = (time.perf_counter() - t0) * 1000
        return label, kpts, conf, t_ms

    def run(self, video_path, run_classification=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        records = []
        frame_idx = 0

        while True:
            t_frame = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            # -------- ROI (first frame only) -------- #
            t_roi = 0.0
            if self.roi is None:
                self.roi, t_roi = self.roi_detector.detect(frame)

           # -------- Pose Estimation (ROI-cropped) -------- #
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = self.roi

            # padded ROI (clamped)
            py1 = max(0, y1 - ROI_PAD_Y)
            py2 = min(h, y2 + ROI_PAD_Y)

            roi_crop = frame[py1:py2, x1:x2]

            poses, t_pose = self.pose_estimator.infer(roi_crop)

            # remap keypoints to full-frame coordinates
            for p in poses:
                p["keypoints"][:, 0] += x1
                p["keypoints"][:, 1] += py1
                if "bbox" in p:
                    p["bbox"][0] += x1
                    p["bbox"][1] += py1
                    p["bbox"][2] += x1
                    p["bbox"][3] += py1

            # -------- Pose Filtering -------- #
            assigned, t_filter = self.pose_filter.filter_and_assign(
                poses,
                self.roi,
            )

            # -------- Classification -------- #
            left_label, left_kpts, left_conf, t_cls_l = self._maybe_classify(
                self.left_buffer,
                assigned,
                run_classification=run_classification,
            )

            right_label, right_kpts, right_conf, t_cls_r = self._maybe_classify(
                self.right_buffer,
                assigned,
                run_classification=run_classification,
            )

            t_total = (time.perf_counter() - t_frame) * 1000

            records.append({
                "frame_idx": frame_idx,

                "time_total_ms": t_total,
                "time_roi_ms": t_roi,
                "time_pose_ms": t_pose,
                "time_filter_ms": t_filter,
                "time_classify_ms": t_cls_l + t_cls_r,

                "roi": self.roi.tolist() if self.roi is not None else [],

                "left_label": left_label,
                "left_confidence": left_conf,
                "left_keypoints": (
                    left_kpts.tolist() if isinstance(left_kpts, np.ndarray) else []
                ),

                "right_label": right_label,
                "right_confidence": right_conf,
                "right_keypoints": (
                    right_kpts.tolist() if isinstance(right_kpts, np.ndarray) else []
                ),
            })

            frame_idx += 1

        cap.release()
        return pd.DataFrame(records)