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
PATH_TCN            = "Training/Checkpoints/best_fold_model.ckpt"

IMG_SIZE_ROI        = 640
IMG_SIZE_POSE       = 1280

ROI_CONF_THRESHOLD  = 0.25
POSE_CONF_THRESHOLD = 0.25

MAX_OUTSIDE_RATIO   = 0.70
MIN_POSE_AREA       = 1200

ROI_PAD_Y           = 50

# ---------------- PIPELINE ---------------- #

class Pipeline:
    def __init__(self, classifier_path=None):
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"

        self.roi_detector   = ROIDetector(PATH_ROI_MODEL, imgsz=IMG_SIZE_ROI, conf=ROI_CONF_THRESHOLD)
        self.pose_estimator = PoseEstimator(PATH_POSE_MODEL, imgsz=IMG_SIZE_POSE, conf=POSE_CONF_THRESHOLD)
        self.pose_filter    = PoseFilter(max_outside_ratio=MAX_OUTSIDE_RATIO, min_area=MIN_POSE_AREA)

        self.classifier     = TCN.load_from_checkpoint(PATH_TCN, map_location=self.device,).eval()

        self.left_buffer    = SkeletonWindowBuffer("LEFT")
        self.right_buffer   = SkeletonWindowBuffer("RIGHT")

        self.label_map      = pd.read_csv(PATH_LABEL_MAP)
        self.id_to_label    = {row["id"]: row["label"] for _, row in self.label_map.iterrows()}

        self.roi = None  # persistent ROI

    def _maybe_classify(self, buffer, assigned, run_classification=True):
        t0 = time.perf_counter()

        fencer = buffer.fencer
        label = "SKIPPED" if not run_classification else "NO_ACTION"

        entry = assigned.get(fencer)
        kpts = entry["keypoints"] if entry else None
        conf = entry["confidence"] if entry else None
        topk = None

        if not run_classification:
            return label, kpts, conf, 0.0, topk

        buffer.add_frame(kpts)

        if buffer.is_ready():
            window = buffer.get_window().to(self.device)
            with torch.no_grad():
                logits = self.classifier(window)[0]
                if logits.dim() == 3:
                    logits = logits[0]          # [T, C]

                probs = torch.softmax(logits[-1], dim=-1)
                topk_probs, topk_ids = torch.topk(probs, k=3)

                topk = [
                    {
                        "label": self.id_to_label[idx.item()],
                        "confidence": prob.item()
                    }
                    for idx, prob in zip(topk_ids, topk_probs)
                ]

                label = topk[0]["label"]

        t_ms = (time.perf_counter() - t0) * 1000
        return label, kpts, conf, t_ms, topk

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
            left_label, left_kpts, left_conf, t_cls_l, topk_l = self._maybe_classify(
                self.left_buffer,
                assigned,
                run_classification=run_classification,
            )

            right_label, right_kpts, right_conf, t_cls_r, topk_r = self._maybe_classify(
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
                "topk_left": topk_l,

                "right_label": right_label,
                "right_confidence": right_conf,
                "right_keypoints": (
                    right_kpts.tolist() if isinstance(right_kpts, np.ndarray) else []
                ),
                "topk_right": topk_r,
            })

            frame_idx += 1

        cap.release()

        self.left_buffer.flush()
        self.right_buffer.flush()

        return pd.DataFrame(records)

    def step(self, frame, run_classification=True):
        t_frame = time.perf_counter()

        # -------- ROI -------- #
        t_roi = 0.0
        if self.roi is None:
            self.roi, t_roi = self.roi_detector.detect(frame)

        # -------- Pose -------- #
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.roi

        py1 = max(0, y1 - ROI_PAD_Y)
        py2 = min(h, y2 + ROI_PAD_Y)

        roi_crop = frame[py1:py2, x1:x2]
        poses, t_pose = self.pose_estimator.infer(roi_crop)

        for p in poses:
            p["keypoints"][:, 0] += x1
            p["keypoints"][:, 1] += py1
            if "bbox" in p:
                p["bbox"][0] += x1
                p["bbox"][1] += py1
                p["bbox"][2] += x1
                p["bbox"][3] += py1

        # -------- Filter -------- #
        assigned, t_filter = self.pose_filter.filter_and_assign(poses, self.roi)

        # -------- Classification -------- #
        left_label, left_kpts, left_conf, t_cls_l, topk_l = self._maybe_classify(
            self.left_buffer, assigned, run_classification
        )

        right_label, right_kpts, right_conf, t_cls_r, topk_r = self._maybe_classify(
            self.right_buffer, assigned, run_classification
        )

        t_total = (time.perf_counter() - t_frame) * 1000

        return {
            "roi": self.roi,
            "left": {
                "label": left_label,
                "kpts": left_kpts,
                "conf": left_conf,
            },
            "right": {
                "label": right_label,
                "kpts": right_kpts,
                "conf": right_conf,
            },
            "timing": {
                "total": t_total,
                "roi": t_roi,
                "pose": t_pose,
                "filter": t_filter,
                "classify": t_cls_l + t_cls_r,
            },
        }