import time
import cv2
import numpy as np

from rtmlib import RTMPose

PATH_MODEL = "Training/Checkpoints/RTMPose-x.onnx"

class PoseEstimator:
    def __init__(self, model_path=None, conf=0.5, device="auto", imgsz=1280):
        self.conf = conf

        # device selection
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except:
                device = "cpu"

        self.model = RTMPose(
            onnx_model=PATH_MODEL,
            model_input_size=(288, 384),
            backend="onnxruntime",
            device=device
        )

    def infer(self, frame):
        t0 = time.perf_counter()

        frame_resized = cv2.resize(frame, (384, 288))
        keypoints, scores = self.model(frame_resized)

        poses = []

        if keypoints is None or len(keypoints) == 0:
            return poses, (time.perf_counter() - t0) * 1000

        for kpts, kpt_scores in zip(keypoints, scores):
            kpts = np.array(kpts)           # (K, 2)
            kpt_scores = np.array(kpt_scores)  # (K,)

            # filter low-confidence keypoints
            valid = kpt_scores > 0.2
            if valid.any():
                mean_conf = float(kpt_scores[valid].mean())
            else:
                mean_conf = 0.0

            if mean_conf < self.conf:
                continue

            poses.append({
                "keypoints": kpts,
                "confidence": mean_conf,
            })

        return poses, (time.perf_counter() - t0) * 1000