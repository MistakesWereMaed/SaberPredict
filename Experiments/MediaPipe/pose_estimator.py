import time
import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    def __init__(self, model_path=None, model_complexity=2, conf=0.5, device='cpu', imgsz=1280):
        """
        MediaPipe-based Pose Estimator

        Args:
            model_complexity (int): 0 (light), 1 (medium), 2 (heavy) model backbone.
            conf (float): Minimum average confidence for keypoints to accept a pose.
            device (str): Placeholder to match other PoseEstimator APIs ('cpu'/'cuda').
        """
        self.conf = conf
        self.mp_pose = mp.solutions.pose
        # MediaPipe runs on CPU by default. GPU requires custom build.
        self.pose_model = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

    def infer(self, frame):
        """
        Run pose estimation on a single frame.

        Args:
            frame (np.ndarray): BGR image (H x W x 3)

        Returns:
            poses (list): List of dicts with keys "keypoints" and "confidence"
            inference_time_ms (float): Time in milliseconds for inference
        """
        t0 = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_model.process(frame_rgb)

        poses = []
        if results.pose_landmarks:
            keypoints = []
            confidences = []
            h, w, _ = frame.shape
            for lm in results.pose_landmarks.landmark:
                keypoints.append([lm.x * w, lm.y * h])
                confidences.append(lm.visibility)
            keypoints = np.array(keypoints)
            confidences = np.array(confidences)
            mean_conf = float(confidences.mean())
            if mean_conf >= self.conf:
                poses.append({"keypoints": keypoints, "confidence": mean_conf})

        inference_time_ms = (time.perf_counter() - t0) * 1000
        return poses, inference_time_ms