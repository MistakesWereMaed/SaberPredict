import numpy as np

from ultralytics import YOLO

PATH_ROI_MODEL      = "Pipeline/Models/Checkpoints/small.pt"
PATH_POSE_MODEL     = "Pipeline/Models/Checkpoints/yolo11x-pose.pt"

IMG_SIZE_ROI        = 640
IMG_SIZE_POSE       = 1280

ROI_CONF_THRESHOLD  = 0.25
POSE_CONF_THRESHOLD = 0.25

MIN_POSE_AREA       = 400  # pixels^2, tune based on resolution / distance

class PoseEstimator:
    def __init__(
        self,
        imgsz_roi=IMG_SIZE_ROI,
        imgsz_pose=IMG_SIZE_POSE,
        roi_conf=ROI_CONF_THRESHOLD,
        pose_conf=POSE_CONF_THRESHOLD,
    ):
        self.imgsz_roi  = imgsz_roi
        self.imgsz_pose = imgsz_pose
        self.roi_conf   = roi_conf
        self.pose_conf  = pose_conf

        self.roi_model  = YOLO(PATH_ROI_MODEL, task="detect")
        self.pose_model = YOLO(PATH_POSE_MODEL, task="pose")

        self.previous_skeletons = []

    # ---------------- ROI ---------------- #

    def _detect_roi(self, frame):
        res = self.roi_model(frame, imgsz=self.imgsz_roi, conf=self.roi_conf, verbose=False)

        boxes = []
        for r in res:
            if r.boxes is None:
                continue
            for box in r.boxes.xyxy.cpu().numpy():
                boxes.append(box)

        if not boxes:
            return None

        # Take largest ROI (strip)
        boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        return boxes[0].astype(int)
    
    def _pose_bbox(self, keypoints):
        """Compute (x1,y1,x2,y2,area) from keypoints"""
        if keypoints is None or len(keypoints) == 0:
            return None

        x = keypoints[:, 0]
        y = keypoints[:, 1]

        x1, x2 = np.min(x), np.max(x)
        y1, y2 = np.min(y), np.max(y)

        area = (x2 - x1) * (y2 - y1)
        return x1, y1, x2, y2, area

    # ---------------- Motion ---------------- #

    def process_frame(self, frame, frame_idx):
        roi = self._detect_roi(frame)

        if roi is None:
            return {
                "frame_idx": frame_idx,
                "LEFT":  {"keypoints": None, "confidence": None},
                "RIGHT": {"keypoints": None, "confidence": None},
                "roi": None
            }

        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]

        res = self.pose_model(crop, imgsz=self.imgsz_pose, conf=self.pose_conf, verbose=False)

        poses = []
        for r in res:
            if r.keypoints is None:
                continue

            kpts_xy = r.keypoints.xy.cpu().numpy()
            kpts_conf = (
                r.keypoints.conf.cpu().numpy()
                if hasattr(r.keypoints, "conf")
                else np.ones(len(kpts_xy))
            )

            for i, kpts in enumerate(kpts_xy):
                k = kpts.copy()
                k[:, 0] += x1
                k[:, 1] += y1

                bbox = self._pose_bbox(k)
                if bbox is None:
                    continue

                bx1, by1, bx2, by2, area = bbox
                if area < MIN_POSE_AREA:
                    continue  # drop small detections

                poses.append({
                    "keypoints": k,
                    "confidence": float(np.nanmean(kpts_conf[i])),
                    "area": area,
                    "cx": (bx1 + bx2) / 2,
                })

        if not poses:
            return {
                "frame_idx": frame_idx,
                "LEFT":  {"keypoints": None, "confidence": None},
                "RIGHT": {"keypoints": None, "confidence": None},
                "roi": roi
            }

        # Keep 2 largest poses
        poses.sort(key=lambda p: p["area"], reverse=True)
        poses = poses[:2]

        # Assign LEFT / RIGHT by x-position
        poses.sort(key=lambda p: p["cx"])

        left  = poses[0] if len(poses) > 0 else None
        right = poses[1] if len(poses) > 1 else None

        return {
            "frame_idx": frame_idx,
            "LEFT": {
                "keypoints":  left["keypoints"] if left else None,
                "confidence": left["confidence"] if left else None,
            },
            "RIGHT": {
                "keypoints":  right["keypoints"] if right else None,
                "confidence": right["confidence"] if right else None,
            },
            "roi": roi
        }
