# pose_estimator.py
import time

from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, model_path, imgsz=1280, conf=0.5):
        self.model = YOLO(model_path, task="pose")
        self.imgsz = imgsz
        self.conf = conf

    def infer(self, frame):
        t0 = time.perf_counter()

        res = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
        )

        poses = []
        for r in res:
            if r.keypoints is None:
                continue

            kpts_xy = r.keypoints.xy.cpu().numpy()
            kpts_conf = (
                r.keypoints.conf.cpu().numpy()
                if hasattr(r.keypoints, "conf")
                else None
            )

            for i, kpts in enumerate(kpts_xy):
                poses.append({
                    "keypoints": kpts,
                    "confidence": (
                        float(kpts_conf[i].mean())
                        if kpts_conf is not None
                        else None
                    ),
                })

        return poses, (time.perf_counter() - t0) * 1000
