import time
import numpy as np

from ultralytics import YOLO

class ROIDetector:
    def __init__(self, model_path, imgsz=640, conf=0.25):
        self.model = YOLO(model_path, task="detect")
        self.imgsz = imgsz
        self.conf = conf

    def detect(self, frame):
        t0 = time.perf_counter()

        res = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
        )

        boxes = []
        for r in res:
            if r.boxes is not None:
                boxes.extend(r.boxes.xyxy.cpu().numpy())

        if not boxes:
            return None, (time.perf_counter() - t0) * 1000

        boxes.sort(
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True,
        )

        roi = boxes[0].astype(int)
        return roi, (time.perf_counter() - t0) * 1000
