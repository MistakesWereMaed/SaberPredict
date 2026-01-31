import time
import numpy as np

class PoseFilter:
    def __init__(self, min_area=1500, max_outside_ratio=0.5):
        self.min_area = min_area
        self.max_outside_ratio = max_outside_ratio

    @staticmethod
    def pose_bbox(keypoints):
        x = keypoints[:, 0]
        y = keypoints[:, 1]
        x1, x2 = x.min(), x.max()
        y1, y2 = y.min(), y.max()
        area = (x2 - x1) * (y2 - y1)
        return x1, y1, x2, y2, area

    @staticmethod
    def outside_ratio(bbox, roi):
        bx1, by1, bx2, by2, area = bbox
        rx1, ry1, rx2, ry2 = roi

        ix1 = max(bx1, rx1)
        iy1 = max(by1, ry1)
        ix2 = min(bx2, rx2)
        iy2 = min(by2, ry2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        return (area - inter) / area if area > 0 else 1.0

    def filter_and_assign(self, poses, roi):
        t0 = time.perf_counter()

        valid = []
        for p in poses:
            bbox = self.pose_bbox(p["keypoints"])
            if bbox[4] < self.min_area:
                continue
            if self.outside_ratio(bbox, roi) > self.max_outside_ratio:
                continue

            valid.append({
                **p,
                "area": bbox[4],
                "cx": (bbox[0] + bbox[2]) / 2,
            })

        valid.sort(key=lambda p: p["area"], reverse=True)
        valid = valid[:2]
        valid.sort(key=lambda p: p["cx"])

        left  = valid[0] if len(valid) > 0 else None
        right = valid[1] if len(valid) > 1 else None

        return (
            {"LEFT": left, "RIGHT": right},
            (time.perf_counter() - t0) * 1000,
        )