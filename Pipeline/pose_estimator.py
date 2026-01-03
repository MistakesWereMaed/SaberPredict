import numpy as np

class OnlinePoseEstimator:
    def __init__(
        self,
        person_model,
        pose_model,
        roi,
        pad=20,
        imgsz=1280,
        person_conf=0.25,
        pose_conf=0.5,
    ):
        self.person_model = person_model
        self.pose_model = pose_model
        self.roi = roi
        self.pad = pad
        self.imgsz = imgsz
        self.person_conf = person_conf
        self.pose_conf = pose_conf

    def _detect_people(self, frame):
        res = self.person_model(
            frame, imgsz=self.imgsz, conf=self.person_conf, verbose=False
        )

        boxes = []
        for r in res:
            for box in r.boxes.xyxy.cpu().numpy():
                boxes.append(tuple(map(int, box)))

        return boxes

    def _filter_and_assign_boxes(self, boxes):
        x1r, y1r, x2r, y2r = self.roi

        # ROI filter
        boxes = [
            b for b in boxes
            if x1r <= (b[0] + b[2]) / 2 <= x2r
            and y1r <= (b[1] + b[3]) / 2 <= y2r
        ]

        # Keep two largest
        boxes.sort(
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True,
        )
        boxes = boxes[:2]

        left_box = right_box = None

        if len(boxes) == 2:
            if (boxes[0][0] + boxes[0][2]) < (boxes[1][0] + boxes[1][2]):
                left_box, right_box = boxes
            else:
                right_box, left_box = boxes
        elif len(boxes) == 1:
            left_box = boxes[0]

        return left_box, right_box
    
    def _pad_box(self, box, frame_shape, pad=10):
        x1, y1, x2, y2 = box
        h, w = frame_shape[:2]
        x1_new = max(0, x1 - pad)
        y1_new = max(0, y1 - pad)
        x2_new = min(w, x2 + pad)
        y2_new = min(h, y2 + pad)
        return int(x1_new), int(y1_new), int(x2_new), int(y2_new)

    def _estimate_pose(self, frame, box):
        if box is None:
            return None, None

        fx1, fy1, fx2, fy2 = self._pad_box(box, frame.shape, self.pad)
        crop = frame[fy1:fy2, fx1:fx2]

        res = self.pose_model(crop, conf=self.pose_conf, verbose=False)

        poses = []
        confs = []

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
                k = kpts.copy()
                k[:, 0] += fx1
                k[:, 1] += fy1
                poses.append(k)

                if kpts_conf is not None and len(kpts_conf) > i:
                    confs.append(float(np.nanmean(kpts_conf[i])))
                else:
                    confs.append(1.0)

        if not poses:
            return None, None

        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        dists = [
            np.linalg.norm(np.mean(p, axis=0) - np.array([cx, cy]))
            for p in poses
        ]

        best = int(np.argmin(dists))
        return poses[best], confs[best]

    def process_frame(self, frame, frame_idx):
        boxes = self._detect_people(frame)
        left_box, right_box = self._filter_and_assign_boxes(boxes)

        left_pose, left_conf = self._estimate_pose(frame, left_box)
        right_pose, right_conf = self._estimate_pose(frame, right_box)

        return {
            "frame_idx": frame_idx,
            "LEFT": {
                "box": left_box,
                "keypoints": left_pose,
                "confidence": left_conf,
            },
            "RIGHT": {
                "box": right_box,
                "keypoints": right_pose,
                "confidence": right_conf,
            },
        }