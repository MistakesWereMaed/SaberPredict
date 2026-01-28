import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cdist

PATH_PERSON_MODEL = "Pipeline/Models/Checkpoints/yolo11x.pt"
PATH_POSE_MODEL   = "Pipeline/Models/Checkpoints/yolo11x-pose.pt"

ROI             = (0, 600, 1900, 850)
PAD             = 20
IMG_SIZE        = 1280

PERSON_CONF_THRESHOLD = 0.25
POSE_CONF_THRESHOLD   = 0.25
MAX_DIST_MOVEMENT     = 0.1  # pixels, threshold to consider pose moved

class PoseEstimator:
    def __init__(
        self,
        roi=ROI,
        pad=PAD,
        imgsz=IMG_SIZE,
        person_conf=PERSON_CONF_THRESHOLD,
        pose_conf=POSE_CONF_THRESHOLD,
    ):
        self.roi        = roi
        self.pad        = pad
        self.imgsz      = imgsz
        self.person_conf= person_conf
        self.pose_conf  = pose_conf

        self.pose_model = YOLO(PATH_POSE_MODEL, task="pose")
        self.previous_skeletons = []  # For motion tracking

    def _in_roi(self, centroid):
        x1, y1, x2, y2 = self.roi
        cx, cy = centroid
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _filter_by_roi(self, poses):
        filtered = []
        for kpts, conf in poses:
            centroid = np.mean(kpts, axis=0)
            if self._in_roi(centroid):
                filtered.append((kpts, conf))
        return filtered

    def _track_motion(self, poses):
        """
        Keep only poses that moved significantly from previous frame.
        Simple nearest-neighbor matching using centroids.
        """
        if not self.previous_skeletons:
            self.previous_skeletons = poses
            return poses[:2]  # take up to 2 poses

        prev_centroids = np.array([np.mean(k, axis=0) for k, _ in self.previous_skeletons])
        new_centroids  = np.array([np.mean(k, axis=0) for k, _ in poses])

        if len(prev_centroids) == 0 or len(new_centroids) == 0:
            self.previous_skeletons = poses
            return poses[:2]

        # Compute distances between previous and current centroids
        dist_matrix = cdist(prev_centroids, new_centroids)
        min_dists = dist_matrix.min(axis=0)  # minimum distance to any previous pose

        # Keep only poses that moved more than threshold
        moving_poses = [(pose, dist) for pose, dist in zip(poses, min_dists) if dist > MAX_DIST_MOVEMENT]

        # Sort by largest movement and keep top 2
        moving_poses.sort(key=lambda x: x[1], reverse=True)
        top_poses = [pose for pose, _ in moving_poses[:2]]

        self.previous_skeletons = poses
        return top_poses

    def process_frame(self, frame, frame_idx):
        # Run full-frame multi-person pose estimation
        res = self.pose_model(frame, conf=self.pose_conf, verbose=False)

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
                poses.append((kpts, float(np.nanmean(kpts_conf[i]))))

        # Filter by ROI
        poses = self._filter_by_roi(poses)

        # Motion filtering
        poses = self._track_motion(poses)
        print(len(poses))

        # Assign left/right based on X centroid
        left_pose = right_pose = None
        if poses:
            poses.sort(key=lambda x: np.mean(x[0][:,0]))  # sort by X centroid
            left_pose = poses[0]
            if len(poses) > 1:
                right_pose = poses[1]

        result = {
            "frame_idx": frame_idx,
            "LEFT": {
                "keypoints": left_pose[0] if left_pose else None,
                "confidence": left_pose[1] if left_pose else None
            },
            "RIGHT": {
                "keypoints": right_pose[0] if right_pose else None,
                "confidence": right_pose[1] if right_pose else None
            },
        }
        return result