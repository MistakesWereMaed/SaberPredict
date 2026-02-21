import torch
import numpy as np

from collections import deque

NUM_JOINTS  = 17
WINDOW_SIZE = 4

class SkeletonWindowBuffer:
    def __init__(self, fencer, window_size=WINDOW_SIZE, num_joints=NUM_JOINTS):
        self.fencer = fencer
        self.window_size = window_size
        self.num_joints = num_joints
        self.buffer = deque(maxlen=window_size)

    def add_frame(self, keypoints):
        """
        keypoints: np.ndarray (17, 2) or None
        """
        if keypoints is None:
            keypoints = np.zeros((self.num_joints, 2), dtype=np.float32)

        self.buffer.append(keypoints.astype(np.float32))

    def is_ready(self):
        return len(self.buffer) == self.window_size

    def _compute_derivatives(self, window_xy):
        """
        window_xy: (T, 17, 2)
        returns:   (T, 17, 2)  first-order temporal derivative
        """
        T = window_xy.shape[0]

        # forward difference for t>0
        d = np.zeros_like(window_xy, dtype=np.float32)
        d[1:] = window_xy[1:] - window_xy[:-1]

        # first frame derivative remains zero
        return d

    def get_window(self):
        """
        Returns tensor shape (1, T, 17, 4)
        where channels = [x, y, dx, dy]
        """
        assert self.is_ready()

        window_xy = np.stack(self.buffer, axis=0)  # (T, 17, 2)

        # compute temporal derivatives
        window_d = self._compute_derivatives(window_xy)

        # concatenate along coordinate dimension
        window_full = np.concatenate([window_xy, window_d], axis=2)  # (T, 17, 4)

        return torch.from_numpy(window_full).unsqueeze(0)

    def flush(self):
        self.buffer = deque(maxlen=self.window_size)