import torch
import numpy as np

from collections import deque

NUM_JOINTS      = 17
WINDOW_SIZE     = 10

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
            # Strictly match training-time shape
            keypoints = np.zeros((self.num_joints, 2), dtype=np.float32)

        self.buffer.append(keypoints.astype(np.float32))

    def is_ready(self):
        return len(self.buffer) == self.window_size

    def get_window(self):
        """
        Returns shape (1, T, 17, 2)
        """
        assert self.is_ready()
        window = np.stack(self.buffer, axis=0)  # (8, 17, 2)
        return torch.from_numpy(window).unsqueeze(0)