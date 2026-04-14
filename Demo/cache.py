import cv2
import pickle
import os
import pandas as pd

from Pipeline.driver import Pipeline


VIDEOS = [
    "Dataset/Data/Videos/Clips/3/26_Right.mp4",
    "Dataset/Data/Videos/Clips/3/12_Left.mp4",
    "Dataset/Data/Videos/Clips/3/2_Right.mp4",
]

PATH_ACTIONS = "Dataset/Data/Processed/actions_filtered.csv"
PATH_CLIPS = "Dataset/Data/Videos/Clips"


# ---------------- LOAD FRAMES ---------------- #

def load_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Failed to load video: {path}")

    return frames


# ---------------- LOAD GT ---------------- #

def load_ground_truth(video_path, action_df):
    """
    Build frame-level GT aligned to pipeline frame index.
    Same logic conceptually as benchmark script.
    """

    rel_path = os.path.relpath(video_path, PATH_CLIPS)

    df = action_df[action_df["file"] == rel_path].copy()

    # Expand to frame-level mapping
    rows = []

    for _, r in df.iterrows():
        for f in range(int(r["start_frame"]), int(r["end_frame"]) + 1):
            rows.append({
                "frame": f,
                "fencer": r["fencer"],
                "label": r["action"]
            })

    gt = pd.DataFrame(rows)

    # pivot to LEFT/RIGHT columns per frame
    gt = gt.pivot(index="frame", columns="fencer", values="label").reset_index()

    return gt


# ---------------- PRECOMPUTE ---------------- #

def precompute(video_path, out_path, action_df):

    pipe = Pipeline()
    frames = load_frames(video_path)
    gt_df = load_ground_truth(video_path, action_df)

    results = []

    H0, W0 = frames[0].shape[:2]

    for i, frame in enumerate(frames):

        if frame.shape[:2] != (H0, W0):
            raise ValueError(
                f"Inconsistent frame size at frame {i}: "
                f"expected {(H0, W0)}, got {frame.shape[:2]}"
            )

        res = pipe.step(frame)

        # ---------------- ALIGN GT ---------------- #
        gt_row = gt_df[gt_df["frame"] == i]

        if len(gt_row) > 0:
            gt_row = gt_row.iloc[0]
            res["left"]["true_label"] = gt_row.get("LEFT", None)
            res["right"]["true_label"] = gt_row.get("RIGHT", None)
        else:
            res["left"]["true_label"] = None
            res["right"]["true_label"] = None

        res["frame_idx"] = i

        results.append(res)

        if i % 50 == 0:
            print(f"[{os.path.basename(video_path)}] {i}/{len(frames)}")

    cache = {
        "frames": frames,
        "results": results,
        "frame_shape": (H0, W0)
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(cache, f)


# ---------------- ENTRY ---------------- #

if __name__ == "__main__":

    action_df = pd.read_csv(PATH_ACTIONS)

    for i, v in enumerate(VIDEOS):
        precompute(v, f"Demo/Cache/video_{i}.pkl", action_df)