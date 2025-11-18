import cv2
import os
import numpy as np
import pandas as pd

from collections import defaultdict
from ultralytics import YOLO

PATH_CLIPS                          = "../../Dataset/Videos/clips/"

PATH_ACTIONS_FILTERED               = "../../Dataset/Data/tmp/actions_filtered.csv"
PATH_KEYPOINTS                      = "../../Dataset/Data/Unprocessed/keypoints.csv"
PATH_METRICS                        = "./metrics.csv"

ROI = (0, 600, 1900, 850)
IMAGE_SIZE = 1280

def pad_box(box, frame_shape, pad=10):
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    x1_new = max(0, x1 - pad)
    y1_new = max(0, y1 - pad)
    x2_new = min(w, x2 + pad)
    y2_new = min(h, y2 + pad)
    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)

def process_video(video_path, person_model, pose_model, frame_ranges, pad=20):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_data = {}

    # Flatten all action frames for quick lookup
    all_action_frames = set()
    for fencer, ranges in frame_ranges.items():
        for _, start, end in ranges:
            all_action_frames.update(range(start, end+1))

    for fid in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames not in action ranges
        if fid not in all_action_frames:
            frame_count -= 1
            continue

        # 1️⃣ Person detection
        res = person_model(frame, imgsz=1280, conf=0.25, verbose=False)
        boxes_full = []
        for r in res:
            for box in r.boxes.xyxy.cpu().numpy():
                boxes_full.append(tuple(map(int, box)))

        # 2️⃣ Filter boxes by ROI and keep up to 2 largest
        x1r, y1r, x2r, y2r = ROI
        boxes_filtered = []
        for b in boxes_full:
            cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
            if x1r <= cx <= x2r and y1r <= cy <= y2r:
                boxes_filtered.append(b)
        boxes_filtered.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        boxes_filtered = boxes_filtered[:2]

        # 3️⃣ Assign left/right based on x-centroids
        left_box = right_box = None
        if len(boxes_filtered) == 2:
            if (boxes_filtered[0][0]+boxes_filtered[0][2])/2 < (boxes_filtered[1][0]+boxes_filtered[1][2])/2:
                left_box, right_box = boxes_filtered
            else:
                right_box, left_box = boxes_filtered
        elif len(boxes_filtered) == 1:
            left_box, right_box = boxes_filtered[0], np.nan

        # 4️⃣ Run pose estimation only if fencer has an action in this frame
        poses_all = []
        kpt_confs = []

        for fencer, box in zip(["LEFT", "RIGHT"], [left_box, right_box]):
            if isinstance(box, float) or box is None:
                poses_all.append(np.nan)
                kpt_confs.append(np.nan)
                continue

            # Check if this fencer has an action in this frame
            fencer_ranges = frame_ranges.get(fencer, [])
            in_action = any(start <= fid <= end for _, start, end in fencer_ranges)
            if not in_action:
                poses_all.append(np.nan)
                kpt_confs.append(np.nan)
                continue

            # Crop + padding
            fx1, fy1, fx2, fy2 = pad_box(box, frame.shape, pad)
            crop = frame[fy1:fy2, fx1:fx2]
            if crop.size == 0:
                poses_all.append(np.nan)
                kpt_confs.append(np.nan)
                continue

            # Pose detection
            res_pose = pose_model(crop, conf=0.5, verbose=False)
            poses_box, confs_box = [], []
            for r in res_pose:
                if r.keypoints is None or len(r.keypoints.xy) == 0:
                    continue
                for kpts in r.keypoints.xy.cpu().numpy():
                    kpts[:,0] += fx1
                    kpts[:,1] += fy1
                    poses_box.append(kpts)
                    if hasattr(r.keypoints, "conf") and len(r.keypoints.conf) > 0:
                        confs_box.append(np.nanmean(r.keypoints.conf[0].cpu().numpy()))
                    else:
                        confs_box.append(1.0)

            # Keep best pose per box (closest to box center)
            if poses_box:
                x1b, y1b, x2b, y2b = fx1, fy1, fx2, fy2
                cx, cy = (x1b+x2b)/2, (y1b+y2b)/2
                dists = [np.linalg.norm(np.mean(p, axis=0) - np.array([cx, cy])) for p in poses_box]
                best_idx = np.argmin(dists)
                poses_all.append(poses_box[best_idx])
                kpt_confs.append(confs_box[best_idx])
            else:
                poses_all.append(np.nan)
                kpt_confs.append(np.nan)

        # Store frame data
        frame_data[fid] = {
            "boxes": boxes_filtered,
            "poses": poses_all,
            "left_fencer": poses_all[0],
            "right_fencer": poses_all[1] if len(poses_all)>1 else np.nan,
        }

    # -------------------
    # 6️⃣ Calculate metrics per action
    metrics = []
    for fencer, ranges in frame_ranges.items():
        for action_id, start, end in ranges:
            expected = end - start + 1
            actual = 0
            for fid in range(start, end+1):
                pose = frame_data[fid].get(f"{fencer.lower()}_fencer", np.nan)
                if not isinstance(pose, float) and not (isinstance(pose, np.ndarray) and np.isnan(pose).all()):
                    actual += 1
            metrics.append({
                "fencer": fencer,
                "action_id": action_id,
                "start_frame": start,
                "end_frame": end,
                "expected": expected,
                "actual": actual,
                "coverage": actual / expected * 100
            })

            print(f"{fencer} action {action_id} coverage: {actual}/{expected} = {actual/expected*100:.2f}%")

    metrics_df = pd.DataFrame(metrics)
    return frame_data, metrics_df

def create_frame_ranges():
    df_filtered = pd.read_csv(PATH_ACTIONS_FILTERED)

    frame_ranges = {}
    for file, group_file in df_filtered.groupby("file"):
        frame_ranges[file] = {}
        for fencer, group_fencer in group_file.groupby("fencer"):
            ranges = list(
                zip(
                    group_fencer["action_id"],
                    group_fencer["start_frame"],
                    group_fencer["end_frame"]
                )
            )

            frame_ranges[file][fencer] = ranges

    return frame_ranges

def process_all(person_model, pose_model, frame_ranges):
    all_frame_data_dfs = []
    all_metrics_dfs = []

    # Traverse all subfolders and files under PATH_CLIPS
    for root, _, files in os.walk(PATH_CLIPS):
        for file in files:
            if not file.lower().endswith((".mp4", ".avi", ".mov")):
                continue  # skip non-video files

            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, PATH_CLIPS)

            print(f"Processing: {rel_path}")

            # Run pipeline
            frame_data, metrics = process_video(
                video_path=full_path,
                person_model=person_model,
                pose_model=pose_model,
                frame_ranges=frame_ranges[rel_path],
            )

            # Convert to DF
            frame_df = pd.DataFrame(frame_data)

            all_metrics_dfs.append(metrics)
            all_frame_data_dfs.append(frame_df)

    # Combine into one DF each
    final_frame_df = pd.concat(all_frame_data_dfs, ignore_index=True)
    final_metrics_df = pd.concat(all_metrics_dfs, ignore_index=True)

    return final_frame_df, final_metrics_df

def main():
    person_model = YOLO("yolo11x.pt", task="detect")
    pose_model   = YOLO("yolo11x-pose.pt", task="pose")
    frame_ranges = create_frame_ranges()

    frame_data_df, metrics_df = process_all(person_model, pose_model, frame_ranges)
    frame_data_df.to_csv(PATH_KEYPOINTS, index=False)
    metrics_df.to_csv(PATH_METRICS, index=False)

if __name__ == "__main__":
    main()