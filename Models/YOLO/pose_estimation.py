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

    # The new output structure: ONE ROW PER FENCER PER FRAME
    frame_rows = []

    # Build lookup for "does LEFT/RIGHT have an action on this frame?"
    action_lookup = {"LEFT": set(), "RIGHT": set()}
    for fencer, ranges in frame_ranges.items():
        for (_, start, end) in ranges:
            action_lookup[fencer].update(range(start, end + 1))

    for fid in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # If neither fencer has an action this frame → skip entirely
        if fid not in action_lookup["LEFT"] and fid not in action_lookup["RIGHT"]:
            continue

        # 1️⃣ Person detection
        res = person_model(frame, imgsz=1280, conf=0.25, verbose=False)
        boxes_full = []
        for r in res:
            for box in r.boxes.xyxy.cpu().numpy():
                boxes_full.append(tuple(map(int, box)))

        # 2️⃣ Filter by ROI + keep 2 largest
        x1r, y1r, x2r, y2r = ROI
        boxes_filtered = []
        for b in boxes_full:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if x1r <= cx <= x2r and y1r <= cy <= y2r:
                boxes_filtered.append(b)
        boxes_filtered.sort(key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
        boxes_filtered = boxes_filtered[:2]

        # 3️⃣ Assign left/right box
        left_box = right_box = None
        if len(boxes_filtered) == 2:
            if (boxes_filtered[0][0] + boxes_filtered[0][2]) / 2 < (boxes_filtered[1][0] + boxes_filtered[1][2]) / 2:
                left_box, right_box = boxes_filtered
            else:
                right_box, left_box = boxes_filtered
        elif len(boxes_filtered) == 1:
            left_box, right_box = boxes_filtered[0], np.nan

        # 4️⃣ Compute poses per fencer — only store if fencer has an action
        for fencer, box in zip(["LEFT", "RIGHT"], [left_box, right_box]):

            # If this fencer has no action → skip row entirely
            if fid not in action_lookup[fencer]:
                continue

            # Default values (for failure cases)
            pose_val = np.nan
            conf_val = np.nan
            box_val = box if not isinstance(box, float) else np.nan

            # Missing/invalid box → store NaN row but do NOT run pose estimation
            if isinstance(box, float) or box is None:
                frame_rows.append({
                    "frame_idx": fid,
                    "fencer": fencer,
                    "box": box_val,
                    "pose": pose_val,
                    "conf": conf_val,
                })
                continue

            # Run pose model
            fx1, fy1, fx2, fy2 = pad_box(box, frame.shape, pad)
            crop = frame[fy1:fy2, fx1:fx2]
            res_pose = pose_model(crop, conf=0.5, verbose=False)

            poses_box = []
            confs_box = []

            for r in res_pose:
                if r.keypoints is None or len(r.keypoints.xy) == 0:
                    continue

                kpts_list = r.keypoints.xy.cpu().numpy()
                conf_list = (
                    r.keypoints.conf.cpu().numpy()
                    if hasattr(r.keypoints, "conf")
                    else None
                )

                for i, kpts in enumerate(kpts_list):
                    kpts_adj = kpts.copy()
                    kpts_adj[:, 0] += fx1
                    kpts_adj[:, 1] += fy1
                    poses_box.append(kpts_adj)
                    if conf_list is not None and len(conf_list) > i:
                        confs_box.append(np.nanmean(conf_list[i]))
                    else:
                        confs_box.append(1.0)

            # Select best pose if available
            if poses_box:
                cx = (fx1 + fx2) / 2
                cy = (fy1 + fy2) / 2
                dists = [
                    np.linalg.norm(np.mean(p, axis=0) - np.array([cx, cy]))
                    for p in poses_box
                ]
                best_idx = np.argmin(dists)
                pose_val = poses_box[best_idx]
                conf_val = confs_box[best_idx]

            # Append final row
            frame_rows.append({
                "frame_idx": fid,
                "fencer": fencer,
                "box": box_val,
                "pose": pose_val,
                "conf": conf_val,
            })

    # Convert all rows to DataFrame
    frame_df = pd.DataFrame(frame_rows)

    # -------- Metrics logic unchanged ------
    metrics = []
    for fencer, ranges in frame_ranges.items():
        for action_id, start, end in ranges:
            expected = end - start + 1
            subset = frame_df[
                (frame_df.fencer == fencer) &
                (frame_df.frame_idx >= start) &
                (frame_df.frame_idx <= end)
            ]
            actual = subset["pose"].apply(
                lambda p: isinstance(p, np.ndarray)
            ).sum()

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
    return frame_df, metrics_df

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

            frame_data["file"] = rel_path

            all_metrics_dfs.append(metrics)
            all_frame_data_dfs.append(frame_data)

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