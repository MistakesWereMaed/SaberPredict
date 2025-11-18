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

def process_video(video_path, person_model, pose_model, frame_ranges):
    cap = cv2.VideoCapture(video_path)

    # --- Rebuild fast lookup: frame → {fencer → action_id}
    frame_to_actions = defaultdict(dict)
    expected_pose_frames = defaultdict(lambda: defaultdict(int))  # action_id → fencer → count
    actual_pose_frames   = defaultdict(lambda: defaultdict(int))

    for fencer, ranges in frame_ranges.items():
        for start, end, action_id in ranges:
            for f in range(start, end + 1):
                frame_to_actions[f][fencer] = action_id
                expected_pose_frames[action_id][fencer] += 1

    # --- Output dictionary for all frames
    frame_data = {
        "frame": [],
        "left_box": [],
        "right_box": [],
        "left_pose": [],
        "right_pose": [],
        "left_action_id": [],
        "right_action_id": [],
    }

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames not involved in ANY action for ANY fencer
        if frame_idx not in frame_to_actions:
            frame_idx += 1
            continue

        active_actions = frame_to_actions[frame_idx]  # dict: {fencer → action_id}

        # --- Step 1: Detect bounding boxes
        det = person_model(frame, verbose=False, imgsz=IMAGE_SIZE)[0]

        boxes = det.boxes.xyxy.cpu().numpy() if det.boxes else []

        # Apply ROI filtering
        filtered = []
        for b in boxes:
            x1, y1, x2, y2 = map(int, b)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if (
                ROI["x1"] <= cx <= ROI["x2"] and
                ROI["y1"] <= cy <= ROI["y2"]
            ):
                filtered.append(b)

        # Keep only the 2 largest boxes (if more than 2 remain)
        if len(filtered) > 2:
            areas = [(b, (b[2]-b[0]) * (b[3]-b[1])) for b in filtered]
            filtered = [x[0] for x in sorted(areas, key=lambda v: v[1], reverse=True)[:2]]

        # Pad boxes slightly
        padded_boxes = []
        pad = 20  # small padding
        h, w = frame.shape[:2]

        for b in filtered:
            x1, y1, x2, y2 = b
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w - 1, x2 + pad)
            y2 = min(h - 1, y2 + pad)
            padded_boxes.append([x1, y1, x2, y2])

        # --- Step 2: Run pose estimation on each box
        poses = []
        for pb in padded_boxes:
            x1, y1, x2, y2 = map(int, pb)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                poses.append(None)
                continue

            r = pose_model(crop, conf=0.5, verbose=False)[0]

            if r.keypoints is None or len(r.keypoints.xy) == 0:
                poses.append(None)
                continue

            # Choose best pose by highest mean keypoint confidence
            best_idx = np.argmax(r.keypoints.conf.cpu().numpy().mean(axis=1))
            kpts = r.keypoints.xy[best_idx].cpu().numpy()

            # Re-offset to full frame coordinates
            kpts[:, 0] += x1
            kpts[:, 1] += y1

            poses.append(kpts)

        # --- Step 3: Assign left/right fencer
        left_pose = right_pose = None
        left_box = right_box = None
        left_action = right_action = np.nan

        if len(padded_boxes) == 2:
            # Left = smaller x1
            if padded_boxes[0][0] < padded_boxes[1][0]:
                left_box, right_box = padded_boxes
                left_pose, right_pose = poses
            else:
                right_box, left_box = padded_boxes
                right_pose, left_pose = poses

        # --- Step 4: Assign action_id if applicable
        if "LEFT" in active_actions:
            left_action = active_actions["LEFT"]
        if "RIGHT" in active_actions:
            right_action = active_actions["RIGHT"]

        # Count metrics
        if left_pose is not None and not np.isnan(left_action):
            actual_pose_frames[left_action]["LEFT"] += 1
        if right_pose is not None and not np.isnan(right_action):
            actual_pose_frames[right_action]["RIGHT"] += 1

        # --- Save frame data
        frame_data["frame"].append(frame_idx)
        frame_data["left_box"].append(left_box)
        frame_data["right_box"].append(right_box)
        frame_data["left_pose"].append(left_pose)
        frame_data["right_pose"].append(right_pose)
        frame_data["left_action_id"].append(left_action)
        frame_data["right_action_id"].append(right_action)

        frame_idx += 1

    cap.release()

    # --- Build final metrics
    metrics = {
        "expected_pose_frames": expected_pose_frames,
        "actual_pose_frames": actual_pose_frames,
        "coverage": (actual_pose_frames / expected_pose_frames) * 100
    }

    print("\nMetrics:")
    print(f"Expected frames: {expected_pose_frames}")
    print(f"Actual frames:   {actual_pose_frames}")
    print(f"Coverage:        {metrics['coverage']:.2f}%\n")

    return frame_data, metrics

def create_frame_ranges():
    df_filtered = pd.read_csv(PATH_ACTIONS_FILTERED)

    frame_ranges = {}
    for file, group_file in df_filtered.groupby("file"):
        frame_ranges[file] = {}
        for fencer, group_fencer in group_file.groupby("fencer"):
            ranges = list(
                zip(
                    group_fencer["start_frame"],
                    group_fencer["end_frame"],
                    group_fencer["action_id"]
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

            if rel_path not in frame_ranges:
                print(f"⚠️ Skipping {rel_path} — no action ranges found")
                continue

            print(f"Processing: {rel_path}")

            # Run pipeline
            frame_data, metrics = process_video(
                video_path=full_path,
                person_model=person_model,
                pose_model=pose_model,
                frame_ranges=frame_ranges[rel_path],
                roi=ROI
            )

            # Convert each to DF
            frame_df = pd.DataFrame(frame_data)
            metrics_df = pd.DataFrame(metrics, rel_path)

            all_frame_data_dfs.append(frame_df)
            all_metrics_dfs.append(metrics_df)

    # Combine into one DF each
    final_frame_df = pd.concat(all_frame_data_dfs, ignore_index=True)
    final_metrics_df = pd.concat(all_metrics_dfs, ignore_index=True)

    return final_frame_df, final_metrics_df

def main():
    person_model = YOLO("yolo11x.pt", task="detect")
    pose_model   = YOLO("yolo11x-pose.pt", task="pose")
    frame_ranges = create_frame_ranges()

    all_metrics_df, frame_data_df = process_all(person_model, pose_model, frame_ranges)
    all_metrics_df.to_csv(PATH_METRICS, index=False)
    frame_data_df.to_csv(PATH_KEYPOINTS, index=False)