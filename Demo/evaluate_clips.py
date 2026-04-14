import os
import cv2
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from Pipeline.driver import Pipeline


# ---------------- CONFIG ---------------- #
PATH_CLIPS = "Dataset/Data/Videos/Clips"
PATH_KEYPOINTS = "Dataset/Data/Unprocessed/keypoints.csv"
PATH_ACTIONS = "Dataset/Data/Processed/actions_filtered.csv"

BOUT_ID = "3/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- LOAD DATA ---------------- #

df_keypoints = pd.read_csv(PATH_KEYPOINTS)
df_actions = pd.read_csv(PATH_ACTIONS)


# ---------------- BUILD FRAME-LEVEL GT ---------------- #

def build_frame_gt(df_keypoints, df_actions, bout_id):
    df = df_keypoints.merge(df_actions, on=["file", "fencer"], how="left")

    df = df[
        (df["frame"] >= df["start_frame"]) &
        (df["frame"] <= df["end_frame"])
    ]

    df = df[df["file"].str.startswith(bout_id)]

    # Pivot to LEFT / RIGHT
    df = df[["file", "frame", "fencer", "action"]]

    wide = df.pivot(
        index=["file", "frame"],
        columns="fencer",
        values="action"
    ).reset_index()

    wide = wide.rename(columns={
        "frame": "frame_idx",
        "LEFT": "left_label",
        "RIGHT": "right_label"
    })

    return wide.sort_values(["file", "frame_idx"]).reset_index(drop=True)


df_gt = build_frame_gt(df_keypoints, df_actions, BOUT_ID)


# ---------------- EVALUATION ---------------- #

def evaluate_pipeline():
    pipeline = Pipeline()

    clip_stats = defaultdict(lambda: {
        "correct": 0,
        "total": 0,
        "latencies": [],
        "preds": [],
        "targets": [],
    })

    all_true = []
    all_pred = []

    for file_name in tqdm(df_gt["file"].unique()):
        video_path = os.path.join(PATH_CLIPS, file_name)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Skipping {file_name}")
            continue

        gt_clip = df_gt[df_gt["file"] == file_name]

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx >= len(gt_clip):
                break

            gt_row = gt_clip.iloc[frame_idx]

            result = pipeline.step(frame)

            # LEFT
            gt_left = gt_row["left_label"]
            pred_left = result["left"]["label"]

            if gt_left != "SKIPPED" and pred_left != "SKIPPED":
                correct = int(gt_left == pred_left)

                clip_stats[file_name]["correct"] += correct
                clip_stats[file_name]["total"] += 1
                clip_stats[file_name]["preds"].append(pred_left)
                clip_stats[file_name]["targets"].append(gt_left)

                all_true.append(gt_left)
                all_pred.append(pred_left)

            # RIGHT
            gt_right = gt_row["right_label"]
            pred_right = result["right"]["label"]

            if gt_right != "SKIPPED" and pred_right != "SKIPPED":
                correct = int(gt_right == pred_right)

                clip_stats[file_name]["correct"] += correct
                clip_stats[file_name]["total"] += 1
                clip_stats[file_name]["preds"].append(pred_right)
                clip_stats[file_name]["targets"].append(gt_right)

                all_true.append(gt_right)
                all_pred.append(pred_right)

            # timing
            clip_stats[file_name]["latencies"].append(result["timing"]["total"])

            frame_idx += 1

        cap.release()

    # ---------------- GLOBAL METRICS ---------------- #

    global_acc = np.mean(np.array(all_true) == np.array(all_pred))
    print(f"\nGlobal Accuracy: {global_acc:.4f}")

    # ---------------- PER-CLIP METRICS ---------------- #

    records = []

    for file, stats in clip_stats.items():
        total = stats["total"]
        correct = stats["correct"]

        acc = correct / total if total > 0 else 0.0
        avg_latency = np.mean(stats["latencies"]) if stats["latencies"] else 0.0

        # instability metric (prediction flips)
        preds = stats["preds"]
        flips = sum(p1 != p2 for p1, p2 in zip(preds[:-1], preds[1:]))

        records.append({
            "file": file,
            "accuracy": acc,
            "num_frames": total,
            "avg_latency_ms": avg_latency,
            "prediction_flips": flips,
        })

    results_df = pd.DataFrame(records)

    # ---------------- SORT ---------------- #

    best = results_df.sort_values(
        by=["accuracy", "num_frames"],
        ascending=[False, False]
    )

    worst = results_df.sort_values(
        by=["accuracy", "num_frames"],
        ascending=[True, False]
    )

    # ---------------- SAVE ---------------- #

    results_df.to_csv("Demo/clips/clip_metrics_pipeline.csv", index=False)
    best.head(10).to_csv("Demo/clips/best_clips.csv", index=False)
    worst.head(10).to_csv("Demo/clips/worst_clips.csv", index=False)

    print("\nTop 5 Best Clips:")
    print(best.head())

    print("\nTop 5 Worst Clips:")
    print(worst.head())


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    evaluate_pipeline()