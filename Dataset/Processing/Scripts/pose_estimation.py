import os
import pandas as pd
import numpy as np

from Pipeline.driver import Pipeline

PATH_CLIPS            = "Dataset/Videos/Clips/"
PATH_ACTIONS_FILTERED = "Dataset/Data/Processed/actions_filtered.csv"

PATH_KEYPOINTS        = "Dataset/Data/Unprocessed/keypoints.csv"
PATH_METRICS          = "Dataset/Data/Unprocessed/metrics.csv"

# ------------------------------------------------------------
# Frame range utilities
# ------------------------------------------------------------

def create_frame_ranges():
    df = pd.read_csv(PATH_ACTIONS_FILTERED)

    frame_ranges = {}
    for file, g_file in df.groupby("file"):
        frame_ranges[file] = {}
        for fencer, g_fencer in g_file.groupby("fencer"):
            frame_ranges[file][fencer] = list(
                zip(
                    g_fencer["action_id"],
                    g_fencer["start_frame"],
                    g_fencer["end_frame"],
                )
            )
    return frame_ranges

# ------------------------------------------------------------
# Video processing
# ------------------------------------------------------------

def process_video(video_path, pipeline, frame_ranges):
    """
    pipeline.run(video_path) -> DataFrame with per-frame outputs
    """

    pipeline.roi = None
    df = pipeline.run(video_path, run_classification=False)

    frame_rows = []

    # Build action lookup
    action_lookup = {"LEFT": set(), "RIGHT": set()}
    for fencer, ranges in frame_ranges.items():
        for _, start, end in ranges:
            action_lookup[fencer].update(range(start, end + 1))

    for _, row in df.iterrows():
        frame_idx = int(row["frame_idx"])

        for fencer in ["LEFT", "RIGHT"]:
            if frame_idx not in action_lookup[fencer]:
                continue

            if fencer == "LEFT":
                kpts = row["left_keypoints"]
                conf = row["left_confidence"]
            else:
                kpts = row["right_keypoints"]
                conf = row["right_confidence"]

            frame_rows.append({
                "frame_idx": frame_idx,
                "fencer": fencer,
                "roi": row["roi"],
                "pose": kpts if isinstance(kpts, list) else [],
                "conf": float(conf) if conf is not None else 0.0,
            })

    frame_df = pd.DataFrame(frame_rows)

    # ---------------- Metrics ----------------

    metrics = []
    for fencer, ranges in frame_ranges.items():
        for action_id, start, end in ranges:
            expected = end - start + 1
            subset = frame_df[
                (frame_df.fencer == fencer)
                & (frame_df.frame_idx >= start)
                & (frame_df.frame_idx <= end)
            ]

            actual = subset["conf"].gt(0).sum()

            metrics.append({
                "fencer": fencer,
                "action_id": action_id,
                "start_frame": start,
                "end_frame": end,
                "expected": expected,
                "actual": actual,
                "coverage": actual / expected * 100.0,
            })

    return frame_df, pd.DataFrame(metrics)

# ------------------------------------------------------------
# Dataset-level driver
# ------------------------------------------------------------

def process_all(pipeline, frame_ranges):
    all_frames = []
    all_metrics = []

    for root, _, files in os.walk(PATH_CLIPS):
        for file in files:
            if not file.lower().endswith((".mp4", ".avi", ".mov")):
                continue

            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, PATH_CLIPS)

            if rel_path not in frame_ranges:
                continue

            print(f"Processing: {rel_path}")

            frame_df, metrics_df = process_video(
                full_path,
                pipeline,
                frame_ranges[rel_path],
            )

            frame_df["file"] = rel_path
            metrics_df["file"] = rel_path

            all_frames.append(frame_df)
            all_metrics.append(metrics_df)

    df_keypoints = pd.concat(all_frames, ignore_index=True)
    df_metrics   = pd.concat(all_metrics, ignore_index=True)

    # Final formatting
    df_keypoints.rename(
        columns={
            "frame_idx": "frame",
            "pose": "keypoints",
            "conf": "confidence",
        },
        inplace=True,
    )

    df_keypoints.sort_values(
        ["file", "fencer", "frame"],
        inplace=True,
    )
    df_keypoints.reset_index(drop=True, inplace=True)

    df_keypoints = df_keypoints[
        ["file", "fencer", "frame", "roi", "confidence", "keypoints"]
    ]

    return df_keypoints, df_metrics

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    pipeline = Pipeline()

    frame_ranges = create_frame_ranges()
    keypoints_df, metrics_df = process_all(pipeline, frame_ranges)

    keypoints_df.to_csv(PATH_KEYPOINTS, index=False)
    metrics_df.to_csv(PATH_METRICS, index=False)

if __name__ == "__main__":
    main()