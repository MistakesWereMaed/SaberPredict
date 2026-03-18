import os
import pandas as pd
import numpy as np

from Pipeline.driver import Pipeline

PATH_CLIPS            = "Dataset/Data/Videos/Clips/"
PATH_ACTIONS_FILTERED = "Dataset/Data/Processed/actions_filtered.csv"

PATH_KEYPOINTS        = "Dataset/Data/Unprocessed/keypoints.csv"
PATH_METRICS          = "Dataset/Data/Unprocessed/metrics.csv"

# ------------------------------------------------------------
# Video processing
# ------------------------------------------------------------

def process_video(video_path, pipeline, action_df):
    """
    Run pose estimation on every frame of a video and calculate coverage metrics per action.
    """
    pipeline.roi = None
    df = pipeline.run(video_path, run_classification=False)

    frame_rows = []

    for _, row in df.iterrows():
        frame_idx = int(row["frame_idx"])

        for fencer in ["LEFT", "RIGHT"]:
            kp_col   = f"{fencer.lower()}_keypoints"
            conf_col = f"{fencer.lower()}_confidence"

            kpts = row.get(kp_col, [])
            conf = row.get(conf_col, 0.0)

            frame_rows.append({
                "frame": frame_idx,
                "fencer": fencer,
                "roi": row.get("roi"),
                "keypoints": kpts if isinstance(kpts, list) else [],
                "confidence": float(conf) if conf is not None else 0.0,
            })

    frame_df = pd.DataFrame(frame_rows)

    # ---------------- Metrics ----------------
    metrics = []

    # Select actions for this video
    video_actions = action_df[action_df["file"] == os.path.relpath(video_path, PATH_CLIPS)]

    for _, action_row in video_actions.iterrows():
        fencer = action_row["fencer"]
        start = action_row["start_frame"]
        end   = action_row["end_frame"]
        action_id = action_row["action_id"]

        expected = end - start + 1

        subset = frame_df[
            (frame_df.fencer == fencer)
            & (frame_df.frame >= start)
            & (frame_df.frame <= end)
        ]

        actual = subset["confidence"].gt(0).sum()

        metrics.append({
            "fencer": fencer,
            "action_id": action_id,
            "start_frame": start,
            "end_frame": end,
            "expected": expected,
            "actual": actual,
            "coverage": actual / expected * 100.0,
        })

    metrics_df = pd.DataFrame(metrics)

    return frame_df, metrics_df

# ------------------------------------------------------------
# Dataset-level driver
# ------------------------------------------------------------

def process_all(pipeline):
    all_frames = []
    all_metrics = []

    action_df = pd.read_csv(PATH_ACTIONS_FILTERED)

    for root, _, files in os.walk(PATH_CLIPS):
        for file in files:
            if not file.lower().endswith((".mp4", ".avi", ".mov")):
                continue

            full_path = os.path.join(root, file)
            rel_path  = os.path.relpath(full_path, PATH_CLIPS)

            print(f"Processing: {rel_path}")

            frame_df, metrics_df = process_video(full_path, pipeline, action_df)

            frame_df["file"]   = rel_path
            metrics_df["file"] = rel_path

            all_frames.append(frame_df)
            all_metrics.append(metrics_df)

    df_keypoints = pd.concat(all_frames, ignore_index=True)
    df_metrics   = pd.concat(all_metrics, ignore_index=True)

    # Final formatting
    df_keypoints.sort_values(["file", "fencer", "frame"], inplace=True)
    df_keypoints.reset_index(drop=True, inplace=True)
    df_keypoints = df_keypoints[["file", "fencer", "frame", "roi", "confidence", "keypoints"]]

    return df_keypoints, df_metrics

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    pipeline = Pipeline()
    keypoints_df, metrics_df = process_all(pipeline)

    keypoints_df.to_csv(PATH_KEYPOINTS, index=False)
    metrics_df.to_csv(PATH_METRICS, index=False)

    print(f"Saved keypoints to {PATH_KEYPOINTS}")
    print(f"Saved metrics to {PATH_METRICS}")

if __name__ == "__main__":
    main()