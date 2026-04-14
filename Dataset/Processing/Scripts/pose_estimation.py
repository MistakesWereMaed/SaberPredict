import os
import pandas as pd
import tqdm

from Pipeline.driver import Pipeline


PATH_CLIPS            = "Dataset/Data/Videos/Clips/"
PATH_ACTIONS_FILTERED = "Dataset/Data/Processed/actions_filtered.csv"

PATH_KEYPOINTS        = "Dataset/Data/Unprocessed/keypoints.csv"
PATH_METRICS          = "Dataset/Data/Unprocessed/metrics.csv"


# ------------------------------------------------------------
# FLATTEN PIPELINE OUTPUT
# ------------------------------------------------------------

def flatten_frames(df):
    rows = []

    for _, r in df.iterrows():
        res = r["result"] if "result" in df.columns else r

        rows.append({
            "frame": r["frame_idx"],

            "roi": res["roi"],

            "left_keypoints": res["left"]["kpts"],
            "left_confidence": res["left"]["conf"],

            "right_keypoints": res["right"]["kpts"],
            "right_confidence": res["right"]["conf"],
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# VIDEO PROCESSING
# ------------------------------------------------------------

def process_video(video_path, pipeline, action_df):

    pipeline.roi = None

    df = pipeline.run(video_path, run_classification=False)
    frame_df = flatten_frames(df)

    metrics = []

    rel_path = os.path.relpath(video_path, PATH_CLIPS)
    video_actions = action_df[action_df["file"] == rel_path]

    for _, action_row in video_actions.iterrows():

        fencer = action_row["fencer"]
        start  = action_row["start_frame"]
        end    = action_row["end_frame"]
        action_id = action_row["action_id"]

        expected = end - start + 1

        subset = frame_df[
            (frame_df.frame >= start) &
            (frame_df.frame <= end)
        ]

        if fencer.lower() == "left":
            conf_series = subset["left_confidence"]
        else:
            conf_series = subset["right_confidence"]

        # improved definition of "pose present"
        actual = (conf_series > 0).sum()

        metrics.append({
            "fencer": fencer,
            "action_id": action_id,
            "start_frame": start,
            "end_frame": end,
            "expected": expected,
            "actual": actual,
            "coverage": (actual / expected * 100.0) if expected > 0 else 0.0,
        })

    metrics_df = pd.DataFrame(metrics)

    return frame_df, metrics_df


# ------------------------------------------------------------
# DATASET DRIVER
# ------------------------------------------------------------

def process_all(pipeline):

    all_frames = []
    all_metrics = []

    action_df = pd.read_csv(PATH_ACTIONS_FILTERED)

    for root, _, files in os.walk(PATH_CLIPS):
        for file in tqdm.tqdm(files, desc="Processing videos"):
            if not file.lower().endswith((".mp4", ".avi", ".mov")):
                continue

            full_path = os.path.join(root, file)
            rel_path  = os.path.relpath(full_path, PATH_CLIPS)

            frame_df, metrics_df = process_video(full_path, pipeline, action_df)

            frame_df["file"] = rel_path
            metrics_df["file"] = rel_path

            all_frames.append(frame_df)
            all_metrics.append(metrics_df)

            break

    df_keypoints = pd.concat(all_frames, ignore_index=True)
    df_metrics   = pd.concat(all_metrics, ignore_index=True)

    df_keypoints.sort_values(["file", "frame"], inplace=True)
    df_keypoints.reset_index(drop=True, inplace=True)

    df_keypoints = df_keypoints[
        ["file", "frame", "roi",
         "left_keypoints", "left_confidence",
         "right_keypoints", "right_confidence"]
    ]

    return df_keypoints, df_metrics


# ------------------------------------------------------------
# MAIN
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