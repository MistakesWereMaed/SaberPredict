import os
import json
import pandas as pd
import tqdm

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from Pipeline.driver import Pipeline


# ----------------------------
# CONFIG
# ----------------------------

PATH_CLIPS     = "Dataset/Data/Videos/Clips"
PATH_DATA      = "Dataset/Data/Processed/cls_data.csv"
PATH_KEYPOINTS = "Dataset/Data/Unprocessed/keypoints.csv"
PATH_ACTIONS   = "Dataset/Data/Processed/actions_filtered.csv"

PATH_TEST_OUT  = "Experiments/test.csv"
PATH_METRICS   = "Experiments/metrics.json"

TEST_BOUT = "3/"


# ----------------------------
# UTILITIES
# ----------------------------

def restructure_df(df_keypoints, df_actions, test_files):
    df = df_keypoints.merge(df_actions, on=["file", "fencer"], how="left")
    df = df[
        (df["frame"] >= df["start_frame"]) &
        (df["frame"] <= df["end_frame"])
    ]

    df = df[["file", "fencer", "action_id", "action", "frame", "start_frame", "end_frame", "confidence", "keypoints"]].reset_index(drop=True)
    df = df[df["file"].isin(test_files)]

    # Pivot to wide format
    wide = (
        df.pivot(
            index=["file", "frame"],
            columns="fencer",
            values="action",
        )
        .reset_index()
    )

    # Rename to pipeline-expected names
    wide = wide.rename(
        columns={
            "frame": "frame_idx",
            "LEFT": "left_label_gt",
            "RIGHT": "right_label_gt",
        }
    )

    return wide.sort_values(["file", "frame_idx"]).reset_index(drop=True)

def save_metrics_json(metrics, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)


def compute_speed_stats(df):
    cols = [
        "time_total_ms",
        "time_roi_ms",
        "time_pose_ms",
        "time_filter_ms",
        "time_classify_ms",
    ]

    stats = {}
    for c in cols:
        stats[c] = {
            "mean": float(df[c].mean()),
            "p50": float(df[c].quantile(0.50)),
            "p90": float(df[c].quantile(0.90)),
            "p99": float(df[c].quantile(0.99)),
        }

    stats["fps_mean"] = (
        1000.0 / stats["time_total_ms"]["mean"]
        if stats["time_total_ms"]["mean"] > 0 else 0.0
    )

    return stats

def print_summary(metrics):
    print("=== Timing (ms) ===")
    for stage, t in metrics["speed"].items():
        if isinstance(t, dict):
            print(f"{stage:>25}: mean={t['mean']:.2f}, p50={t['p50']:.2f}, p90={t['p90']:.2f}, p99={t['p99']:.2f}")
        else:
            print(f"{stage:>25}: {t:.2f}")

    print("=== Per-class F1 ===")
    for cls, m in metrics["per_class"].items():
        print(f"{cls:>25}: f1={m['f1']:.3f}, precision={m['precision']:.3f}, recall={m['recall']:.3f}, support={m['support']}")

    print("\n=== Overall Accuracy ===")
    print(f"Combined: {metrics['accuracy']:.4f}")


# ----------------------------
# FLATTEN PIPELINE OUTPUT
# ----------------------------

def flatten_pipeline_output(pred_df):
    """
    Converts nested pipeline output into flat dataframe.
    """

    records = []

    for _, row in pred_df.iterrows():
        res = row["result"] if "result" in pred_df.columns else row

        records.append({
            "frame_idx": row["frame_idx"],
            "file": row.get("file", None),

            # LEFT
            "left_label_pred": res["left"]["label"],
            "left_conf": res["left"]["conf"],
            "left_keypoints": res["left"]["kpts"],

            # RIGHT
            "right_label_pred": res["right"]["label"],
            "right_conf": res["right"]["conf"],
            "right_keypoints": res["right"]["kpts"],

            # ROI
            "roi": res["roi"],

            # TIMING
            "time_total_ms": res["timing"]["total"],
            "time_roi_ms": res["timing"]["roi"],
            "time_pose_ms": res["timing"]["pose"],
            "time_filter_ms": res["timing"]["filter"],
            "time_classify_ms": res["timing"]["classify"],
        })

    return pd.DataFrame(records)


# ----------------------------
# EVALUATION
# ----------------------------

def evaluate_pipeline_results(df, label_order):
    metrics = {}

    true, pred = [], []

    # LEFT
    mask_l = (
        df["left_label_gt"].notna() &
        df["left_label_pred"].notna() &
        (df["left_label_gt"] != "SKIPPED") &
        (df["left_label_pred"] != "SKIPPED")
    )

    true.extend(df.loc[mask_l, "left_label_gt"].tolist())
    pred.extend(df.loc[mask_l, "left_label_pred"].tolist())

    # RIGHT
    mask_r = (
        df["right_label_gt"].notna() &
        df["right_label_pred"].notna() &
        (df["right_label_gt"] != "SKIPPED") &
        (df["right_label_pred"] != "SKIPPED")
    )

    true.extend(df.loc[mask_r, "right_label_gt"].tolist())
    pred.extend(df.loc[mask_r, "right_label_pred"].tolist())

    true = pd.Series(true)
    pred = pd.Series(pred)

    # ---------------- ACCURACY ----------------
    metrics["accuracy"] = float(accuracy_score(true, pred))

    # ---------------- PER CLASS ----------------
    p, r, f1, support = precision_recall_fscore_support(
        true,
        pred,
        labels=label_order,
        zero_division=0,
    )

    metrics["per_class"] = {
        cls: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(label_order)
    }

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(true, pred, labels=label_order)

    metrics["confusion_matrix"] = {
        "labels": label_order,
        "matrix": cm.tolist(),
    }

    # ---------------- SPEED ----------------
    metrics["speed"] = compute_speed_stats(df)

    return metrics


# ----------------------------
# RUN PIPELINE ON CLIPS
# ----------------------------

def run_full_test(pipeline, test_df, clips_root, run_classification=True):
    all_records = []

    for file_name in tqdm.tqdm(test_df["file"].unique(), desc="Processing videos"):
        video_path = os.path.join(clips_root, file_name)
        gt = test_df[test_df["file"] == file_name]

        pred_df = pipeline.run(
            video_path,
            run_classification=run_classification,
        )

        flat_pred = flatten_pipeline_output(pred_df)

        # merge GT labels
        merged = flat_pred.merge(
            gt,
            on="frame_idx",
            how="left",
        )

        merged["file"] = file_name
        all_records.append(merged)

    return pd.concat(all_records, ignore_index=True)


# ----------------------------
# MAIN
# ----------------------------

def main():
    pipeline = Pipeline()
    label_order = pipeline.label_map["label"].tolist()

    df_test = pd.read_csv(PATH_DATA)
    df_test = df_test[df_test["file"].str.contains(TEST_BOUT)]

    df_keypoints = pd.read_csv(PATH_KEYPOINTS)
    df_actions = pd.read_csv(PATH_ACTIONS)

    test_files = df_test["file"].unique()
    df_test = restructure_df(df_keypoints, df_actions, test_files)

    df = run_full_test(pipeline, df_test, PATH_CLIPS)
    metrics = evaluate_pipeline_results(df, label_order)

    df.to_csv(PATH_TEST_OUT, index=False)
    save_metrics_json(metrics, PATH_METRICS)
    print_summary(metrics)

if __name__ == "__main__":
    main()