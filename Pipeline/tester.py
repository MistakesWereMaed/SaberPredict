import os
import json
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from Pipeline.driver import Pipeline

PATH_CLIPS       = "Dataset/Videos/Clips"

PATH_TEST        = "Dataset/Data/Processed/cls_test.csv"
PATH_TEST_OUT    = "Dataset/Data/test.csv"
PATH_METRICS     = "Dataset/Data/metrics.json"

# ----------------------------
# Utilities
# ----------------------------

def restructure_test_df(test_df: pd.DataFrame) -> pd.DataFrame:
    df = test_df.copy()

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
            "LEFT": "left_label",
            "RIGHT": "right_label",
        }
    )

    return wide.sort_values(["file", "frame_idx"]).reset_index(drop=True)

def print_summary(metrics):
    print("\n=== Overall Accuracy ===")
    for side in ["left", "right"]:
        acc = metrics["accuracy"][side]
        print(f"{side.capitalize():>6}: {acc:.4f}")
    print(f"{'Mean':>6}: {metrics['accuracy']['mean']:.4f}")

    print("=== Pose Availability ===")
    for side in ["left", "right"]:
        avail = metrics["pose_availability"][side]
        print(f"{side.capitalize():>6}: {avail:.3f}")

    print("=== Timing (ms) ===")
    for stage, t in metrics["speed"].items():
        if isinstance(t, dict):
            print(f"{stage:>25}: mean={t['mean']:.2f}, p50={t['p50']:.2f}, p90={t['p90']:.2f}, p99={t['p99']:.2f}")
        else:
            print(f"{stage:>25}: {t:.2f}")

    print("=== Per-class F1 ===")
    for side in ["left", "right"]:
        print(f"\n-- {side.upper()} --")
        for cls, m in metrics["per_class"][side].items():
            print(f"{cls:>25}: f1={m['f1']:.3f}, precision={m['precision']:.3f}, recall={m['recall']:.3f}, support={m['support']}")

def save_metrics_json(metrics, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

# ----------------------------
# Evaluation entry point
# ----------------------------

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
        if stats["time_total_ms"]["mean"] > 0
        else 0.0
    )

    return stats

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

def evaluate_labels(y_true, y_pred, label_order):
    mask = (y_true != "SKIPPED") & (y_pred != "SKIPPED")
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    acc = float(accuracy_score(y_true, y_pred))

    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_order,
        zero_division=0,
    )

    per_class = {
        label: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(label_order)
    }

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=label_order,
    ).tolist()  # critical

    return {
        "accuracy": acc,
        "per_class": per_class,
        "confusion_matrix": cm,
    }

def evaluate_pipeline_results(results_df, label_order):
    metrics = {}

    # ---- Left fencer ----
    left = evaluate_labels(
        results_df["left_label_gt"].values,
        results_df["left_label_pred"].values,
        label_order,
    )

    # ---- Right fencer ----
    right = evaluate_labels(
        results_df["right_label_gt"].values,
        results_df["right_label_pred"].values,
        label_order,
    )

    metrics["accuracy"] = {
        "left": left["accuracy"],
        "right": right["accuracy"],
        "mean": 0.5 * (left["accuracy"] + right["accuracy"]),
    }

    metrics["per_class"] = {
        "left": left["per_class"],
        "right": right["per_class"],
    }

    metrics["confusion_matrix"] = {
        "left": left["confusion_matrix"],
        "right": right["confusion_matrix"],
        "labels": label_order,
    }

    metrics["pose_availability"] = {
        "left": float((results_df["left_keypoints"].str.len() > 0).mean()),
        "right": float((results_df["right_keypoints"].str.len() > 0).mean()),
    }

    metrics["speed"] = compute_speed_stats(results_df)

    return metrics

# ----------------------------
# Main test runner
# ----------------------------

def run_full_test(
    pipeline,
    test_df,
    clips_root,
    run_classification=True,
):
    all_records = []

    for file_name in test_df["file"].unique():
        print(f"Processing {file_name}...")
        video_path = os.path.join(clips_root, file_name)
        gt = test_df[test_df["file"] == file_name]

        pred_df = pipeline.run(
            video_path,
            run_classification=run_classification,
        )

        merged = pred_df.merge(
            gt,
            on="frame_idx",
            how="left",
            suffixes=("_pred", "_gt"),
        )

        merged["file"] = file_name
        all_records.append(merged)

    full_df = pd.concat(all_records, ignore_index=True)
    print(f"Done.")
    return full_df

def main():
    pipeline = Pipeline()
    label_order = pipeline.label_map["label"].tolist()

    test_df = pd.read_csv(PATH_TEST)
    test_df = restructure_test_df(test_df)

    df = run_full_test(pipeline, test_df, PATH_CLIPS)
    metrics = evaluate_pipeline_results(df, label_order)

    df.rename(columns={"frame_idx": "frame"}, inplace=True)
    df = df[["file", "frame", "left_label_gt", "left_label_pred", "right_label_gt", "right_label_pred", "roi", "left_keypoints", "right_keypoints"]]
    
    df.to_csv(PATH_TEST_OUT, index=False)
    save_metrics_json(metrics, PATH_METRICS)

    print_summary(metrics)

if __name__ == "__main__":
    main()