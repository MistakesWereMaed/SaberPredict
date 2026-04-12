import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from Pipeline.classifier import TCN

# ---------------- CONFIG ---------------- #
CSV_PATH = "Dataset/Data/Processed/cls_data.csv"
MODEL_PATH = "Training/Checkpoints/best_fold_model.ckpt"
BOUT_ID = "3"  # test bout

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- LOAD MODEL ---------------- #
model = TCN.load_from_checkpoint(MODEL_PATH, map_location=DEVICE).eval().to(DEVICE)


# ---------------- LOAD DATA ---------------- #
df = pd.read_csv(CSV_PATH)

# Filter test bout
df = df[df["file"].str.startswith(f"{BOUT_ID}/")].copy()


# ---------------- PREPROCESS ---------------- #
# Extract keypoint columns
kpt_cols = [c for c in df.columns if c.startswith("x") or c.startswith("y")]

# Ensure correct ordering (x0,y0,x1,y1,...)
kpt_cols = sorted(kpt_cols, key=lambda x: (int(x[1:]), x[0]))


# ---------------- GROUP WINDOWS ---------------- #
windows = []

group_cols = ["file", "fencer", "window_id"]

for (file, fencer, window_id), g in tqdm(df.groupby(group_cols)):
    g = g.sort_values("frame")

    # shape: (T, 34)
    kpts = g[kpt_cols].values

    # reshape to (T, 17, 2)
    kpts = kpts.reshape(len(g), 17, 2)

    label = g["action"].iloc[0]

    windows.append({
        "file": file,
        "fencer": fencer,
        "window_id": window_id,
        "keypoints": kpts,
        "label": label,
    })


# ---------------- LABEL MAP ---------------- #
labels = sorted(df["action"].unique())
label_to_id = {l: i for i, l in enumerate(labels)}
id_to_label = {i: l for l, i in label_to_id.items()}


# ---------------- INFERENCE ---------------- #
clip_stats = defaultdict(lambda: {
    "correct": 0,
    "total": 0,
    "latencies": [],
    "preds": [],
    "targets": [],
})

for w in tqdm(windows):
    x = torch.tensor(w["keypoints"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    y = label_to_id[w["label"]]

    # timing
    start = torch.cuda.Event(enable_timing=True) if DEVICE == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if DEVICE == "cuda" else None

    if DEVICE == "cuda":
        start.record()

    with torch.no_grad():
        logits, _ = model(x)

    if DEVICE == "cuda":
        end.record()
        torch.cuda.synchronize()
        latency = start.elapsed_time(end)  # ms
    else:
        latency = 0.0

    probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()

    file = w["file"]

    clip_stats[file]["total"] += 1
    clip_stats[file]["correct"] += int(pred == y)
    clip_stats[file]["latencies"].append(latency)
    clip_stats[file]["preds"].append(pred)
    clip_stats[file]["targets"].append(y)


# ---------------- METRICS ---------------- #
records = []

for file, stats in clip_stats.items():
    total = stats["total"]
    correct = stats["correct"]

    acc = correct / total if total > 0 else 0.0
    avg_latency = np.mean(stats["latencies"]) if stats["latencies"] else 0.0

    # simple class diversity (optional usefulness metric)
    unique_preds = len(set(stats["preds"]))

    records.append({
        "file": file,
        "accuracy": acc,
        "num_windows": total,
        "avg_latency_ms": avg_latency,
        "unique_pred_classes": unique_preds,
    })


results_df = pd.DataFrame(records)


# ---------------- SORTING ---------------- #
# Best clips (high accuracy, reasonable size)
best = results_df.sort_values(
    by=["accuracy", "num_windows"],
    ascending=[False, False]
)

# Worst clips (good for failure demo)
worst = results_df.sort_values(
    by=["accuracy", "num_windows"],
    ascending=[True, False]
)


# ---------------- SAVE ---------------- #
results_df.to_csv("clip_metrics.csv", index=False)
best.head(10).to_csv("best_clips.csv", index=False)
worst.head(10).to_csv("worst_clips.csv", index=False)


print("\nTop 5 Best Clips:")
print(best.head())

print("\nTop 5 Worst Clips:")
print(worst.head())