#!/usr/bin/env bash
set -euo pipefail

# ---------------- USER CONFIG ---------------- #

DATASET_ROOT="../Dataset/YoloROI"
DATA_YAML="${DATASET_ROOT}/data.yaml"

MODEL_SIZE="x"
MODEL="yolo26${MODEL_SIZE}.pt"
IMG_SIZE=640
EPOCHS=30
BATCH=16
DEVICE=0
WORKERS=2

PROJECT="Checkpoints"
NAME="yolo26${MODEL_SIZE}_finetune"

# --------------------------------------------- #

echo "============================================"
echo " Training YOLO Strip ROI Detector"
echo "============================================"

yolo detect train \
  model=${MODEL} \
  data=${DATA_YAML} \
  imgsz=${IMG_SIZE} \
  epochs=${EPOCHS} \
  batch=${BATCH} \
  workers=${WORKERS} \
  device=${DEVICE} \
  project=${PROJECT} \
  name=${NAME} \
  cache=True \
  cos_lr=True \
  patience=15 \
  verbose=True

echo "============================================"
echo " Training complete"
echo "============================================"

RUN_DIR="${PROJECT}/${NAME}"
RESULTS_CSV="${RUN_DIR}/results.csv"
METRICS_JSON="${RUN_DIR}/metrics.json"

if [ ! -f "$RESULTS_CSV" ]; then
    echo "results.csv not found!"
    exit 1
fi

echo "Extracting best metrics..."

python3 - <<EOF
import pandas as pd
import json
import os

results_path = "${RESULTS_CSV}"
out_path = "${METRICS_JSON}"

df = pd.read_csv(results_path)

# Best epoch = highest mAP50-95
best_idx = df["metrics/mAP50-95(B)"].idxmax()
best_row = df.loc[best_idx]

metrics = {
    "best_epoch": int(best_row["epoch"]),
    "precision": float(best_row["metrics/precision(B)"]),
    "recall": float(best_row["metrics/recall(B)"]),
    "mAP50": float(best_row["metrics/mAP50(B)"]),
    "mAP50-95": float(best_row["metrics/mAP50-95(B)"]),
}

with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved metrics to:", out_path)
print(json.dumps(metrics, indent=2))
EOF

echo "============================================"
echo " Best model:"
echo " ${RUN_DIR}/weights/best.pt"
echo " Metrics saved to:"
echo " ${METRICS_JSON}"
echo "============================================"