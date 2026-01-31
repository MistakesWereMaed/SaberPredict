#!/usr/bin/env bash
set -euo pipefail

# ---------------- USER CONFIG ---------------- #

DATASET_ROOT="../../Dataset/YoloROI"
DATA_YAML="${DATASET_ROOT}/data.yaml"

MODEL="yolo11n.pt"          # start small; upgrade only if needed
IMG_SIZE=640
EPOCHS=60
BATCH=16
DEVICE=0                    # GPU id, use "cpu" if needed
WORKERS=1

PROJECT="Checkpoints/strip_roi"
NAME="yolo11m_finetune"

# --------------------------------------------- #

echo "============================================"
echo " Training YOLO Strip ROI Detector"
echo "============================================"
echo "Dataset:  ${DATA_YAML}"
echo "Model:    ${MODEL}"
echo "Img size: ${IMG_SIZE}"
echo "Epochs:   ${EPOCHS}"
echo "Batch:    ${BATCH}"
echo "Device:   ${DEVICE}"
echo "--------------------------------------------"

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
echo " Best model:"
echo " Checkpoints/${PROJECT}/${NAME}/weights/best.pt"
echo "============================================"
