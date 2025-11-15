import os
import csv
import random

from tqdm import tqdm
from roboflow import Roboflow

# === CONFIGURATION ===
WORKSPACE_ID = "research-ogpqe"
PROJECT_ID = "bladedetection-gaewy"
BATCH_NAME = "fencing_frames"
FRAMES_DIR = "../low_res/frames"
METADATA_FILE = os.path.join(FRAMES_DIR, "metadata.csv")
TRAIN_SPLIT = 0.8                      # 80% train, 20% test

# ======================

# --- Retrieve API key from environment ---
API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "   Missing Roboflow API key. Please set it using:\n"
        "   export ROBOFLOW_API_KEY='your_key_here'  (Linux/macOS)\n"
        "   setx ROBOFLOW_API_KEY your_key_here      (Windows)"
    )

# --- Initialize Roboflow project ---
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)

# --- Load metadata ---
metadata = {}
with open(METADATA_FILE, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        metadata[row["image_filename"]] = row

# --- Collect all image files ---
image_files = [
    f for f in os.listdir(FRAMES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_files:
    raise FileNotFoundError(f"No image files found in {FRAMES_DIR}/")

# --- Shuffle and split ---
random.shuffle(image_files)
split_idx = int(len(image_files) * TRAIN_SPLIT)
train_files = image_files[:split_idx]
test_files = image_files[split_idx:]

print(f"   Found {len(image_files)} images total")
print(f"   → {len(train_files)} for training")
print(f"   → {len(test_files)} for testing\n")

# --- Upload function ---
def upload_batch(file_list, split):
    for img_file in tqdm(file_list, desc=f"Uploading {split} images"):
        img_path = os.path.join(FRAMES_DIR, img_file)
        row = metadata.get(img_file, {})

        tags = []
        if row:
            tags = [
                f"source_{row.get('source_dir', 'unknown')}",
                f"clip_{os.path.splitext(row.get('clip_filename', 'unknown'))[0]}",
                f"timestamp_{row.get('timestamp_sec', 'unknown')}",
            ]

        try:
            project.upload(
                image_path=img_path,
                batch_name=BATCH_NAME,
                split=split,
                num_retry_uploads=3,
                tag_names=tags
            )
        except Exception as e:
            print(f"Failed to upload {img_file}: {e}")

# --- Upload both splits ---
upload_batch(train_files, "train")
upload_batch(test_files, "test")

print("\nAll uploads complete!")