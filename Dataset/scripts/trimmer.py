import os
import cv2
import numpy as np
import pandas as pd

CLIPS_DIR = "../clips"
OUTPUT_DIR = "../last30"
TARGET_FRAMES = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_clip(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Failed to open {input_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Start extracting from last TARGET_FRAMES frames
    start_frame = max(0, total_frames - TARGET_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Pad with black frames if too short
    while len(frames) < TARGET_FRAMES:
        frames.insert(0, np.zeros_like(frames[0]))

    # Ensure exactly TARGET_FRAMES
    frames = frames[-TARGET_FRAMES:]

    # Save new clip
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    return True

def main():
    metadata = []
    counter = 1

    for root, _, files in os.walk(CLIPS_DIR):
        bout_id = os.path.basename(root)  # folder name is bout ID
        for file in files:
            if not file.endswith(".mp4"):
                continue

            input_path = os.path.join(root, file)
            new_filename = f"{counter}.mp4"
            output_path = os.path.join(OUTPUT_DIR, new_filename)

            success = process_clip(input_path, output_path)
            if success:
                # Extract label (everything after underscore in filename)
                if "_" in file:
                    label = file.split("_")[-1].replace(".mp4", "")
                else:
                    label = "Unknown"

                metadata.append({
                    "new_clip_id": counter,
                    "bout_id": bout_id,
                    "source_filename": file,
                    "label": label
                })

                print(f"Processed {file} -> {new_filename}")
                counter += 1

    # Save metadata
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
    print(f"Saved metadata with {len(metadata)} entries.")

if __name__ == "__main__":
    main()