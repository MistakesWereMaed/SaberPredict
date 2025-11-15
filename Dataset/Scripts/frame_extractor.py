import cv2
import os
import random
import csv

CLIPS_DIR = "../low_res/clips" 
OUTPUT_DIR = "../low_res/frames"
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")
NUM_FRAMES = 10 # Number of random frames to extract per clip

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create/open metadata file
with open(METADATA_FILE, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_filename", "source_dir", "clip_filename", "frame_number", "timestamp_sec"])

    # Traverse numbered subdirectories
    for source_dir in sorted(os.listdir(CLIPS_DIR)):
        source_path = os.path.join(CLIPS_DIR, source_dir)
        if not os.path.isdir(source_path):
            continue

        print(f"Processing source: {source_dir}")

        # Process each clip in the subdirectory
        for clip_file in sorted(os.listdir(source_path)):
            if not clip_file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue

            clip_path = os.path.join(source_path, clip_file)
            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened():
                print(f"Failed to open {clip_path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            if frame_count == 0:
                print(f"No frames found in {clip_path}")
                cap.release()
                continue

            # Select random unique frame indices
            selected_frames = random.sample(range(frame_count), min(NUM_FRAMES, frame_count))

            for frame_idx in selected_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    print(f"Could not read frame {frame_idx} from {clip_path}")
                    continue

                timestamp = frame_idx / fps
                img_filename = f"{source_dir}_{os.path.splitext(clip_file)[0]}_frame{frame_idx}.jpg"
                img_path = os.path.join(OUTPUT_DIR, img_filename)

                # Save image
                cv2.imwrite(img_path, frame)

                # Write metadata
                writer.writerow([img_filename, source_dir, clip_file, frame_idx, f"{timestamp:.2f}"])

            cap.release()

print(f"\n Frame extraction complete!")
print(f"Images saved in: {OUTPUT_DIR}")
print(f"Metadata saved at: {METADATA_FILE}")