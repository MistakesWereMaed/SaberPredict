import os
import csv
import subprocess
import re

RAW_DIR = "../Raw"
TIMESTAMP_DIR = "../Timestamps"
CLIPS_DIR = "../Clips"

os.makedirs(CLIPS_DIR, exist_ok=True)
GLOBAL_METADATA_PATH = os.path.join(CLIPS_DIR, "metadata.csv")

def split_video(source_id, video_path, timestamp_file, global_writer):
    bout_clip_dir = os.path.join(CLIPS_DIR, str(source_id))
    os.makedirs(bout_clip_dir, exist_ok=True)

    with open(timestamp_file, "r") as f_in:
        reader = csv.reader(f_in)

        for row in reader:
            if len(row) < 3:
                continue

            start, end, raw_label = row
            clip_id = raw_label
            label = re.sub(r'\d+_', '', raw_label)
            clip_name = f"{raw_label}.mp4"
            clip_path = os.path.join(bout_clip_dir, clip_name)

            # Re-encode the clip for frame-accurate trimming and smaller size
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-ss", start,
                "-to", end,
                "-c:v", "libx264",
                "-crf", "23",           # quality/size tradeoff
                "-preset", "veryfast",  # encoding speed
                "-g", "30",             # keyframe interval (~1s at 30fps)
                "-an",                  # remove audio
                "-y",                   # overwrite if exists
                clip_path,
            ]

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Record global metadata
            global_writer.writerow([source_id, clip_id, label])
    print(f"Bout {source_id} complete")

def main():
    # Create (or overwrite) the global metadata file
    with open(GLOBAL_METADATA_PATH, "w", newline="") as global_file:
        writer = csv.writer(global_file)
        writer.writerow(["source_id", "clip_id", "label"])

        for file in os.listdir(TIMESTAMP_DIR):
            if not file.endswith(".csv"):
                continue

            source_id = os.path.splitext(file)[0]
            if source_id in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                continue

            timestamp_file = os.path.join(TIMESTAMP_DIR, file)
            video_file = os.path.join(RAW_DIR, f"{source_id}.mp4")

            if not os.path.exists(video_file):
                print(f"Missing video for bout {source_id}, skipping.")
                continue

            print(f"Processing bout {source_id}...")
            split_video(source_id, video_file, timestamp_file, writer)

    print(f"\nAll clips processed. Combined metadata saved to {GLOBAL_METADATA_PATH}")

if __name__ == "__main__":
    main()