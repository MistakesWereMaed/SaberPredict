import os
import csv
import subprocess

RAW_DIR = "../raw"
TIMESTAMP_DIR = "../timestamps"
CLIPS_DIR = "../clips"

os.makedirs(CLIPS_DIR, exist_ok=True)

def split_video(bout_id, video_path, timestamp_file):
    bout_clip_dir = os.path.join(CLIPS_DIR, str(bout_id))
    os.makedirs(bout_clip_dir, exist_ok=True)

    metadata_path = os.path.join(bout_clip_dir, "metadata.csv")
    with open(timestamp_file, "r") as f_in, open(metadata_path, "w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["clip_filename", "bout_id", "start_time", "end_time", "label"])

        for row in reader:
            if len(row) < 3:
                continue
            start, end, label = row
            clip_name = f"{label}.mp4"
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
                "-g", "30",             # keyframe interval (~2s at 30fps)
                "-an",                  # remove audio
                "-y",                   # overwrite if exists
                clip_path,
            ]

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Saved: {clip_path}")

            # record metadata
            writer.writerow([clip_name, bout_id, start, end, label])

def main():
    for file in os.listdir(TIMESTAMP_DIR):
        if not file.endswith(".csv"):
            continue

        bout_id = os.path.splitext(file)[0]  # e.g. "1"
        timestamp_file = os.path.join(TIMESTAMP_DIR, file)
        video_file = os.path.join(RAW_DIR, f"{bout_id}.mp4")

        if not os.path.exists(video_file):
            print(f"Missing video for bout {bout_id}, skipping.")
            continue

        print(f"Processing bout {bout_id}...")
        split_video(bout_id, video_file, timestamp_file)

if __name__ == "__main__":
    main()
