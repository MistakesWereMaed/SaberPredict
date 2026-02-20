import os
import csv
import subprocess

RAW_DIR = "../Raw"
BOUTS_FILE = "../manifest.csv"

def download_and_process(id, url):
    # temp download path (yt-dlp will decide extension)
    temp_path = os.path.join(RAW_DIR, f"{id}.%(ext)s")

    # Step 1: Download with yt-dlp
    cmd_download = [
        "yt-dlp",
        "-f", "bestvideo+bestaudio/best",
        "-o", temp_path,
        url,
    ]
    subprocess.run(cmd_download, check=True)

    # Find the actual downloaded file (yt-dlp replaces %(ext)s)
    downloaded_files = [
        f for f in os.listdir(RAW_DIR)
        if f.startswith(f"{id}.") and not f.endswith(".mp4")
    ]
    if not downloaded_files:
        print(f"No file downloaded for bout {id}")
        return
    downloaded_file = os.path.join(RAW_DIR, downloaded_files[0])

    # Step 2: Re-encode with keyframe every frame (-g 1) to MP4
    final_path = os.path.join(RAW_DIR, f"{id}.mp4")
    cmd_encode = [
        "ffmpeg",
        "-i", downloaded_file,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-g", "1",       # keyframe every frame
        "-c:a", "aac",
        "-y",            # overwrite
        final_path,
    ]
    subprocess.run(cmd_encode, check=True)

    # Step 3: Cleanup
    if downloaded_file != final_path:
        os.remove(downloaded_file)

    print(f"Saved processed bout {id} → {final_path}")


def main():
    with open(BOUTS_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id = row["id"]
            url = row["url"]
            download_and_process(id, url)


if __name__ == "__main__":
    main()