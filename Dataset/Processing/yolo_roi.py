import cv2
import pandas as pd

from pathlib import Path
from collections import defaultdict

# ---------------- CONFIG ---------------- #

VIDEO_ROOT  = Path("../Videos/Clips")       # root of nested video folders
OUTPUT_ROOT = Path("../YoloROI")            # YOLO dataset root

TRAIN_CSV   = Path("../Data/Processed/train_roi.csv")
VAL_CSV     = Path("../Data/Processed/test_roi.csv")

IMG_EXT     = ".jpg"
CLASS_ID    = 0  # strip

# --------------------------------------- #

def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh


def load_frame_map(csv_path):
    """
    Returns:
        dict: {video_path: {frame_idx: (x1,y1,x2,y2)}}
    """
    df = pd.read_csv(csv_path)
    frame_map = defaultdict(dict)

    for _, row in df.iterrows():
        video = row["file"]
        frame = int(row["frame"])  # subtract 1 here if frames are 1-based
        frame_map[video][frame] = (
            row["xtl"], row["ytl"], row["xbr"], row["ybr"]
        )

    return frame_map


def process_split(frame_map, split):
    img_dir = OUTPUT_ROOT / "images" / split
    lbl_dir = OUTPUT_ROOT / "labels" / split

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for rel_video_path, frames in frame_map.items():
        print(f"Processing {rel_video_path}")

        video_path = VIDEO_ROOT / rel_video_path
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video_path}")

        needed_frames = set(frames.keys())
        current_frame = 0

        while needed_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame in needed_frames:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = frames[current_frame]

                # Save image
                video_path_text = rel_video_path.replace("/", "_")
                p = Path(video_path_text)
                filename = p.with_suffix('').name
                
                img_name = f"{filename}_frame_{current_frame:06d}{IMG_EXT}"
                img_path = img_dir / img_name
                cv2.imwrite(str(img_path), frame)

                # Save label
                xc, yc, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
                lbl_path = lbl_dir / img_name.replace(IMG_EXT, ".txt")
                with open(lbl_path, "w") as f:
                    f.write(f"{CLASS_ID} {xc} {yc} {bw} {bh}\n")

                needed_frames.remove(current_frame)

            current_frame += 1

        cap.release()

def write_data_yaml():
    yaml_path = OUTPUT_ROOT / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(
            f"""path: {OUTPUT_ROOT.resolve()}
train: images/train
val: images/val

names:
  0: strip
"""
        )


def main():
    train_map = load_frame_map(TRAIN_CSV)
    val_map   = load_frame_map(VAL_CSV)

    process_split(train_map, "train")
    process_split(val_map, "val")

    write_data_yaml()
    print("YOLO dataset created successfully.")


if __name__ == "__main__":
    main()
