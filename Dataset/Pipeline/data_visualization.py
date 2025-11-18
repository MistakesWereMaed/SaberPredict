import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

from IPython.display import display, HTML

PATH_CLIPS = "../../Dataset/Videos/Clips/"

ROI = (0, 600, 1900, 850)
EDGES = [
    (0, 1), (1, 3),
    (3, 5), (1, 2),
    (0, 2), (2, 4),
    (4, 6),
    (5, 7), (7, 9),     # left arm
    (6, 8), (8, 10),    # right arm
    (5, 6),             # shoulders
    (11, 12),           # hips
    (5, 11), (6, 12),   # torso
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16)  # right leg
]

def show_stats(df):
    def display_side_by_side(dfs: list, titles: list = None):
        html_str = ""
        for i, df in enumerate(dfs):
            title = f"<h4>{titles[i]}</h4>" if titles else ""
            html_str += f"""
            <div style="display: inline-block; vertical-align: top; margin-right: 30px;">
                {title}
                {df.to_html(index=False)}
            </div>
            """
        display(HTML(html_str))

    def get_stats(df):
        df = df.copy()
        df["duration"] = df["end_frame"] - df["start_frame"]
        stats = (
            df.groupby("action")["duration"]
            .agg(["min", "max", "mean", "std"])
            .reset_index()
        )

        stats["count"] = df["action"].value_counts().reindex(stats["action"]).values
        return stats.sort_values(by="count", ascending=False)

    df_offense = df[df["action"].str.startswith("ATTACK", na=False)]
    df_defense = df[df["action"].str.startswith("DEFENSE", na=False)]

    stats_offense = get_stats(df_offense)
    stats_defense = get_stats(df_defense)

    total_offensive = len(df_offense)
    total_defensive = len(df_defense)
    total_actions = len(df)

    display_side_by_side([stats_offense, stats_defense], titles=["Offensive Action Frame Durations", "Defensive Actions Frame Durations"])

    print(f"Total Offensive Actions: {total_offensive}")
    print(f"Total Defensive Actions: {total_defensive}")

    print(f"\nTotal Actions: {total_actions}")
    print("Total action classes: ", df["action"].nunique())
    print("")

def plot_skeleton(frame, keypoints, color=(0,1,0), alpha=0.8):
    for (i,j) in EDGES:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        x1, y1 = keypoints[i]
        x2, y2 = keypoints[j]
        plt.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=2)
    keypoints = np.array(keypoints)
    plt.scatter(keypoints[:,0], keypoints[:,1], color=color, s=20)

def draw_random_frame(df):
    # Pick a random row from the dataframe
    row = df.sample(1).iloc[0]
    video_path = PATH_CLIPS + row["file"]
    frame_idx = row["frame"]
    keypoints = row["keypoints"]
    box = row["box"]
    confidence = row["confidence"]

    # Open video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {frame_idx}")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12,6))
    plt.imshow(frame_rgb)
    plt.axis("off")

    # Draw bounding box
    if box and not isinstance(box, float):
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="red", linewidth=2))

    # Draw skeleton
    if keypoints and not (isinstance(keypoints, float) and np.isnan(keypoints)):
        plot_skeleton(frame_rgb, keypoints, color=(0,1,0))

    plt.title(f"Frame {frame_idx}, Fencer: {row['fencer']}, Action: {row['action']}, Confidence: {confidence:.2f}")
    plt.show()

    cap.release()

def draw_random_window(df_windows, window_size=4):
    import random

    # Pick a random window_id
    window_id = random.choice(df_windows["window_id"].unique())
    window_df = df_windows[df_windows["window_id"] == window_id].sort_values("frame")

    # Get video file path (assume all rows have the same file)
    video_file = window_df["file"].iloc[0]
    video_path = f"{PATH_CLIPS}/{video_file}"

    cap = cv2.VideoCapture(video_path)
    frame_indices = window_df["frame"].values
    keypoints_list = window_df["keypoints"].values
    fencer_list = window_df["fencer"].values

    action = window_df["action"].iloc[0]

    plt.figure(figsize=(20, 5))
    
    for i, (fid, kpts, fencer) in enumerate(zip(frame_indices, keypoints_list, fencer_list)):
        # Go to the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax = plt.subplot(1, window_size, i+1)
        ax.imshow(frame_rgb)
        ax.axis("off")
        ax.set_title(f"{fencer} - Frame {fid} - Action: {action}")

        # Convert keypoints string/list to numpy if necessary
        if isinstance(kpts, str):
            kpts = np.array(eval(kpts))
        elif isinstance(kpts, list):
            kpts = np.array(kpts)

        if kpts is not None and not (isinstance(kpts, float) and np.isnan(kpts)):
            plot_skeleton(ax, kpts, color=(0,1,0) if fencer=="LEFT" else (1,0,0))

    plt.tight_layout()
    plt.show()
    cap.release()
