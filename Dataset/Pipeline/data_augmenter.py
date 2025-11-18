import numpy as np
import pandas as pd

LEFT_RIGHT_PAIRS = [
    (1, 2),   # upper chest / clavicles
    (3, 4),   # upper arms
    (5, 6),   # shoulders

    (7, 8),   # elbows
    (9, 10),  # wrists

    (11, 12), # hips
    (13, 14), # knees
    (15, 16), # ankles
]

def duplicate_and_flip_actions(df, image_width, start_new_id=None):
    df_mirror = df.copy()

    # Determine new action_id starting point
    if start_new_id is None:
        if df["action_id"].dtype.kind in 'iuf':  # numeric
            start_new_id = df["action_id"].max() + 1
        else:
            start_new_id = 0

    # Map original action_id -> new mirrored action_id
    unique_action_ids = df["action_id"].unique()
    action_id_map = {aid: start_new_id + i for i, aid in enumerate(unique_action_ids)}

    # Mirror keypoints
    mirrored_keypoints = []
    for kpts in df_mirror["keypoints"]:
        if isinstance(kpts, list):
            kpts = np.array(kpts)
        if kpts is not None and not (isinstance(kpts, float) and np.isnan(kpts)):
            kpts_flipped = kpts.copy()
            kpts_flipped[:, 0] = np.round(image_width - kpts_flipped[:, 0], 2)
            mirrored_keypoints.append([(float(x), float(y)) for x, y in kpts_flipped])
        else:
            mirrored_keypoints.append(np.nan)
    df_mirror["keypoints"] = mirrored_keypoints

    if "box" in df.columns:
        flipped_boxes = []
        for b in df_mirror["box"]:
            if isinstance(b, str):
                b = tuple(map(float, eval(b)))
            if b is None or (isinstance(b, float) and np.isnan(b)):
                flipped_boxes.append(np.nan)
            else:
                x1, y1, x2, y2 = b
                flipped_boxes.append((float(image_width - x2), float(y1), float(image_width - x1), float(y2)))
        df_mirror["box"] = flipped_boxes

    # Swap fencer
    df_mirror["fencer"] = df_mirror["fencer"].map({"LEFT": "RIGHT", "RIGHT": "LEFT"}).fillna(df_mirror["fencer"])   

    # Assign new action_ids
    df_mirror["action_id"] = df_mirror["action_id"].map(action_id_map)

    # Concatenate original + mirrored
    df_combined = pd.concat([df, df_mirror], ignore_index=True)
    return df_combined

def augment_keypoints(keypoints, jitter_std=1.5, noise_std=1.0, scale_std=0.05, rotate_deg=4):
    keypoints = np.array(keypoints).astype(float)

    # ---- 1. Random global scale ----
    scale = np.random.normal(1.0, scale_std)
    keypoints *= scale

    # ---- 2. Small random rotation ----
    theta = np.radians(np.random.uniform(-rotate_deg, rotate_deg))
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    centroid = keypoints.mean(axis=0)
    keypoints = (keypoints - centroid) @ R.T + centroid

    # ---- 3. Joint jitter (per joint) ----
    keypoints += np.random.normal(0, jitter_std, keypoints.shape)

    # ---- 4. Sensor noise ----
    keypoints += np.random.normal(0, noise_std, keypoints.shape)

    return keypoints.tolist()

def augment_all_actions(df, jitter_std=1.5, noise_std=1.0, scale_std=0.05, rotate_deg=4):
    df_augmented_rows = []

    for _, group in df.groupby("action_id"):
        group = group.sort_values("frame")

        for _, row in group.iterrows():
            augmented_keypoints = augment_keypoints(row.keypoints, jitter_std, noise_std, scale_std, rotate_deg)
            new_row = row.copy()
            new_row["keypoints"] = augmented_keypoints
            df_augmented_rows.append(new_row)

    return pd.DataFrame(df_augmented_rows)

def create_action_windows(df, window_size=4, base_windows=5, class_weights=None, random_state=42):
    rng = np.random.default_rng(random_state)
    window_rows = []
    window_counter = 0

    # Sort for safety
    df = df.sort_values(["action_id", "frame"]).reset_index(drop=True)

    for _, group in df.groupby(["action_id"]):
        action = group["action"].iloc[0]

        # Determine number of windows for this action
        weight = class_weights.get(action, 1.0) if class_weights else 1.0
        windows_per_action = max(1, int(round(base_windows * weight)))

        frames = group["frame"].values
        max_start = len(frames) - window_size
        if max_start < 0:
            continue  # action too short for a single window

        start_indices = rng.choice(
            np.arange(0, max_start + 1),
            size=windows_per_action,
            replace=False if windows_per_action <= max_start + 1 else True
        )

        for start in start_indices:
            window_counter += 1
            window_id = window_counter

            window_slice = group.iloc[start:start + window_size].copy()
            window_slice["window_id"] = window_id
            window_rows.append(window_slice)

    return pd.concat(window_rows, ignore_index=True)

def explode_keypoints(df):
    # Expand each tuple into separate x,y columns
    exploded = df["keypoints"].apply(
        lambda kp: [coord for point in kp for coord in point]
    )

    # Create column names: x0, y0, x1, y1, ...
    num_points = len(df.iloc[0]["keypoints"])
    cols = [f"x{i}" for i in range(num_points)] + [f"y{i}" for i in range(num_points)]
    
    new_df = pd.DataFrame(exploded.tolist(), columns=cols)
    return pd.concat([df.drop(columns=["keypoints"]), new_df], axis=1)