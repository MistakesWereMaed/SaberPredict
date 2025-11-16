import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

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

def horizontal_flip(keypoints, image_width):
    keypoints = np.array(keypoints).copy()

    # Flip x
    keypoints[:, 0] = image_width - keypoints[:, 0]

    # Swap LR indexes
    for a, b in LEFT_RIGHT_PAIRS:
        keypoints[a], keypoints[b] = keypoints[b].copy(), keypoints[a].copy()

    return keypoints

def duplicate_and_flip_actions(df, image_width):
    flipped_rows = []
    last_action_id = df["action_id"].max().astype(int) + 1

    for _, group in df.groupby(["action_id"]):
        fencer = group["fencer"].iloc[0]
        action = group["action"].iloc[0]

        # Flip fencer label
        new_fencer = "LEFT" if fencer == "RIGHT" else "RIGHT"

        for _, row in group.iterrows():
            flipped_keypoints = horizontal_flip(row.keypoints, image_width)

            new_row = row.copy()
            new_row["fencer"] = new_fencer
            new_row["keypoints"] = flipped_keypoints
            new_row["action"] = action
            new_row["action_id"] = last_action_id
            flipped_rows.append(new_row)
    
        last_action_id += 1

    df_flipped = pd.DataFrame(flipped_rows)
    return pd.concat([df, df_flipped], ignore_index=True)


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

def create_action_windows(
    df,
    window_size=4,
    windows_per_action=5,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    window_rows = []
    window_counter = 0

    # Sort for safety
    df = df.sort_values(["action_id", "frame"]).reset_index(drop=True)

    for _, group in df.groupby(["action_id"]):
        frames = group["frame"].values

        # Determine all valid starting indices
        max_start = len(frames) - window_size

        # If action is barely long enough, take only 1 window
        n_windows = min(windows_per_action, max_start + 1)

        # Sample random starting indices
        start_indices = rng.choice(
            np.arange(0, max_start + 1),
            size=n_windows,
            replace=False if n_windows <= max_start + 1 else True
        )

        for start in start_indices:
            window_counter += 1
            window_id = window_counter

            window_slice = group.iloc[start:start + window_size].copy()
            window_slice["window_id"] = window_id

            window_rows.append(window_slice)

    return pd.concat(window_rows, ignore_index=True)

def split_data(df, id_column="window_id", train_frac=0.8, val_frac=0.1, test_frac=0.1, random_state=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"

    # Get unique IDs
    unique_ids = df[id_column].unique()

    # Split train vs remaining
    train_ids, temp_ids = train_test_split(unique_ids, train_size=train_frac, random_state=random_state)

    # Compute relative fraction for val/test split
    val_relative = val_frac / (val_frac + test_frac)
    val_ids, test_ids = train_test_split(temp_ids, train_size=val_relative, random_state=random_state)

    # Filter dataframes
    train_df = df[df[id_column].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[id_column].isin(val_ids)].reset_index(drop=True)
    test_df = df[df[id_column].isin(test_ids)].reset_index(drop=True)

    print(f"Train: {len(train_df)} rows, {len(train_ids)} actions")
    print(f"Test: {len(test_df)} rows, {len(test_ids)} actions")
    print(f"Val: {len(val_df)} rows, {len(val_ids)} actions")

    return train_df, val_df, test_df

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