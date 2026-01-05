import numpy as np
import pandas as pd

def interpolate_action_group(group):
    stats = {
        "processed": 1,
        "dropped": 0,
        "skipped": 0,
        "interpolated": 0,
        "interpolated_frames": 0,
        "copied_frames": 0,
    }

    # Sort correctly
    group = group.sort_values("frame").reset_index(drop=True)

    frames = group["frame"].to_numpy()
    keypoints_list = group["keypoints"].to_list()

    # Drop actions with only 1 frame
    if len(keypoints_list) <= 1:
        stats["dropped"] = 1
        return None, stats

    start_f = group["start_frame"].iloc[0]
    end_f = group["end_frame"].iloc[0]

    expected_len = end_f - start_f + 1
    actual_len = len(group)

    # Early skip if no missing frames
    if expected_len == actual_len:
        stats["skipped"] = 1
        # Standardize keypoints format
        group["keypoints"] = group["keypoints"].apply(
            lambda kp: [tuple(j) for j in kp]
        )
        return group.copy(), stats

    # Otherwise interpolate
    full_frames = np.arange(start_f, end_f + 1)

    # Convert keypoints list -> ndarray (N, 16, 2)
    keypoints_array = np.array([np.array(kp) for kp in keypoints_list])

    # --- Count missing frame stats ---
    known_set = set(frames)
    missing_frames = [f for f in full_frames if f not in known_set]

    # Border-fill frames: frames outside the min and max of real frames
    min_real = frames.min()
    max_real = frames.max()

    border_fill_frames = [f for f in missing_frames if f < min_real or f > max_real]
    inner_missing_frames = [f for f in missing_frames if min_real < f < max_real]

    stats["interpolated_frames"] = len(inner_missing_frames)
    stats["copied_frames"] = len(border_fill_frames)
    stats["interpolated"] = 1

    # --- Interpolation process ---
    interp_keypoints = []

    for kp_idx in range(keypoints_array.shape[1]):
        xy = keypoints_array[:, kp_idx, :]  # (N, 2)
        x_vals = xy[:, 0]
        y_vals = xy[:, 1]

        # Linear interpolation
        x_interp = np.round(np.interp(full_frames, frames, x_vals), 2)
        y_interp = np.round(np.interp(full_frames, frames, y_vals), 2)

        interp_keypoints.append(np.vstack([x_interp, y_interp]).T)

    # Rearrange to shape (num_frames, 16, 2)
    interp_keypoints = np.stack(interp_keypoints, axis=1)

    # Convert to list of tuples
    final_keypoints = [
        [tuple(j) for j in frame_kps]
        for frame_kps in interp_keypoints
    ]

    # Build result dataframe
    out = pd.DataFrame({
        "file": group["file"].iloc[0],
        "fencer": group["fencer"].iloc[0],
        "action": group["action"].iloc[0],
        "action_id": group["action_id"].iloc[0],
        "frame": full_frames,
        "start_frame": start_f,
        "end_frame": end_f,
        "keypoints": final_keypoints
    })

    return out, stats

def interpolate_frames(df):
    all_groups = []
    total_stats = {
        "processed": 0,
        "dropped": 0,
        "skipped": 0,
        "interpolated": 0,
        "interpolated_frames": 0,
        "copied_frames": 0,
    }

    groups = df.groupby(["file", "fencer", "action" , "start_frame", "end_frame"])
    for _, group in groups:
        new_group, stats = interpolate_action_group(group)

        # accumulate stats
        for k in total_stats:
            total_stats[k] += stats[k]

        # keep only valid groups
        if new_group is not None:
            all_groups.append(new_group)

    df_interpolated = pd.concat(all_groups, ignore_index=True)
    stats = pd.Series(total_stats)

    return df_interpolated, stats