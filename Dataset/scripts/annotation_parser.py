import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

# TODO: Save action attributes too

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    records = []

    # --------------------------------------------------------------------------
    # STEP 1: Extract task-level metadata (id → source filename)
    # --------------------------------------------------------------------------
    task_sources = {}
    for task in root.findall(".//meta/project/tasks/task"):
        task_id = task.findtext("id")
        source = task.findtext("source")
        name = task.findtext("name")

        if task_id:
            task_sources[task_id] = {
                "source": source,
                "name": name
            }

    # --------------------------------------------------------------------------
    # STEP 2: Parse all tracks and link to the correct task
    # --------------------------------------------------------------------------
    for track in root.findall("track"):
        task_id = track.get("task_id")

        # Match this track to its source video via task_id
        task_info = task_sources.get(task_id, {})
        source = task_info.get("source", "unknown")
        task_name = task_info.get("name", "unknown")

        # ---- Extract track-level attributes ----
        track_attrs = {
            attr.get("name"): attr.text.strip() if attr.text else ""
            for attr in track.findall("attribute")
        }

        # ---- Extract per-frame boxes and attributes ----
        for box in track.findall("box"):
            frame = int(box.get("frame"))

            # Frame-level attributes
            frame_attrs = {
                attr.get("name"): attr.text.strip() if attr.text else ""
                for attr in box.findall("attribute")
            }

            # Merge attributes
            all_attrs = {**track_attrs, **frame_attrs}

            records.append({
                "task_name": task_name,
                "source": source,
                "frame": frame,
                **all_attrs
            })

    return pd.DataFrame(records)

def combine_frames(df):
    df["clip_number"] = df["clip_id"].str.extract(r"^(\d+)").astype(int)
    df["frame"] = df.groupby(["source_id", "clip_id"])["frame"].transform(lambda x: x - x.min())

    # Sort by hierarchy including numeric clip number
    df = df.sort_values(["source_id", "clip_number", "fencer", "frame"]).reset_index(drop=True)

    # Columns to group by for hierarchy
    group_cols = ["source_id", "clip_number", "clip_id", "fencer"]

    results = []

    # Iterate over groups
    for _, grp in df.groupby(group_cols):
        grp = grp.sort_values("frame").reset_index(drop=True)
        
        # Use shift/cumsum to identify consecutive runs
        grp["run"] = (grp["action"] != grp["action"].shift()).cumsum()
        
        # Aggregate start/end frames per run
        run_df = grp.groupby(["run", "action"]).agg(
            start_frame=("frame", "min"),
            end_frame=("frame", "max")
        ).reset_index(drop=False)
        
        # Add hierarchy columns
        run_df["source_id"] = grp["source_id"].iloc[0]
        run_df["clip_id"] = grp["clip_id"].iloc[0]
        run_df["fencer"] = grp["fencer"].iloc[0]
        
        # Keep only desired columns
        run_df = run_df[["source_id", "clip_id", "fencer", "action", "start_frame", "end_frame"]]
        results.append(run_df)

    # Combine all groups
    return pd.concat(results, ignore_index=True)

def parse_annotations(path_annotations):
    df = parse_xml(path_annotations)

    df["task_name"] = df["task_name"].str.replace("Bout ", "", regex=False)
    df["task_name"] = df["task_name"].replace("Test Upload", "1")

    df.rename(columns={"task_name": "source_id"}, inplace=True)
    df.rename(columns={"source": "clip_id"}, inplace=True)
    df.rename(columns={"ID": "fencer"}, inplace=True)
    df.rename(columns={"Action": "action"}, inplace=True)

    cols = ["source_id", "clip_id", "fencer", "action", "frame"]
    df = df[cols]

    return combine_frames(df)

def sample_action_snippets(df, snippet_len=4, n_samples=3, random_state=None):
    np.random.seed(random_state)
    snippets = []

    for _, row in df.iterrows():
        start, end = row["start_frame"], row["end_frame"]

        # possible start indices for a 4-frame window
        possible_starts = np.arange(start, end - snippet_len + 2)
        if len(possible_starts) == 0:
            continue

        # random sampling without replacement
        n_to_sample = min(n_samples, len(possible_starts))
        sampled_starts = np.random.choice(possible_starts, n_to_sample, replace=False)

        for s in sampled_starts:
            snippets.append({
                "file": row["file"],
                "fencer": row["fencer"],
                "action": row["action"],
                "start_frame": s,
                "end_frame": s + snippet_len - 1,
            })

    return pd.DataFrame(snippets)