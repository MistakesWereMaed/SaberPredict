import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from IPython.display import display, HTML

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

def plot_poses(df, start_index=0, end_index=3, cols=4):
    SKELETON = [
        (0, 1), (1, 3),
        (3, 5), (1, 2),
        (0, 2), (2, 4),
        (4, 6),
        (5, 7), (7, 9),  # left arm
        (6, 8), (8, 10), # right arm
        (5, 6),          # shoulders
        (11, 12),        # hips
        (5, 11), (6, 12),# torso
        (11, 13), (13, 15), # left leg
        (12, 14), (14, 16)  # right leg
    ]

    # Setup grid
    n = end_index - start_index + 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 4))
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    # Build skeleton graph once
    G = nx.Graph()
    G.add_edges_from(SKELETON)

    for i in range(start_index, end_index + 1):
        row = df.iloc[i]
        keypoints = row["keypoints"]

        fencer = row["fencer"] if "fencer" in row else "N/A"
        action = row["action"] if "action" in row else "N/A"

        # Create node positions
        ax = axes[i - start_index]
        pos = {j: keypoints[j] for j in range(len(keypoints))}
        labels = {label: label for label in G.nodes}

        # Draw the skeleton
        nx.draw(G, pos, ax=ax, labels=labels, node_size=60, node_color='red', edge_color='blue', width=2)
        ax.set_title(f"Index {i} \nFencer: {fencer} \nAction: {action}")
        ax.invert_yaxis()
        ax.axis("equal")

    plt.tight_layout()
    plt.show()

