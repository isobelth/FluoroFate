"""Plotting functions for persistent and snapshot cell-fate analyses."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from colours import assign_colours, build_category_colormap


def plot_persistent_percentages(summary_dataframe, fluorophore_names, title="Persistent Positive Cells Over Time"):
    """Plot cumulative percent-positive cells over frames (persistent mode).

    Each fluorophore gets its own coloured line plus a black "Total" line
    summing all fates. The y-axis is fixed to 0–100% so different
    experiments can be compared at a glance.

    Parameters
    ----------
    summary_dataframe : pandas.DataFrame
        Output of :func:`fate_assignment.compute_persistent_percentages`.
    fluorophore_names : list[str]
        Fluorophore names; each must have a ``<name>_pct`` column in
        ``summary_dataframe``.
    title : str
        Plot title.

    Returns
    -------
    figure, axis : matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    figure, axis = plt.subplots(figsize=(8, 4))
    colour_assignments = assign_colours(fluorophore_names)
    for fluorophore_name in fluorophore_names:
        sns.lineplot(data=summary_dataframe, x="frame", y=f"{fluorophore_name}_pct", color=colour_assignments[fluorophore_name]["mpl"], label=fluorophore_name, ax=axis)
    sns.lineplot(data=summary_dataframe, x="frame", y="total_positive_pct", color="black", label="Total", ax=axis)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set(xlabel="Frame", ylabel="% Cells", ylim=(0, 100), title=title)
    plt.tight_layout()
    return figure, axis


def plot_snapshot_percentages(summary_dataframe, categories, title="Per-Frame Categories Over Time"):
    """Plot percent-cells per category over frames (snapshot mode).

    Each category gets a colour derived from
    :func:`colours.build_category_colormap`. Unlike persistent plots,
    these lines can go up and down between frames.

    Parameters
    ----------
    summary_dataframe : pandas.DataFrame
        Output of :func:`fate_assignment.compute_snapshot_percentages`.
    categories : list[str]
        Category names; each must have a ``<category>_pct`` column in
        ``summary_dataframe``.
    title : str
        Plot title.

    Returns
    -------
    figure, axis : matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    figure, axis = plt.subplots(figsize=(8, 4))
    category_colormap = build_category_colormap(categories)
    for category in categories:
        sns.lineplot(data=summary_dataframe, x="frame", y=f"{category}_pct", color=category_colormap.get(category, (0.5, 0.5, 0.5)), label=category, ax=axis)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set(xlabel="Frame", ylabel="% Cells", ylim=(0, 100), title=title)
    plt.tight_layout()
    return figure, axis


def plot_snapshot_trajectories(tracks_dataframe, snapshot_dataframe, title="Snapshot Trajectories by Category"):
    """Plot cell XY trajectories coloured by snapshot category at each step.

    Each cell's track is drawn as a series of line segments, where each
    segment is coloured according to the cell's category in that frame.
    If lineage data is available (``parent_track_id`` column in
    ``tracks_dataframe``) dashed lines connect parent track endpoints to
    daughter track start points.

    Parameters
    ----------
    tracks_dataframe : pandas.DataFrame
        TrackMate tracks with columns ``track_id, t, x, y`` and
        optionally ``parent_track_id`` for lineage connections.
    snapshot_dataframe : pandas.DataFrame
        Output of :func:`fate_assignment.assign_snapshot_fates`.
    title : str
        Plot title.

    Returns
    -------
    figure, axis : matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    if tracks_dataframe is None or len(tracks_dataframe) == 0:
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.set_title(title)
        axis.text(0.5, 0.5, "No tracks available", ha="center", va="center")
        axis.axis("off")
        plt.tight_layout()
        return figure, axis

    categories = sorted(snapshot_dataframe["category"].unique())
    category_colormap = build_category_colormap(categories)
    if "negative" not in category_colormap:
        category_colormap["negative"] = (0.6, 0.6, 0.6)

    track_points = tracks_dataframe.copy()
    track_points["frame"] = track_points["t"].astype(int)
    track_points["label_id"] = track_points["track_id"].astype(int) + 1
    category_lookup = snapshot_dataframe[["label_id", "frame", "category"]].copy()
    merged_tracks = track_points.merge(category_lookup, on=["label_id", "frame"], how="left")
    merged_tracks["category"] = merged_tracks["category"].fillna("negative")

    figure, axis = plt.subplots(figsize=(8, 6))
    all_x_coordinates = []
    all_y_coordinates = []
    for label_id, track_group in merged_tracks.groupby("label_id"):
        track_group = track_group.sort_values("frame")
        coordinates = track_group[["x", "y"]].to_numpy(dtype=float)
        if len(coordinates) < 2:
            continue
        all_x_coordinates.extend(coordinates[:, 0].tolist())
        all_y_coordinates.extend(coordinates[:, 1].tolist())
        segments = np.stack([coordinates[:-1], coordinates[1:]], axis=1)
        segment_colours = [category_colormap.get(category, (0.6, 0.6, 0.6)) for category in track_group["category"].iloc[:-1]]
        axis.add_collection(LineCollection(segments, colors=segment_colours, linewidths=1.5, alpha=0.85))

    if "parent_track_id" in tracks_dataframe.columns:
        track_endpoints = {}
        for track_id_value, track_group in merged_tracks.groupby("track_id"):
            track_group = track_group.sort_values("frame")
            track_endpoints[int(track_id_value)] = {"first_xy": (float(track_group.iloc[0]["x"]), float(track_group.iloc[0]["y"])), "last_xy": (float(track_group.iloc[-1]["x"]), float(track_group.iloc[-1]["y"]))}
        for _, daughter_row in merged_tracks.drop_duplicates("track_id").iterrows():
            parent_track_id_value = daughter_row.get("parent_track_id")
            if pd.isna(parent_track_id_value):
                continue
            parent_track_id_value = int(parent_track_id_value)
            child_track_id = int(daughter_row["track_id"])
            if parent_track_id_value in track_endpoints and child_track_id in track_endpoints:
                parent_x, parent_y = track_endpoints[parent_track_id_value]["last_xy"]
                child_x, child_y = track_endpoints[child_track_id]["first_xy"]
                axis.plot([parent_x, child_x], [parent_y, child_y], color="black", linewidth=1.0, alpha=0.5, linestyle="--", zorder=0)

    if all_x_coordinates and all_y_coordinates:
        axis.set_xlim(min(all_x_coordinates) - 10, max(all_x_coordinates) + 10)
        axis.set_ylim(min(all_y_coordinates) - 10, max(all_y_coordinates) + 10)
    axis.invert_yaxis()
    axis.set(xlabel="x", ylabel="y", title=title, aspect="equal")
    legend_handles = [plt.Line2D([0], [0], color=category_colormap[category], lw=2, label=category) for category in sorted(category_colormap.keys())]
    axis.legend(handles=legend_handles, title="Snapshot category", loc="best")
    plt.tight_layout()
    return figure, axis


def plot_snapshot_cell_timelines(snapshot_dataframe, tracks_dataframe=None, title="Cell Status Over Time"):
    """Horizontal bar chart showing each cell's category across all frames.

    Each row is a cell, each column is a frame, and the colour of each
    bar is the cell's snapshot category in that frame.

    If lineage information is available (``lineage_id`` and
    ``parent_track_id`` columns in ``tracks_dataframe``) cells are sorted
    by lineage and dashed vertical lines are drawn at each daughter
    track's first *tracked* frame (i.e. the actual division frame as
    detected by TrackMate, not the daughter's first snapshot frame).

    Parameters
    ----------
    snapshot_dataframe : pandas.DataFrame
        Output of :func:`fate_assignment.assign_snapshot_fates`.
    tracks_dataframe : pandas.DataFrame or None, default None
        TrackMate tracks CSV. If it contains ``lineage_id`` and
        ``parent_track_id`` columns, lineage sorting and division lines
        are enabled.
    title : str
        Plot title.

    Returns
    -------
    figure, axis : matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    if snapshot_dataframe is None or len(snapshot_dataframe) == 0:
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.set_title(title)
        axis.text(0.5, 0.5, "No snapshot data", ha="center", va="center", transform=axis.transAxes)
        axis.axis("off")
        plt.tight_layout()
        return figure, axis

    categories = sorted(snapshot_dataframe["category"].unique())
    category_colormap = build_category_colormap(categories)
    has_lineage = (tracks_dataframe is not None and len(tracks_dataframe) > 0 and "lineage_id" in tracks_dataframe.columns)

    if has_lineage:
        tracks_with_label = tracks_dataframe.copy()
        tracks_with_label["label_id"] = tracks_with_label["track_id"].astype(int) + 1
        lineage_lookup = tracks_with_label.drop_duplicates("label_id")[["label_id", "lineage_id", "parent_track_id"]].set_index("label_id")
        # Actual division frame for each daughter = the daughter's first
        # frame as recorded by TrackMate (not its first snapshot frame).
        first_tracked_frame_per_label = tracks_with_label.groupby("label_id")["t"].min().astype(int).to_dict()
        cell_ids = sorted(snapshot_dataframe["label_id"].unique(), key=lambda label_id: tuple(int(part) if part.isdigit() else 0 for part in str(lineage_lookup.loc[label_id, "lineage_id"] if label_id in lineage_lookup.index else label_id).split(".")))
        label_display = {label_id: (str(lineage_lookup.loc[label_id, "lineage_id"]) if label_id in lineage_lookup.index else str(label_id)) for label_id in cell_ids}
    else:
        cell_ids = sorted(snapshot_dataframe["label_id"].unique())
        label_display = {label_id: str(label_id) for label_id in cell_ids}

    num_cells = len(cell_ids)
    cell_y_position = {cell_id: row_index for row_index, cell_id in enumerate(cell_ids)}
    minimum_frame = int(snapshot_dataframe["frame"].min())
    maximum_frame = int(snapshot_dataframe["frame"].max())

    plot_height = max(3, min(num_cells * 0.25 + 1, 40))
    figure, axis = plt.subplots(figsize=(max(8, (maximum_frame - minimum_frame) * 0.15 + 2), plot_height))

    for _, row in snapshot_dataframe.iterrows():
        axis.barh(cell_y_position[row["label_id"]], width=1, left=int(row["frame"]), height=0.8, color=category_colormap.get(row["category"], (0.5, 0.5, 0.5)), edgecolor="none", linewidth=0)

    if has_lineage:
        for label_id in cell_ids:
            if label_id not in lineage_lookup.index:
                continue
            parent_track_id_value = lineage_lookup.loc[label_id, "parent_track_id"]
            if pd.isna(parent_track_id_value):
                continue
            parent_label_id = int(parent_track_id_value) + 1
            if parent_label_id not in cell_y_position:
                continue
            division_frame = first_tracked_frame_per_label.get(label_id)
            if division_frame is None:
                continue
            axis.plot([division_frame, division_frame], [cell_y_position[parent_label_id], cell_y_position[label_id]], color="black", linewidth=0.8, alpha=0.6, linestyle="--")

    axis.set_xlim(minimum_frame - 0.5, maximum_frame + 1.5)
    axis.set_ylim(-0.5, num_cells - 0.5)
    axis.set_xlabel("Frame", fontsize=11)
    axis.set_ylabel("Cell (lineage ID)" if has_lineage else "Cell", fontsize=11)
    axis.set_title(title, fontsize=12)
    if num_cells <= 60:
        axis.set_yticks(range(num_cells))
        axis.set_yticklabels([label_display[cell_id] for cell_id in cell_ids], fontsize=max(4, 8 - num_cells // 20))
    else:
        axis.set_yticks([])
    legend_handles = [Patch(facecolor=category_colormap[category], edgecolor="none", label=category) for category in categories]
    axis.legend(handles=legend_handles, title="Category", loc="upper right", fontsize=8, title_fontsize=9, framealpha=0.8)
    figure.tight_layout()
    return figure, axis
