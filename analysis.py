"""Fluorescence fate analysis + plotting for the FluoroFate GUI.

Single entry point ``run_analysis`` consumes the in-memory arrays the GUI
already has (the original image, the tracked/segmented labels and, when
available, the TrackMate tracks), writes the per-cell CSV and every
percentage / trajectory / timeline figure the old ``fluorofate.py`` GUI
produced, and returns a summary record plus napari layer specs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from skimage.filters import (gaussian, threshold_mean, threshold_minimum,
                             threshold_otsu, threshold_triangle, threshold_yen)
from skimage.measure import label, regionprops

THRESHOLD_METHODS = {
    "otsu": threshold_otsu,
    "mean": threshold_mean,
    "yen": threshold_yen,
    "triangle": threshold_triangle,
    "minimum": threshold_minimum,
}

SAVE_EXTENSIONS = {"PNG": "png", "JPEG": "jpeg", "SVG": "svg", "PDF": "pdf"}


def build_category_colormap(categories, fluorophore_colours):
    """Colour per dynamic category: single fluorophores use their channel colour, compounds are averaged, negative is grey."""
    category_colormap = {}
    for category in categories:
        if category == "negative":
            category_colormap[category] = "gray"
            continue
        component_rgbs = [mcolors.to_rgb(fluorophore_colours[part]) for part in category.split("+")]
        category_colormap[category] = tuple(np.mean(component_rgbs, axis=0))
    return category_colormap


def measure_all_cells_in_frame(label_image):
    """Return {cell_id: (area_px, roundness)} for every cell in one label image."""
    measurements = {}
    for region in regionprops(label_image):
        roundness = region.minor_axis_length / region.major_axis_length if region.major_axis_length > 0 else np.nan
        measurements[region.label] = (region.area, roundness)
    return measurements


def compute_cell_positivity(linked_labels, positive_label_stacks, fluorophore_names):
    """Assign fluorescence blobs to tracked cells per frame by majority pixel overlap.

    Returns (frame_cell_positive_area, positive_cell_labels), where
    frame_cell_positive_area[name][frame_index][cell_id] is the positive pixel area
    and positive_cell_labels[name] is a label stack of the cells positive in each frame.
    """
    num_frames = linked_labels.shape[0]
    frame_cell_positive_area = {}
    positive_cell_labels = {}
    for fluorophore_name in fluorophore_names:
        frame_cell_positive_area[fluorophore_name] = {}
        per_frame_positive_labels = np.zeros_like(linked_labels)
        for frame_index in range(num_frames):
            positive_labels_frame = positive_label_stacks[fluorophore_name][frame_index]
            linked_labels_frame = linked_labels[frame_index]
            cell_positive_area = {}
            for region in regionprops(positive_labels_frame):
                cell_labels_under_blob = linked_labels_frame[region.coords[:, 0], region.coords[:, 1]]
                non_background = cell_labels_under_blob[cell_labels_under_blob > 0]
                if non_background.size == 0:
                    continue
                unique_cell_ids, pixel_counts = np.unique(non_background, return_counts=True)
                winning_cell_id = int(unique_cell_ids[int(np.argmax(pixel_counts))])
                cell_positive_area[winning_cell_id] = cell_positive_area.get(winning_cell_id, 0) + int(region.area)
            frame_cell_positive_area[fluorophore_name][frame_index] = cell_positive_area
            if cell_positive_area:
                positive_ids = np.fromiter(cell_positive_area.keys(), dtype=np.uint32)
                is_positive = np.isin(linked_labels_frame, positive_ids)
                per_frame_positive_labels[frame_index] = np.where(is_positive, linked_labels_frame, 0)
        positive_cell_labels[fluorophore_name] = per_frame_positive_labels
    return frame_cell_positive_area, positive_cell_labels


def compute_per_cell_intensity_area(linked_labels, fluorophore_stacks):
    """Per-(frame, cell) area and summed raw intensity for each fluorophore channel."""
    fluorophore_names = list(fluorophore_stacks.keys())
    num_frames = linked_labels.shape[0]
    intensity_columns = [f"{name}_total_intensity" for name in fluorophore_names]
    rows = []
    for frame_index in range(num_frames):
        frame_labels = linked_labels[frame_index]
        flat_labels = frame_labels.ravel()
        max_label = int(flat_labels.max()) if flat_labels.size else 0
        if max_label == 0:
            continue
        areas = np.bincount(flat_labels, minlength=max_label + 1)
        intensity_sums = {}
        for fluorophore_name in fluorophore_names:
            channel_pixels = fluorophore_stacks[fluorophore_name][frame_index].ravel().astype(np.float64)
            intensity_sums[fluorophore_name] = np.bincount(flat_labels, weights=channel_pixels, minlength=max_label + 1)
        present_cell_ids = np.nonzero(areas[1:])[0] + 1
        for cell_id in present_cell_ids:
            row = {"frame": int(frame_index), "cell_id": int(cell_id), "area_px": int(areas[cell_id])}
            for fluorophore_name in fluorophore_names:
                row[f"{fluorophore_name}_total_intensity"] = float(intensity_sums[fluorophore_name][cell_id])
            rows.append(row)
    return pd.DataFrame(rows, columns=["frame", "cell_id", "area_px"] + intensity_columns)


def assign_persistent_fates(linked_labels, frame_cell_positive_area):
    """Give each cell a permanent fate = the fluorophore it turns positive for first.

    Returns (fates_dataframe, locked_labels, per_frame_dataframe). locked_labels[name]
    paints each fated cell from its first-positive frame onwards.
    """
    num_frames = linked_labels.shape[0]
    fluorophore_names = list(frame_cell_positive_area.keys())

    frame_measurements = {frame_index: measure_all_cells_in_frame(linked_labels[frame_index]) for frame_index in range(num_frames)}
    all_label_ids = sorted({label_id for measurements in frame_measurements.values() for label_id in measurements})

    summary_rows = []
    per_frame_rows = []
    for label_id in all_label_ids:
        first_positive_frame_per_fluorophore = {fluorophore_name: None for fluorophore_name in fluorophore_names}
        cumulative_positive_area = {fluorophore_name: 0 for fluorophore_name in fluorophore_names}
        cell_areas = []
        cell_roundnesses = []
        for frame_index in range(num_frames):
            if label_id not in frame_measurements[frame_index]:
                continue
            area, roundness = frame_measurements[frame_index][label_id]
            cell_areas.append(area)
            cell_roundnesses.append(roundness)
            per_frame_record = {"label_id": label_id, "frame": frame_index, "area": area, "roundness": roundness}
            for fluorophore_name in fluorophore_names:
                positive_area = frame_cell_positive_area[fluorophore_name][frame_index].get(label_id, 0)
                cumulative_positive_area[fluorophore_name] += positive_area
                per_frame_record[f"{fluorophore_name}_positive_area"] = positive_area
                if first_positive_frame_per_fluorophore[fluorophore_name] is None and positive_area > 0:
                    first_positive_frame_per_fluorophore[fluorophore_name] = frame_index
            per_frame_rows.append(per_frame_record)

        fate = "negative"
        first_positive_frame = np.nan
        earliest_frame = num_frames + 1
        for fluorophore_name in fluorophore_names:
            candidate = first_positive_frame_per_fluorophore[fluorophore_name]
            if candidate is not None and candidate < earliest_frame:
                earliest_frame = candidate
                fate = fluorophore_name
                first_positive_frame = candidate

        summary_row = {"label_id": label_id, "mean_area": float(np.mean(cell_areas)) if cell_areas else np.nan, "mean_roundness": float(np.nanmean(cell_roundnesses)) if cell_roundnesses else np.nan}
        for fluorophore_name in fluorophore_names:
            summary_row[f"first_{fluorophore_name}_frame"] = first_positive_frame_per_fluorophore[fluorophore_name] if first_positive_frame_per_fluorophore[fluorophore_name] is not None else np.nan
            summary_row[f"{fluorophore_name}_positive_area"] = cumulative_positive_area[fluorophore_name]
        summary_row["first_positive_frame"] = first_positive_frame
        summary_row["fate"] = fate
        summary_rows.append(summary_row)

    fates_dataframe = pd.DataFrame(summary_rows).sort_values(["fate", "first_positive_frame", "label_id"]).reset_index(drop=True)

    locked_labels = {}
    for fluorophore_name in fluorophore_names:
        fate_rows = fates_dataframe[fates_dataframe["fate"] == fluorophore_name]
        stack = np.zeros_like(linked_labels, dtype=np.uint32)
        first_frame_column = f"first_{fluorophore_name}_frame"
        for fate_row_index, fate_row in fate_rows.iterrows():
            cell_id = int(fate_row["label_id"])
            first_positive_frame = fate_row[first_frame_column]
            if pd.isna(first_positive_frame):
                continue
            first_positive_frame = int(first_positive_frame)
            frames_slice = linked_labels[first_positive_frame:]
            stack[first_positive_frame:] = np.where(frames_slice == cell_id, cell_id, stack[first_positive_frame:])
        locked_labels[fluorophore_name] = stack

    per_frame_dataframe = pd.DataFrame(per_frame_rows)
    if len(per_frame_dataframe) > 0:
        fate_lookup = fates_dataframe.set_index("label_id")["fate"]
        per_frame_dataframe["fate"] = per_frame_dataframe["label_id"].map(fate_lookup)
        per_frame_dataframe = per_frame_dataframe.sort_values(["label_id", "frame"]).reset_index(drop=True)

    return fates_dataframe, locked_labels, per_frame_dataframe


def assign_dynamic_fates(linked_labels, frame_cell_positive_area):
    """Classify every cell independently in every frame; category joins positive fluorophores with '+'."""
    fluorophore_names = list(frame_cell_positive_area.keys())
    dynamic_rows = []
    for frame_index in range(linked_labels.shape[0]):
        cell_shapes = measure_all_cells_in_frame(linked_labels[frame_index])
        for label_id, (area, roundness) in cell_shapes.items():
            is_positive = {fluorophore_name: frame_cell_positive_area[fluorophore_name][frame_index].get(label_id, 0) > 0 for fluorophore_name in fluorophore_names}
            category = "+".join(fluorophore_name for fluorophore_name in fluorophore_names if is_positive[fluorophore_name]) or "negative"
            dynamic_row = {"label_id": label_id, "frame": frame_index, "area": area, "roundness": roundness}
            dynamic_row.update(is_positive)
            for fluorophore_name in fluorophore_names:
                dynamic_row[f"{fluorophore_name}_positive_area"] = frame_cell_positive_area[fluorophore_name][frame_index].get(label_id, 0)
            dynamic_row["category"] = category
            dynamic_rows.append(dynamic_row)
    return pd.DataFrame(dynamic_rows)


def compute_persistent_percentages(assignments_dataframe, num_frames, fluorophore_names):
    """Cumulative percent-positive cells per frame (monotonic, persistent mode)."""
    total_cells = len(assignments_dataframe)
    if total_cells == 0:
        raise ValueError("No tracked cells available for persistent percentage computation.")
    frames = np.arange(num_frames)
    columns = {"frame": frames}
    total_positive = np.zeros(num_frames)
    for fluorophore_name in fluorophore_names:
        first_positive_frames = assignments_dataframe.loc[assignments_dataframe["fate"] == fluorophore_name, "first_positive_frame"].dropna().to_numpy()
        cumulative_counts = np.array([np.sum(first_positive_frames <= frame_index) for frame_index in frames])
        percentages = 100.0 * cumulative_counts / total_cells
        columns[f"{fluorophore_name}_pct"] = percentages
        total_positive += percentages
    columns["total_positive_pct"] = total_positive
    return pd.DataFrame(columns)


def compute_dynamic_percentages(dynamic_dataframe, num_frames):
    """Percent-cells per category per frame (dynamic mode). Returns (summary_dataframe, categories)."""
    categories = sorted(dynamic_dataframe["category"].unique(), key=lambda category: (category == "negative", category))
    counts = dynamic_dataframe.groupby(["frame", "category"]).size().unstack(fill_value=0)
    totals_per_frame = counts.sum(axis=1)
    percentages = counts.div(totals_per_frame, axis=0) * 100.0
    percentages = percentages.reindex(range(num_frames), fill_value=0.0)
    columns = {"frame": np.arange(num_frames)}
    for category in categories:
        columns[f"{category}_pct"] = percentages[category].values if category in percentages.columns else np.zeros(num_frames)
    return pd.DataFrame(columns), categories


def filter_by_frame_presence(tracks_dataframe, dynamic_dataframe, num_frames, minimum_percentage):
    """Drop cells present in fewer than minimum_percentage% of frames (dynamic mode)."""
    minimum_frame_count = num_frames * minimum_percentage / 100.0
    frame_counts = dynamic_dataframe.groupby("label_id")["frame"].nunique()
    keep_label_ids = frame_counts[frame_counts >= minimum_frame_count].index
    filtered_dynamic_dataframe = dynamic_dataframe[dynamic_dataframe["label_id"].isin(keep_label_ids)].copy()
    if tracks_dataframe is not None and len(tracks_dataframe) > 0:
        keep_track_ids = keep_label_ids.astype(int) - 1
        filtered_tracks_dataframe = tracks_dataframe[tracks_dataframe["track_id"].isin(keep_track_ids)].copy()
    else:
        filtered_tracks_dataframe = tracks_dataframe
    return filtered_tracks_dataframe, filtered_dynamic_dataframe


def filter_persistent_by_frame_presence(assignments_dataframe, linked_labels, num_frames, minimum_percentage):
    """Drop cells present in fewer than minimum_percentage% of frames (persistent mode)."""
    minimum_frame_count = num_frames * minimum_percentage / 100.0
    frame_presence = {}
    for frame_index in range(linked_labels.shape[0]):
        for cell_id in np.unique(linked_labels[frame_index]):
            if cell_id > 0:
                frame_presence[cell_id] = frame_presence.get(cell_id, 0) + 1
    keep_label_ids = [cell_id for cell_id, count in frame_presence.items() if count >= minimum_frame_count]
    return assignments_dataframe[assignments_dataframe["label_id"].isin(keep_label_ids)].copy()


def plot_persistent_percentages(summary_dataframe, fluorophore_names, fluorophore_colours, title="Persistent Positive Cells Over Time"):
    """Line plot of cumulative percent-positive cells per fluorophore plus a total (persistent mode)."""
    figure, axis = plt.subplots(figsize=(8, 4))
    for fluorophore_name in fluorophore_names:
        sns.lineplot(data=summary_dataframe, x="frame", y=f"{fluorophore_name}_pct", color=fluorophore_colours[fluorophore_name], label=fluorophore_name, ax=axis)
    sns.lineplot(data=summary_dataframe, x="frame", y="total_positive_pct", color="black", label="Total", ax=axis)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set(xlabel="Frame", ylabel="% Cells", ylim=(0, 100), title=title)
    plt.tight_layout()
    return figure, axis


def plot_dynamic_percentages(summary_dataframe, categories, fluorophore_colours, title="Per-Frame Categories Over Time"):
    """Line plot of percent-cells per category over frames (dynamic mode)."""
    figure, axis = plt.subplots(figsize=(8, 4))
    category_colormap = build_category_colormap(categories, fluorophore_colours)
    for category in categories:
        sns.lineplot(data=summary_dataframe, x="frame", y=f"{category}_pct", color=category_colormap[category], label=category, ax=axis)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set(xlabel="Frame", ylabel="% Cells", ylim=(0, 100), title=title)
    plt.tight_layout()
    return figure, axis


def plot_dynamic_trajectories(tracks_dataframe, dynamic_dataframe, fluorophore_colours, title="dynamic Trajectories by Category"):
    """Plot cell XY tracks coloured by dynamic category, with dashed parent-daughter links when available."""
    if tracks_dataframe is None or len(tracks_dataframe) == 0:
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.set_title(title)
        axis.text(0.5, 0.5, "No tracks available", ha="center", va="center")
        axis.axis("off")
        plt.tight_layout()
        return figure, axis

    categories = sorted(dynamic_dataframe["category"].unique())
    category_colormap = build_category_colormap(categories, fluorophore_colours)
    if "negative" not in category_colormap:
        category_colormap["negative"] = "gray"

    track_points = tracks_dataframe.copy()
    track_points["frame"] = track_points["t"].astype(int)
    track_points["label_id"] = track_points["track_id"].astype(int) + 1
    category_lookup = dynamic_dataframe[["label_id", "frame", "category"]].copy()
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
        for daughter_index, daughter_row in merged_tracks.drop_duplicates("track_id").iterrows():
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
    axis.legend(handles=legend_handles, title="dynamic category", loc="best")
    plt.tight_layout()
    return figure, axis


def plot_dynamic_cell_timelines(dynamic_dataframe, fluorophore_colours, tracks_dataframe=None, title="Cell Status Over Time"):
    """Horizontal-bar timeline of each cell's dynamic category per frame, sorted by lineage when available."""
    if dynamic_dataframe is None or len(dynamic_dataframe) == 0:
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.set_title(title)
        axis.text(0.5, 0.5, "No dynamic data", ha="center", va="center", transform=axis.transAxes)
        axis.axis("off")
        plt.tight_layout()
        return figure, axis

    categories = sorted(dynamic_dataframe["category"].unique())
    category_colormap = build_category_colormap(categories, fluorophore_colours)
    has_lineage = (tracks_dataframe is not None and len(tracks_dataframe) > 0 and "lineage_id" in tracks_dataframe.columns)

    if has_lineage:
        tracks_with_label = tracks_dataframe.copy()
        tracks_with_label["label_id"] = tracks_with_label["track_id"].astype(int) + 1
        lineage_lookup = tracks_with_label.drop_duplicates("label_id")[["label_id", "lineage_id", "parent_track_id"]].set_index("label_id")
        first_tracked_frame_per_label = tracks_with_label.groupby("label_id")["t"].min().astype(int).to_dict()
        cell_ids = sorted(dynamic_dataframe["label_id"].unique(), key=lambda label_id: tuple(int(part) if part.isdigit() else 0 for part in str(lineage_lookup.loc[label_id, "lineage_id"] if label_id in lineage_lookup.index else label_id).split(".")))
        label_display = {label_id: (str(lineage_lookup.loc[label_id, "lineage_id"]) if label_id in lineage_lookup.index else str(label_id)) for label_id in cell_ids}
    else:
        cell_ids = sorted(dynamic_dataframe["label_id"].unique())
        label_display = {label_id: str(label_id) for label_id in cell_ids}

    num_cells = len(cell_ids)
    cell_y_position = {cell_id: row_index for row_index, cell_id in enumerate(cell_ids)}
    minimum_frame = int(dynamic_dataframe["frame"].min())
    maximum_frame = int(dynamic_dataframe["frame"].max())

    plot_height = max(3, min(num_cells * 0.25 + 1, 40))
    figure, axis = plt.subplots(figsize=(max(8, (maximum_frame - minimum_frame) * 0.15 + 2), plot_height))

    for row_index, row in dynamic_dataframe.iterrows():
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


def run_analysis(original_image, linked_labels, tracks_df, channel_settings,is_2d, analysis_type,
                 output_directory, file_stem, save_options, frame_presence_threshold, progress_callback=None):
    """Run the full persistent + dynamic fate analysis for one image.

    Parameters
    ----------
    original_image : numpy.ndarray, shape (T, C, Y, X)
        The raw multi-channel image.
    linked_labels : numpy.ndarray, shape (T, Y, X), integer dtype
        Tracked (or, for single frames / no tracking, segmented) cell labels.
    tracks_df : pandas.DataFrame or None
        TrackMate tracks (columns ``track_id, t, x, y`` and optionally the
        lineage columns). ``None`` when no tracking was performed.
    channel_settings : dict[int, dict]
        ``{channel_index: {"name", "colour", "threshold"}}`` for each
        fluorescent channel. ``threshold`` is a method name string
        (e.g. ``"Otsu"``) or a numeric value.
    output_directory : str or pathlib.Path
        Folder to write the CSVs and figures into (created if needed).
    file_stem : str
        Name used for plot titles.
    is_2d : bool
        Whether the image is 2D (single frame) or 3D (time-lapse).
    analysis_type : str
        Type of analysis to perform ("persistent", "dynamic", or "both").
    frame_presence_threshold : float
        Minimum fraction of frames a cell must be present in to be included in the analysis.
    progress_callback : callable or None
        Called as ``progress_callback(fraction, message)`` (fraction 0-1).

    Returns
    -------
    dict
        Keys ``"summary_record"`` (dict), ``"layer_specs"`` (list of
        ``{"name", "labels", "colour"}`` for napari) and
        ``"per_frame_cells"`` (the consolidated DataFrame).
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # File extensions for every figure format the user ticked (e.g. "PNG" -> "png").
    save_extensions = [SAVE_EXTENSIONS[name] for name in save_options if name in SAVE_EXTENSIONS]

    def report(fraction, message):
        if progress_callback is not None:
            progress_callback(fraction, message)

    # Fluorophore channels in ascending channel order.
    channels = sorted(channel_settings)
    fluorophore_names = [channel_settings[channel]["name"] for channel in channels]
    fluorophore_colours = {channel_settings[channel]["name"]: channel_settings[channel]["colour"] for channel in channels}
    fluorophore_thresholds = {channel_settings[channel]["name"]: channel_settings[channel]["threshold"] for channel in channels}
    num_frames = linked_labels.shape[0]

    # --- Threshold each fluorophore channel into positive-blob labels ---
    report(0.05, "Thresholding fluorescence...")
    fluorophore_stacks = {}
    positive_label_stacks = {}
    for channel in channels:
        name = channel_settings[channel]["name"]
        threshold = channel_settings[channel]["threshold"]
        stack = original_image[:, channel].astype(np.float64)
        fluorophore_stacks[name] = stack
        blurred_stack = np.stack([gaussian(frame, sigma=1, preserve_range=True) for frame in stack], axis=0)
        blob_labels = np.zeros(blurred_stack.shape, dtype=np.uint32)
        for frame_index in range(num_frames):
            if isinstance(threshold, str):
                threshold_value = THRESHOLD_METHODS[threshold.lower()](blurred_stack[frame_index])
            else:
                threshold_value = float(threshold)
            blob_labels[frame_index] = label(blurred_stack[frame_index] > threshold_value).astype(np.uint32)
        positive_label_stacks[name] = blob_labels

    report(0.25, "Assigning blobs to cells...")
    frame_cell_positive_area, positive_cell_labels = compute_cell_positivity(linked_labels, positive_label_stacks, fluorophore_names)

    layer_specs = []
    mode = analysis_type.value.lower()
    label_text = f"cells in \u2265{frame_presence_threshold}% of frames"

    # --- Persistent mode ---
    if mode == "persistent":
        report(0.5, "Persistent fate assignment...")
        persistent_fates_df, locked_labels, per_frame_fates_df = assign_persistent_fates(linked_labels, frame_cell_positive_area)
        persistent_fates_df = persistent_fates_df.sort_values("label_id").reset_index(drop=True)
        # Kept for the batch summary record; the unfiltered CSV/plot are intentionally not written.
        persistent_summary_df = compute_persistent_percentages(persistent_fates_df, num_frames, fluorophore_names)

        # Only the frame-presence-filtered variant is saved (cells present in >= N% of frames).
        report(0.75, "Frame-presence-filtered plots...")
        persistent_filtered = filter_persistent_by_frame_presence(persistent_fates_df, linked_labels, num_frames, frame_presence_threshold)
        if len(persistent_filtered) > 0:
            persistent_summary_filtered = compute_persistent_percentages(persistent_filtered, num_frames, fluorophore_names)
            persistent_summary_filtered.to_csv(output_directory / f"percentages_persistent_cells_in_{frame_presence_threshold}_pct_frames.csv", index=False)
            figure, plot_axis = plot_persistent_percentages(persistent_summary_filtered, fluorophore_names, fluorophore_colours, title=f"{file_stem} — persistent ({label_text}, n={len(persistent_filtered)})")
            for extension in save_extensions:
                figure.savefig(str(output_directory / f"percentages_persistent_cells_in_{frame_presence_threshold}_pct_frames.{extension}"), bbox_inches="tight")
            plt.close(figure)

        # Only the persistent positive layers belong in a persistent run.
        for name, locked_label_image in locked_labels.items():
            layer_specs.append({"name": f"Persistent: {name} positive", "labels": locked_label_image, "colour": fluorophore_colours[name]})

    # --- dynamic mode ---
    elif mode == "dynamic":
        report(0.5, "dynamic fate assignment...")
        dynamic_df = assign_dynamic_fates(linked_labels, frame_cell_positive_area).sort_values(["label_id", "frame"]).reset_index(drop=True)

        # Only the frame-presence-filtered variants are saved (cells present in >= N% of frames).
        report(0.75, "Frame-presence-filtered plots...")
        tracks_filtered, dynamic_filtered = filter_by_frame_presence(tracks_df, dynamic_df, num_frames, frame_presence_threshold)
        if len(dynamic_filtered) > 0:
            dynamic_summary_filtered, dynamic_categories_filtered = compute_dynamic_percentages(dynamic_filtered, num_frames)
            dynamic_summary_filtered.to_csv(output_directory / f"percentages_dynamic_cells_in_{frame_presence_threshold}_pct_frames.csv", index=False)
            figure, plot_axis = plot_dynamic_percentages(dynamic_summary_filtered, dynamic_categories_filtered, fluorophore_colours, title=f"{file_stem} — dynamic ({label_text})")
            for extension in save_extensions:
                figure.savefig(str(output_directory / f"percentages_dynamic_cells_in_{frame_presence_threshold}_pct_frames.{extension}"), bbox_inches="tight")
            plt.close(figure)
            figure, plot_axis = plot_dynamic_trajectories(tracks_filtered, dynamic_filtered, fluorophore_colours, title=f"{file_stem} — dynamic trajectories ({label_text})")
            for extension in save_extensions:
                figure.savefig(str(output_directory / f"dynamic_trajectories_cells_in_{frame_presence_threshold}_pct_frames.{extension}"), bbox_inches="tight")
            plt.close(figure)
            figure, plot_axis = plot_dynamic_cell_timelines(dynamic_filtered, fluorophore_colours, tracks_dataframe=tracks_filtered, title=f"{file_stem} — cell timelines ({label_text})")
            for extension in save_extensions:
                figure.savefig(str(output_directory / f"dynamic_timelines_cells_in_{frame_presence_threshold}_pct_frames.{extension}"), bbox_inches="tight")
            plt.close(figure)

        # Only the dynamic positive layers belong in a dynamic run.
        for name, positive_label_image in positive_cell_labels.items():
            layer_specs.append({"name": f"dynamic: {name} positive", "labels": positive_label_image, "colour": fluorophore_colours[name]})

    # --- Consolidated per-(frame, cell) CSV ---
    report(0.9, "Writing per-cell CSV...")
    per_frame_cells_df = compute_per_cell_intensity_area(linked_labels, fluorophore_stacks)
    for name in fluorophore_names:
        area_lookup = frame_cell_positive_area[name]
        per_frame_cells_df[f"Thresholded {name} Area (Pixels)"] = [
            int(area_lookup.get(int(frame), {}).get(int(cell), 0))
            for frame, cell in zip(per_frame_cells_df["frame"], per_frame_cells_df["cell_id"])
        ]
    if mode == "persistent":
        fate_by_cell = persistent_fates_df.set_index("label_id")["fate"]
        mapped_fate = per_frame_cells_df["cell_id"].map(fate_by_cell)
    for name in fluorophore_names:
        if mode == "persistent":
            per_frame_cells_df[f"Persistently {name}?"] = np.where(mapped_fate.eq(name), "Y", "N")
        elif mode == "dynamic":
            per_frame_cells_df[f"dynamic {name}?"] = np.where(per_frame_cells_df[f"Thresholded {name} Area (Pixels)"] > 0, "Y", "N")
        per_frame_cells_df[f"{name} Threshold Method"] = str(fluorophore_thresholds[name])

    # Lineage columns come from the tracks table (track_id == cell_id - 1).
    lineage_columns = ["track_id", "lineage_id", "parent_track_id", "generation"]
    if tracks_df is not None and "lineage_id" in tracks_df.columns:
        lineage_lookup = tracks_df.drop_duplicates("track_id")[lineage_columns].copy()
        per_frame_cells_df["track_id"] = per_frame_cells_df["cell_id"].astype(int) - 1
        per_frame_cells_df = per_frame_cells_df.merge(lineage_lookup, on="track_id", how="left")
    else:
        per_frame_cells_df["track_id"] = pd.NA
        for lineage_column in ("lineage_id", "parent_track_id", "generation"):
            per_frame_cells_df[lineage_column] = pd.NA

    rename_map = {
        "frame": "Frame ID", "cell_id": "Cell ID", "area_px": "Cell Area (pixels)",
        "track_id": "Track ID", "lineage_id": "Lineage ID",
        "parent_track_id": "Parent Track ID", "generation": "Generation",
    }
    for name in fluorophore_names:
        rename_map[f"{name}_total_intensity"] = f"{name} Fluorescence (Sum)"
    per_frame_cells_df = per_frame_cells_df.rename(columns=rename_map)
    column_order = ["Frame ID", "Cell ID", "Track ID", "Lineage ID", "Parent Track ID", "Generation", "Cell Area (pixels)"]
    column_order += [f"{name} Fluorescence (Sum)" for name in fluorophore_names]
    column_order += [f"Thresholded {name} Area (Pixels)" for name in fluorophore_names]
    column_order += [f"{name} Threshold Method" for name in fluorophore_names]
    if mode == "persistent":
        column_order += [f"Persistently {name}?" for name in fluorophore_names]
    elif mode == "dynamic":
        column_order += [f"dynamic {name}?" for name in fluorophore_names]
    per_frame_cells_df = per_frame_cells_df[column_order]
    per_frame_cells_df.to_csv(output_directory / "per_frame_cells.csv", index=False)

    # --- Summary record ---
    summary_record = {
        "filename": file_stem,
        "n_frames": num_frames,
        "n_tracked_cells": int(tracks_df["track_id"].nunique()) if tracks_df is not None and len(tracks_df) > 0 else 0,
    }
    if mode == "persistent":
        summary_record["n_segmented_cells"] = int(len(persistent_fates_df))
        summary_record["n_negative_persistent"] = int((persistent_fates_df["fate"] == "negative").sum())
        summary_record["final_total_pct_persistent"] = float(persistent_summary_df["total_positive_pct"].iloc[-1])
        for name in fluorophore_names:
            summary_record[f"persistent_n_{name}"] = int((persistent_fates_df["fate"] == name).sum())
            summary_record[f"persistent_final_pct_{name}"] = float(persistent_summary_df[f"{name}_pct"].iloc[-1])
    elif mode == "dynamic":
        last_frame_dynamic = dynamic_df[dynamic_df["frame"] == num_frames - 1]
        last_frame_total = max(len(last_frame_dynamic), 1)
        for category in sorted(dynamic_df["category"].unique()):
            summary_record[f"dynamic_final_pct_{category}"] = 100.0 * (last_frame_dynamic["category"] == category).sum() / last_frame_total

    report(1.0, "Analysis complete")
    return {"summary_record": summary_record, "layer_specs": layer_specs, "per_frame_cells": per_frame_cells_df}
