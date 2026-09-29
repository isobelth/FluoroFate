"""Fluorescence fate analysis + plotting for the FluoroFate GUI.

Single entry point ``run_analysis`` consumes the in-memory arrays the GUI
already has (the original image, the tracked/segmented labels and, when
available, the TrackMate tracks), writes the per-cell CSV and every
percentage / trajectory / timeline figure the old ``fluorofate.py`` GUI
produced, and returns a summary record plus napari layer specs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.filters import (gaussian, threshold_mean, threshold_minimum,
                             threshold_otsu, threshold_triangle, threshold_yen)
from skimage.measure import label

from fate_assignment import (assign_persistent_fates, assign_dynamic_fates,
                             compute_persistent_percentages, compute_dynamic_percentages,
                             filter_by_frame_presence, filter_persistent_by_frame_presence)
from measurement import compute_cell_positivity, compute_per_cell_intensity_area
from plotting import (plot_persistent_percentages, plot_dynamic_cell_timelines,
                      plot_dynamic_percentages, plot_dynamic_trajectories)

THRESHOLD_METHODS = {
    "otsu": threshold_otsu,
    "mean": threshold_mean,
    "yen": threshold_yen,
    "triangle": threshold_triangle,
    "minimum": threshold_minimum,
}

SAVE_EXTENSIONS = {"PNG": "png", "JPEG": "jpeg", "SVG": "svg", "PDF": "pdf"}


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
        persistent_fates_df, locked_labels, _ = assign_persistent_fates(linked_labels, frame_cell_positive_area)
        persistent_fates_df = persistent_fates_df.sort_values("label_id").reset_index(drop=True)
        # Kept for the batch summary record; the unfiltered CSV/plot are intentionally not written.
        persistent_summary_df = compute_persistent_percentages(persistent_fates_df, num_frames, fluorophore_names)

        # Only the frame-presence-filtered variant is saved (cells present in >= N% of frames).
        report(0.75, "Frame-presence-filtered plots...")
        persistent_filtered = filter_persistent_by_frame_presence(persistent_fates_df, linked_labels, num_frames, frame_presence_threshold)
        if len(persistent_filtered) > 0:
            persistent_summary_filtered = compute_persistent_percentages(persistent_filtered, num_frames, fluorophore_names)
            persistent_summary_filtered.to_csv(output_directory / f"percentages_persistent_cells_in_{frame_presence_threshold}_pct_frames.csv", index=False)
            figure, _ = plot_persistent_percentages(persistent_summary_filtered, fluorophore_names, title=f"{file_stem} — persistent ({label_text}, n={len(persistent_filtered)})")
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
            figure, _ = plot_dynamic_percentages(dynamic_summary_filtered, dynamic_categories_filtered, title=f"{file_stem} — dynamic ({label_text})")
            for extension in save_extensions:
                figure.savefig(str(output_directory / f"percentages_dynamic_cells_in_{frame_presence_threshold}_pct_frames.{extension}"), bbox_inches="tight")
            plt.close(figure)
            figure, _ = plot_dynamic_trajectories(tracks_filtered, dynamic_filtered, title=f"{file_stem} — dynamic trajectories ({label_text})")
            for extension in save_extensions:
                figure.savefig(str(output_directory / f"dynamic_trajectories_cells_in_{frame_presence_threshold}_pct_frames.{extension}"), bbox_inches="tight")
            plt.close(figure)
            figure, _ = plot_dynamic_cell_timelines(dynamic_filtered, tracks_dataframe=tracks_filtered, title=f"{file_stem} — cell timelines ({label_text})")
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
