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

from fate_assignment import (assign_persistent_fates, assign_snapshot_fates,
                             compute_persistent_percentages, compute_snapshot_percentages,
                             filter_by_frame_presence, filter_persistent_by_frame_presence)
from measurement import compute_cell_positivity, compute_per_cell_intensity_area
from plotting import (plot_persistent_percentages, plot_snapshot_cell_timelines,
                      plot_snapshot_percentages, plot_snapshot_trajectories)

THRESHOLD_METHODS = {
    "otsu": threshold_otsu,
    "mean": threshold_mean,
    "yen": threshold_yen,
    "triangle": threshold_triangle,
    "minimum": threshold_minimum,
}


def run_analysis(original_image, linked_labels, tracks_df, channel_settings,
                 output_directory, file_stem, blur_sigma=1.0,
                 frame_presence_thresholds=(40, 60, 80), progress_callback=None):
    """Run the full persistent + snapshot fate analysis for one image.

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
    blur_sigma : float
        Gaussian blur applied before thresholding fluorescence.
    frame_presence_thresholds : tuple[int, ...]
        Minimum %-of-frames filters to also produce plots for.
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
        blurred_stack = np.stack([gaussian(frame, sigma=blur_sigma, preserve_range=True) for frame in stack], axis=0)
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

    # --- Persistent mode ---
    report(0.4, "Persistent fate assignment...")
    persistent_fates_df, locked_labels, _ = assign_persistent_fates(linked_labels, frame_cell_positive_area)
    persistent_fates_df = persistent_fates_df.sort_values("label_id").reset_index(drop=True)
    persistent_summary_df = compute_persistent_percentages(persistent_fates_df, num_frames, fluorophore_names)
    persistent_summary_df.to_csv(output_directory / "percentages_persistent.csv", index=False)
    figure, _ = plot_persistent_percentages(persistent_summary_df, fluorophore_names, title=file_stem)
    figure.savefig(str(output_directory / "percentages_persistent.pdf"), bbox_inches="tight")
    plt.close(figure)

    # --- Snapshot mode ---
    report(0.6, "Snapshot fate assignment...")
    snapshot_df = assign_snapshot_fates(linked_labels, frame_cell_positive_area).sort_values(["label_id", "frame"]).reset_index(drop=True)
    snapshot_summary_df, snapshot_categories = compute_snapshot_percentages(snapshot_df, num_frames)
    snapshot_summary_df.to_csv(output_directory / "percentages_snapshot.csv", index=False)
    figure, _ = plot_snapshot_percentages(snapshot_summary_df, snapshot_categories, title=file_stem)
    figure.savefig(str(output_directory / "percentages_snapshot.pdf"), bbox_inches="tight")
    plt.close(figure)
    figure, _ = plot_snapshot_trajectories(tracks_df, snapshot_df, title=f"{file_stem} — snapshot trajectories")
    figure.savefig(str(output_directory / "snapshot_trajectories.pdf"), bbox_inches="tight")
    plt.close(figure)
    figure, _ = plot_snapshot_cell_timelines(snapshot_df, tracks_dataframe=tracks_df, title=f"{file_stem} — cell timelines")
    figure.savefig(str(output_directory / "snapshot_timelines.pdf"), bbox_inches="tight")
    plt.close(figure)

    # --- Frame-presence-filtered plot variants (cells present in >= N% of frames) ---
    report(0.75, "Frame-presence-filtered plots...")
    for min_pct in frame_presence_thresholds:
        suffix = f"min{min_pct}pct"
        label_text = f"\u2265{min_pct}% of frames"
        persistent_filtered = filter_persistent_by_frame_presence(persistent_fates_df, linked_labels, num_frames, min_pct)
        if len(persistent_filtered) > 0:
            persistent_summary_filtered = compute_persistent_percentages(persistent_filtered, num_frames, fluorophore_names)
            persistent_summary_filtered.to_csv(output_directory / f"percentages_persistent_{suffix}.csv", index=False)
            figure, _ = plot_persistent_percentages(persistent_summary_filtered, fluorophore_names, title=f"{file_stem} — persistent ({label_text}, n={len(persistent_filtered)})")
            figure.savefig(str(output_directory / f"percentages_persistent_{suffix}.pdf"), bbox_inches="tight")
            plt.close(figure)
        tracks_filtered, snapshot_filtered = filter_by_frame_presence(tracks_df, snapshot_df, num_frames, min_pct)
        if len(snapshot_filtered) > 0:
            snapshot_summary_filtered, snapshot_categories_filtered = compute_snapshot_percentages(snapshot_filtered, num_frames)
            snapshot_summary_filtered.to_csv(output_directory / f"percentages_snapshot_{suffix}.csv", index=False)
            figure, _ = plot_snapshot_percentages(snapshot_summary_filtered, snapshot_categories_filtered, title=f"{file_stem} — snapshot ({label_text})")
            figure.savefig(str(output_directory / f"percentages_snapshot_{suffix}.pdf"), bbox_inches="tight")
            plt.close(figure)
            figure, _ = plot_snapshot_trajectories(tracks_filtered, snapshot_filtered, title=f"{file_stem} — snapshot trajectories ({label_text})")
            figure.savefig(str(output_directory / f"snapshot_trajectories_{suffix}.pdf"), bbox_inches="tight")
            plt.close(figure)
            figure, _ = plot_snapshot_cell_timelines(snapshot_filtered, tracks_dataframe=tracks_filtered, title=f"{file_stem} — cell timelines ({label_text})")
            figure.savefig(str(output_directory / f"snapshot_timelines_{suffix}.pdf"), bbox_inches="tight")
            plt.close(figure)

    # --- Consolidated per-(frame, cell) CSV ---
    report(0.9, "Writing per-cell CSV...")
    per_frame_cells_df = compute_per_cell_intensity_area(linked_labels, fluorophore_stacks)
    for name in fluorophore_names:
        area_lookup = frame_cell_positive_area[name]
        per_frame_cells_df[f"Thresholded {name} Area (Pixels)"] = [
            int(area_lookup.get(int(frame), {}).get(int(cell), 0))
            for frame, cell in zip(per_frame_cells_df["frame"], per_frame_cells_df["cell_id"])
        ]
    fate_by_cell = persistent_fates_df.set_index("label_id")["fate"]
    mapped_fate = per_frame_cells_df["cell_id"].map(fate_by_cell)
    for name in fluorophore_names:
        per_frame_cells_df[f"Persistently {name}?"] = np.where(mapped_fate.eq(name), "Y", "N")
        per_frame_cells_df[f"Snapshot {name}?"] = np.where(per_frame_cells_df[f"Thresholded {name} Area (Pixels)"] > 0, "Y", "N")
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
    column_order += [f"Persistently {name}?" for name in fluorophore_names]
    column_order += [f"Snapshot {name}?" for name in fluorophore_names]
    per_frame_cells_df = per_frame_cells_df[column_order]
    per_frame_cells_df.to_csv(output_directory / "per_frame_cells.csv", index=False)

    # --- Summary record ---
    summary_record = {
        "filename": file_stem,
        "n_frames": num_frames,
        "n_segmented_cells": int(len(persistent_fates_df)),
        "n_tracked_cells": int(tracks_df["track_id"].nunique()) if tracks_df is not None and len(tracks_df) > 0 else 0,
        "n_negative_persistent": int((persistent_fates_df["fate"] == "negative").sum()),
        "final_total_pct_persistent": float(persistent_summary_df["total_positive_pct"].iloc[-1]),
    }
    for name in fluorophore_names:
        summary_record[f"persistent_n_{name}"] = int((persistent_fates_df["fate"] == name).sum())
        summary_record[f"persistent_final_pct_{name}"] = float(persistent_summary_df[f"{name}_pct"].iloc[-1])
    last_frame_snapshot = snapshot_df[snapshot_df["frame"] == num_frames - 1]
    last_frame_total = max(len(last_frame_snapshot), 1)
    for category in sorted(snapshot_df["category"].unique()):
        summary_record[f"snapshot_final_pct_{category}"] = 100.0 * (last_frame_snapshot["category"] == category).sum() / last_frame_total

    # --- Napari layers: one persistent + one snapshot layer per fluorophore ---
    layer_specs = []
    for name, locked_label_image in locked_labels.items():
        layer_specs.append({"name": f"Persistent: {name} positive", "labels": locked_label_image, "colour": fluorophore_colours[name]})
    for name, positive_label_image in positive_cell_labels.items():
        layer_specs.append({"name": f"Snapshot: {name} positive", "labels": positive_label_image, "colour": fluorophore_colours[name]})

    report(1.0, "Analysis complete")
    return {"summary_record": summary_record, "layer_specs": layer_specs, "per_frame_cells": per_frame_cells_df}
