"""Fate assignment, percentage computation, and cell filtering.

Two complementary modes are supported:

* **Persistent** — once a cell turns positive for a fluorophore, it
  stays that way. Appropriate for irreversible markers (e.g. Annexin V,
  PI). One row per cell summarising its fate.
* **Snapshot** — the category of every cell is recomputed independently
  in every frame, so cells can switch categories over time. One row per
  ``(cell, frame)``.

The shared input is ``frame_cell_positive_area`` produced by
:func:`measurement.compute_cell_positivity`, with structure
``{fluorophore_name: {frame_index: {cell_label_id: positive_pixel_area}}}``.
"""

import numpy as np
import pandas as pd

from measurement import measure_all_cells_in_frame


def assign_persistent_fates(linked_labels, frame_cell_positive_area):
    """Assign each cell a permanent fate based on its first-positive fluorophore.

    For every cell the function records the first frame at which each
    fluorophore becomes positive (``positive_pixel_area > 0``). The
    fluorophore with the earliest first-positive frame wins and becomes
    the cell's fate. Cells that never turn positive get
    ``fate == "negative"``. Once assigned, a cell's fate does not change.

    Parameters
    ----------
    linked_labels : numpy.ndarray, shape (n_frames, H, W), integer dtype
        Tracked cell segmentation.
    frame_cell_positive_area : dict[str, dict[int, dict[int, int]]]
        Output of :func:`measurement.compute_cell_positivity`.

    Returns
    -------
    fates_dataframe : pandas.DataFrame
        One row per cell with columns ``label_id, mean_area,
        mean_roundness, first_<fluorophore>_frame, <fluorophore>_positive_area
        (cumulative), first_positive_frame, fate``.
    locked_labels : dict[str, numpy.ndarray]
        ``{fluorophore_name: (n_frames, H, W) uint32 labels}`` containing
        only the cells assigned to that fate.
    per_frame_dataframe : pandas.DataFrame
        Long-format table with one row per ``(cell, frame)`` including
        area, roundness, per-fluorophore positive area and the cell's fate.
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
        fate_cell_ids = fates_dataframe.loc[fates_dataframe["fate"] == fluorophore_name, "label_id"].to_numpy(dtype=np.uint32)
        locked_labels[fluorophore_name] = np.where(np.isin(linked_labels, fate_cell_ids), linked_labels, 0).astype(np.uint32)

    per_frame_dataframe = pd.DataFrame(per_frame_rows)
    if len(per_frame_dataframe) > 0:
        fate_lookup = fates_dataframe.set_index("label_id")["fate"]
        per_frame_dataframe["fate"] = per_frame_dataframe["label_id"].map(fate_lookup)
        per_frame_dataframe = per_frame_dataframe.sort_values(["label_id", "frame"]).reset_index(drop=True)

    return fates_dataframe, locked_labels, per_frame_dataframe


def assign_snapshot_fates(linked_labels, frame_cell_positive_area):
    """Classify every cell independently in every frame (no memory across time).

    The category for a cell in a given frame is built by joining the
    names of all fluorophores it is positive for with ``"+"``. Examples:

    * Positive for ``Red`` only          → ``"Red"``
    * Positive for ``Red`` AND ``Green`` → ``"Red+Green"``
    * Positive for nothing               → ``"negative"``

    Unlike :func:`assign_persistent_fates`, a cell can switch categories
    between frames.

    Parameters
    ----------
    linked_labels : numpy.ndarray, shape (n_frames, H, W), integer dtype
        Tracked cell segmentation.
    frame_cell_positive_area : dict[str, dict[int, dict[int, int]]]
        Output of :func:`measurement.compute_cell_positivity`.

    Returns
    -------
    pandas.DataFrame
        One row per ``(cell, frame)`` with columns ``label_id, frame,
        area, roundness``, one bool column per fluorophore,
        ``<fluorophore>_positive_area`` and ``category``.
    """
    fluorophore_names = list(frame_cell_positive_area.keys())
    snapshot_rows = []
    for frame_index in range(linked_labels.shape[0]):
        cell_shapes = measure_all_cells_in_frame(linked_labels[frame_index])
        for label_id, (area, roundness) in cell_shapes.items():
            is_positive = {fluorophore_name: frame_cell_positive_area[fluorophore_name][frame_index].get(label_id, 0) > 0 for fluorophore_name in fluorophore_names}
            category = "+".join(fluorophore_name for fluorophore_name in fluorophore_names if is_positive[fluorophore_name]) or "negative"
            snapshot_row = {"label_id": label_id, "frame": frame_index, "area": area, "roundness": roundness}
            snapshot_row.update(is_positive)
            for fluorophore_name in fluorophore_names:
                snapshot_row[f"{fluorophore_name}_positive_area"] = frame_cell_positive_area[fluorophore_name][frame_index].get(label_id, 0)
            snapshot_row["category"] = category
            snapshot_rows.append(snapshot_row)
    return pd.DataFrame(snapshot_rows)


def compute_persistent_percentages(assignments_dataframe, num_frames, fluorophore_names):
    """Build a time-series of cumulative percent-positive cells (persistent mode).

    For each frame ``f`` and each fluorophore the function counts how
    many cells satisfy ``fate == fluorophore`` AND
    ``first_positive_frame <= f``, divided by total cell count. Because
    fates are permanent, the resulting curves are monotonically
    non-decreasing.

    Parameters
    ----------
    assignments_dataframe : pandas.DataFrame
        Output of :func:`assign_persistent_fates` (one row per cell).
    num_frames : int
        Total number of frames in the time-lapse.
    fluorophore_names : list[str]
        Fluorophore names to compute percentages for.

    Returns
    -------
    pandas.DataFrame
        Columns ``frame, <fluorophore>_pct`` (one per channel) and
        ``total_positive_pct``.
    """
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


def compute_snapshot_percentages(snapshot_dataframe, num_frames):
    """Build a time-series of percent-cells per category (snapshot mode).

    Unlike persistent percentages, snapshot percentages can go up or
    down between frames because a cell can change category.

    Parameters
    ----------
    snapshot_dataframe : pandas.DataFrame
        Output of :func:`assign_snapshot_fates` (one row per
        ``(cell, frame)``).
    num_frames : int
        Total number of frames in the time-lapse.

    Returns
    -------
    summary_dataframe : pandas.DataFrame
        Columns ``frame, <category>_pct`` for each unique category.
    categories : list[str]
        Sorted list of category names (positive categories first,
        ``"negative"`` last).
    """
    categories = sorted(snapshot_dataframe["category"].unique(), key=lambda category: (category == "negative", category))
    counts = snapshot_dataframe.groupby(["frame", "category"]).size().unstack(fill_value=0)
    totals_per_frame = counts.sum(axis=1)
    percentages = counts.div(totals_per_frame, axis=0) * 100.0
    percentages = percentages.reindex(range(num_frames), fill_value=0.0)
    columns = {"frame": np.arange(num_frames)}
    for category in categories:
        columns[f"{category}_pct"] = percentages[category].values if category in percentages.columns else np.zeros(num_frames)
    return pd.DataFrame(columns), categories


def filter_by_frame_presence(tracks_dataframe, snapshot_dataframe, num_frames, minimum_percentage):
    """Drop cells that appear in fewer than *minimum_percentage* % of frames (snapshot mode).

    Cells tracked for only a small fraction of the time-lapse are often
    segmentation artefacts or cells entering/leaving the field of view.

    Parameters
    ----------
    tracks_dataframe : pandas.DataFrame or None
        TrackMate tracks CSV (columns: ``track_id, t, x, y, ...``). If
        ``None`` or empty, only the snapshot table is filtered.
    snapshot_dataframe : pandas.DataFrame
        Output of :func:`assign_snapshot_fates`.
    num_frames : int
        Total frames in the time-lapse.
    minimum_percentage : float
        Minimum percent of frames a cell must appear in to be kept (0–100).

    Returns
    -------
    filtered_tracks_dataframe : pandas.DataFrame or None
        Filtered tracks (or the original if ``tracks_dataframe`` is
        ``None``/empty).
    filtered_snapshot_dataframe : pandas.DataFrame
        Filtered snapshot data.
    """
    minimum_frame_count = num_frames * minimum_percentage / 100.0
    frame_counts = snapshot_dataframe.groupby("label_id")["frame"].nunique()
    keep_label_ids = frame_counts[frame_counts >= minimum_frame_count].index
    filtered_snapshot_dataframe = snapshot_dataframe[snapshot_dataframe["label_id"].isin(keep_label_ids)].copy()
    if tracks_dataframe is not None and len(tracks_dataframe) > 0:
        keep_track_ids = keep_label_ids.astype(int) - 1
        filtered_tracks_dataframe = tracks_dataframe[tracks_dataframe["track_id"].isin(keep_track_ids)].copy()
    else:
        filtered_tracks_dataframe = tracks_dataframe
    return filtered_tracks_dataframe, filtered_snapshot_dataframe


def filter_persistent_by_frame_presence(assignments_dataframe, linked_labels, num_frames, minimum_percentage):
    """Drop cells that appear in fewer than *minimum_percentage* % of frames (persistent mode).

    Counts how many frames each cell appears in within ``linked_labels``
    and removes cells below the threshold from ``assignments_dataframe``.

    Parameters
    ----------
    assignments_dataframe : pandas.DataFrame
        Output of :func:`assign_persistent_fates`.
    linked_labels : numpy.ndarray, shape (n_frames, H, W)
        Tracked cell segmentation.
    num_frames : int
        Total frames in the time-lapse.
    minimum_percentage : float
        Minimum percent of frames a cell must appear in to be kept (0–100).

    Returns
    -------
    pandas.DataFrame
        Filtered copy of ``assignments_dataframe``.
    """
    minimum_frame_count = num_frames * minimum_percentage / 100.0
    frame_presence = {}
    for frame_index in range(linked_labels.shape[0]):
        for cell_id in np.unique(linked_labels[frame_index]):
            if cell_id > 0:
                frame_presence[cell_id] = frame_presence.get(cell_id, 0) + 1
    keep_label_ids = [cell_id for cell_id, count in frame_presence.items() if count >= minimum_frame_count]
    return assignments_dataframe[assignments_dataframe["label_id"].isin(keep_label_ids)].copy()
