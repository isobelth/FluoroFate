"""Cell measurement and fluorescence-to-cell assignment functions.

The blob → cell mapping uses *majority-overlap voting*: each
fluorescence-positive connected component is assigned to whichever
tracked cell label covers the most of its pixels. This is more robust
than centroid lookup (which silently dropped blobs whose centroid
landed on background or straddled two cells).
"""

import logging

import numpy as np
import pandas as pd
from skimage.measure import regionprops

LOGGER = logging.getLogger(__name__)


def measure_all_cells_in_frame(label_image):
    """Measure area and roundness for every cell in a single label image.

    Parameters
    ----------
    label_image : numpy.ndarray, shape (H, W), integer dtype
        Label image where each cell has a unique integer ID
        (0 = background).

    Returns
    -------
    dict[int, tuple[int, float]]
        ``{cell_label_id: (area_in_pixels, roundness)}`` for every cell.
        Roundness = ``minor_axis / major_axis`` (1.0 for a circle, closer
        to 0 for an elongated cell). NaN if the major axis length is 0.
    """
    measurements = {}
    for region in regionprops(label_image):
        roundness = region.minor_axis_length / region.major_axis_length if region.major_axis_length > 0 else np.nan
        measurements[region.label] = (region.area, roundness)
    return measurements


def assign_positive_blobs_to_cells(positive_labels_frame, linked_labels_frame):
    """Assign fluorescence-positive blobs to tracked cells by majority overlap.

    For each connected component in ``positive_labels_frame`` the
    function counts how many of its pixels fall inside each tracked cell
    and assigns the entire blob's area to the cell label that wins the
    pixel-count vote. Blobs lying entirely on background are skipped and
    a debug message is logged.

    Parameters
    ----------
    positive_labels_frame : numpy.ndarray, shape (H, W), integer dtype
        Connected-component labels of the thresholded fluorescence image
        for one frame.
    linked_labels_frame : numpy.ndarray, shape (H, W), integer dtype
        Tracked cell segmentation labels for the same frame
        (0 = background; non-zero = cell ID).

    Returns
    -------
    cell_positive_area : dict[int, int]
        ``{cell_label_id: total fluorescence-positive pixel area}``. Only
        cells that received at least one blob are present in the dict.
    """
    if positive_labels_frame.shape != linked_labels_frame.shape:
        raise ValueError(f"positive_labels_frame shape {positive_labels_frame.shape} does not match linked_labels_frame shape {linked_labels_frame.shape}.")
    cell_positive_area = {}
    dropped_blob_count = 0
    for region in regionprops(positive_labels_frame):
        cell_labels_under_blob = linked_labels_frame[region.coords[:, 0], region.coords[:, 1]]
        non_background = cell_labels_under_blob[cell_labels_under_blob > 0]
        if non_background.size == 0:
            dropped_blob_count += 1
            continue
        unique_cell_ids, pixel_counts = np.unique(non_background, return_counts=True)
        winning_cell_id = int(unique_cell_ids[int(np.argmax(pixel_counts))])
        cell_positive_area[winning_cell_id] = cell_positive_area.get(winning_cell_id, 0) + int(region.area)
    if dropped_blob_count:
        LOGGER.debug("Dropped %d fluorescence blob(s) lying entirely on background.", dropped_blob_count)
    return cell_positive_area


def compute_cell_positivity(linked_labels, positive_label_stacks, fluorophore_names):
    """Map fluorescence blobs to tracked cells for every frame and channel.

    Wraps :func:`assign_positive_blobs_to_cells` over all frames and
    channels and additionally returns a per-channel label image where
    only cells that picked up at least one positive blob in that frame
    are painted with their cell ID (everything else is 0).

    Parameters
    ----------
    linked_labels : numpy.ndarray, shape (n_frames, H, W), integer dtype
        Tracked cell segmentation produced by TrackMate.
    positive_label_stacks : dict[str, numpy.ndarray]
        ``{fluorophore_name: (n_frames, H, W) labels}`` from
        :func:`segmentation.segment_fluorescence`.
    fluorophore_names : list[str]
        Names matching the keys of ``positive_label_stacks``.

    Returns
    -------
    frame_cell_positive_area : dict[str, dict[int, dict[int, int]]]
        Nested lookup. ``frame_cell_positive_area[fluorophore][frame_index][cell_id]``
        gives the total fluorescence-positive pixel area assigned to that
        cell in that frame.
    positive_cell_labels : dict[str, numpy.ndarray]
        ``{fluorophore_name: (n_frames, H, W) uint32 labels}`` containing
        only the cell IDs that were positive in each frame.
    """
    num_frames = linked_labels.shape[0]
    frame_cell_positive_area = {}
    positive_cell_labels = {}
    for fluorophore_name in fluorophore_names:
        frame_cell_positive_area[fluorophore_name] = {}
        per_frame_positive_labels = np.zeros_like(linked_labels)
        for frame_index in range(num_frames):
            cell_positivity = assign_positive_blobs_to_cells(positive_label_stacks[fluorophore_name][frame_index], linked_labels[frame_index])
            frame_cell_positive_area[fluorophore_name][frame_index] = cell_positivity
            if cell_positivity:
                positive_ids = np.fromiter(cell_positivity.keys(), dtype=np.uint32)
                is_positive = np.isin(linked_labels[frame_index], positive_ids)
                per_frame_positive_labels[frame_index] = np.where(is_positive, linked_labels[frame_index], 0)
        positive_cell_labels[fluorophore_name] = per_frame_positive_labels
    return frame_cell_positive_area, positive_cell_labels


def compute_per_cell_intensity_area(linked_labels, fluorophore_stacks):
    """Compute per-cell, per-frame area and summed fluorescence intensity.

    For every tracked cell present in a frame, sums the raw pixel
    intensity of each fluorescence channel inside that cell's mask and
    records the cell's area in pixels. No thresholding is applied — this
    is a direct measurement on the tracked label image.

    Parameters
    ----------
    linked_labels : numpy.ndarray, shape (n_frames, H, W), integer dtype
        Tracked cell segmentation produced by TrackMate
        (0 = background; non-zero = cell ID, consistent across frames).
    fluorophore_stacks : dict[str, numpy.ndarray]
        ``{fluorophore_name: (n_frames, H, W) array}`` of raw
        fluorescence images.

    Returns
    -------
    pandas.DataFrame
        Columns: ``frame``, ``cell_id``, ``area_px``, and one
        ``<fluorophore>_total_intensity`` column per fluorophore. One
        row per (frame, cell present in that frame).
    """
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
