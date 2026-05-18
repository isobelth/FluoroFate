"""TrackMate integration and lineage-tree construction.

Public functions
----------------
build_lineage_dataframe(spot_records, edges)
    Convert raw TrackMate spot/edge data into a per-spot DataFrame with
    hierarchical lineage IDs (e.g. "1", "1.1", "1.2", "1.1.1") and
    parent/generation columns.
generate_trackmate_labels(masks_path, output_directory, ...)
    Run TrackMate headlessly on a 2-D + time mask stack, write the
    tracks CSV, lineage summary CSV and a re-labelled "linked labels"
    TIFF where every cell carries a globally unique track-derived ID.

Convention
----------
Throughout the FluoroFate pipeline ``label_id == track_id + 1`` because
label 0 is reserved for background in label images.
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from utils import configure_java_home

LOGGER = logging.getLogger(__name__)


def build_lineage_dataframe(spot_records, edges):
    """Build a lineage tree from TrackMate spot rows and directed edges.

    Detects division events (one source spot with two or more outgoing
    edges to *different* tracks) and assigns hierarchical lineage IDs of
    the form "1", "1.1", "1.2", "1.1.1", ... Both daughter tracks of a
    division share the same ``parent_track_id`` and their sibling indices
    (``.1`` vs ``.2``) are assigned deterministically by ascending
    (first frame, y centroid, x centroid) so re-runs produce identical
    labels.

    Parameters
    ----------
    spot_records : list[dict]
        One dict per spot. Each dict must contain at least ``spot_id``,
        ``track_id``, ``t`` (frame index), ``y`` and ``x`` (centroid).
    edges : list[tuple[int, int]]
        ``(source_spot_id, target_spot_id)`` pairs from TrackMate. Each
        pair must be ordered so the source frame is less than or equal
        to the target frame.

    Returns
    -------
    pandas.DataFrame
        The input spot rows extended with ``lineage_id`` (str),
        ``parent_track_id`` (Int64, NA for founders) and ``generation``
        (int; 0 for founders, 1 for first-generation daughters, ...).
    """
    spots_dataframe = pd.DataFrame(spot_records)
    if len(spots_dataframe) == 0:
        spots_dataframe["lineage_id"] = pd.Series(dtype=str)
        spots_dataframe["parent_track_id"] = pd.Series(dtype="Int64")
        spots_dataframe["generation"] = pd.Series(dtype=int)
        return spots_dataframe

    spot_to_track = dict(zip(spots_dataframe["spot_id"], spots_dataframe["track_id"]))
    track_first_appearance = (spots_dataframe.sort_values("t").groupby("track_id").first()[["t", "y", "x"]].to_dict("index"))

    children_per_source_spot = defaultdict(list)
    for source_spot_id, target_spot_id in edges:
        children_per_source_spot[source_spot_id].append(target_spot_id)

    parent_track_of = {}
    sibling_index_of = {}
    for source_spot_id, target_spot_ids in children_per_source_spot.items():
        if len(target_spot_ids) < 2:
            continue
        source_track_id = spot_to_track[source_spot_id]
        daughter_track_ids = []
        for target_spot_id in target_spot_ids:
            target_track_id = spot_to_track[target_spot_id]
            if target_track_id != source_track_id and target_track_id not in daughter_track_ids:
                daughter_track_ids.append(target_track_id)
        if not daughter_track_ids:
            continue
        if len(daughter_track_ids) > 2:
            LOGGER.warning("Division at spot %s produced %d daughters (track ids %s); treating all as siblings.", source_spot_id, len(daughter_track_ids), daughter_track_ids)
        parent_continues = any(spot_to_track[target_spot_id] == source_track_id for target_spot_id in target_spot_ids)
        sibling_track_ids = list(daughter_track_ids)
        if parent_continues and source_track_id not in sibling_track_ids:
            sibling_track_ids.append(source_track_id)
        sibling_track_ids.sort(key=lambda track_id: (track_first_appearance[track_id]["t"], track_first_appearance[track_id]["y"], track_first_appearance[track_id]["x"]))
        for sibling_index, child_track_id in enumerate(sibling_track_ids, start=1):
            if child_track_id not in parent_track_of:
                parent_track_of[child_track_id] = source_track_id
                sibling_index_of[child_track_id] = sibling_index

    all_track_ids = sorted(spots_dataframe["track_id"].unique())
    root_track_ids = [track_id for track_id in all_track_ids if track_id not in parent_track_of]

    lineage_id_of = {track_id: str(root_index) for root_index, track_id in enumerate(root_track_ids, start=1)}
    children_per_parent = defaultdict(list)
    for child_track_id, parent_track_id in parent_track_of.items():
        children_per_parent[parent_track_id].append(child_track_id)

    queue = list(root_track_ids)
    visited = set(root_track_ids)
    while queue:
        track_id = queue.pop(0)
        for child_track_id in sorted(children_per_parent.get(track_id, []), key=lambda child_id: sibling_index_of.get(child_id, 0)):
            if child_track_id in visited:
                continue
            lineage_id_of[child_track_id] = f"{lineage_id_of.get(track_id, '?')}.{sibling_index_of.get(child_track_id, 1)}"
            visited.add(child_track_id)
            queue.append(child_track_id)

    for track_id in all_track_ids:
        if track_id not in lineage_id_of:
            lineage_id_of[track_id] = str(track_id)

    spots_dataframe["lineage_id"] = spots_dataframe["track_id"].map(lineage_id_of)
    spots_dataframe["parent_track_id"] = spots_dataframe["track_id"].map(lambda track_id: parent_track_of.get(track_id, pd.NA)).astype("Int64")
    spots_dataframe["generation"] = spots_dataframe["lineage_id"].str.count(r"\.").astype(int)
    return spots_dataframe


def generate_trackmate_labels(masks_path, output_directory, target_channel=1, simplify_contours=False, initial_search_radius=30.0, search_radius=150.0, max_frame_gap=3, allow_track_splitting=True, splitting_max_distance=15.0, allow_track_merging=False, imagej_instance=None):
    """Run TrackMate headlessly on a Cellpose mask stack and produce linked labels.

    Reads the mask stack, links cell objects across time using TrackMate's
    ``LabelImageDetectorFactory`` + ``AdvancedKalmanTrackerFactory``,
    optionally infers mother/daughter relationships when track splitting
    is enabled, then re-paints the masks so each tracked cell carries a
    globally unique label across all frames.

    Three files are written into ``output_directory``:

    * ``trackmate_tracks.csv``        — one row per (spot, frame) with
      ``track_id, t, y, x, quality, lineage_id, parent_track_id, generation``.
    * ``lineage_summary.csv``         — one row per ``track_id`` with its
      lineage ID, parent track, generation and frame range.
    * ``linked_labels_trackmate.tiff``— uint32 label stack where every
      cell in every frame has the same ID across time.

    Parameters
    ----------
    masks_path : str or pathlib.Path
        Path to a TIFF file containing the Cellpose mask stack
        (``(n_frames, H, W)``, uint16/uint32).
    output_directory : str or pathlib.Path
        Directory where output files will be written. Created if missing.
    target_channel : int, default 1
        Channel index passed to TrackMate's label detector (1-based).
    simplify_contours : bool, default False
        Forwarded to TrackMate's label detector.
    initial_search_radius : float, default 30.0
        Linking distance used by the Kalman tracker for the first frame.
    search_radius : float, default 150.0
        Maximum search radius used by the Kalman tracker after the first
        frame.
    max_frame_gap : int, default 3
        Maximum number of frames a track may be missing before TrackMate
        terminates it.
    allow_track_splitting : bool, default True
        Whether to allow track splitting (cell division). When False the
        ``lineage_id`` / ``parent_track_id`` / ``generation`` columns
        will be trivial (every cell looks like a founder). Defaults to
        True so mother/daughter relationships are populated by default.
    splitting_max_distance : float, default 15.0
        Maximum distance for splitting links. Ignored when splitting is
        disabled.
    allow_track_merging : bool, default False
        Whether to allow track merging.
    imagej_instance : ImageJ instance or None, default None
        Pass an existing PyImageJ instance to reuse it across calls (this
        is significantly faster than re-initialising for every file).

    Returns
    -------
    dict
        Keys: ``"imagej_instance"``, ``"trackmate_tracks_df"``,
        ``"linked_labels"``, ``"linked_labels_path"``, ``"tracks_csv"``.
    """
    import imagej as imagej_module
    import scyjava as scyjava_module
    from imagej import Mode as ImageJMode

    if imagej_instance is None:
        configure_java_home()
        imagej_instance = imagej_module.init("sc.fiji:fiji", mode=ImageJMode.HEADLESS, add_legacy=True)

    IJ = scyjava_module.jimport("ij.IJ")
    HashMap = scyjava_module.jimport("java.util.HashMap")
    Integer = scyjava_module.jimport("java.lang.Integer")
    Double = scyjava_module.jimport("java.lang.Double")
    Model = scyjava_module.jimport("fiji.plugin.trackmate.Model")
    Settings = scyjava_module.jimport("fiji.plugin.trackmate.Settings")
    TrackMate = scyjava_module.jimport("fiji.plugin.trackmate.TrackMate")
    Logger = scyjava_module.jimport("fiji.plugin.trackmate.Logger")
    LabelImageDetectorFactory = scyjava_module.jimport("fiji.plugin.trackmate.detection.LabelImageDetectorFactory")
    AdvancedKalmanTrackerFactory = scyjava_module.jimport("fiji.plugin.trackmate.tracking.kalman.AdvancedKalmanTrackerFactory")

    LOGGER.info("TrackMate: opening %s", masks_path)
    image_plus = IJ.openImage(str(masks_path))
    if image_plus is None:
        raise RuntimeError(f"Could not open mask stack: {masks_path}")
    num_timepoints = int(image_plus.getNFrames())
    if num_timepoints <= 1:
        num_timepoints = int(image_plus.getStackSize())
    image_plus.setDimensions(1, 1, num_timepoints)
    image_plus.setOpenAsHyperStack(True)

    trackmate_model = Model()
    trackmate_model.setLogger(Logger.IJ_LOGGER)
    settings = Settings(image_plus)
    settings.detectorFactory = LabelImageDetectorFactory()
    detector_settings = HashMap()
    detector_settings.put("TARGET_CHANNEL", Integer.valueOf(int(target_channel)))
    detector_settings.put("SIMPLIFY_CONTOURS", bool(simplify_contours))
    settings.detectorSettings = detector_settings

    settings.trackerFactory = AdvancedKalmanTrackerFactory()
    tracker_settings = HashMap(settings.trackerFactory.getDefaultSettings())
    tracker_settings.put("KALMAN_SEARCH_RADIUS", Double.valueOf(float(search_radius)))
    tracker_settings.put("LINKING_MAX_DISTANCE", Double.valueOf(float(initial_search_radius)))
    tracker_settings.put("MAX_FRAME_GAP", Integer.valueOf(int(max_frame_gap)))
    tracker_settings.put("ALLOW_TRACK_SPLITTING", bool(allow_track_splitting))
    tracker_settings.put("SPLITTING_MAX_DISTANCE", Double.valueOf(float(splitting_max_distance)))
    tracker_settings.put("ALLOW_TRACK_MERGING", bool(allow_track_merging))
    feature_penalties = HashMap()
    for feature_name in ("POSITION_X", "POSITION_Y", "AREA"):
        feature_penalties.put(feature_name, Double.valueOf(1.0))
    tracker_settings.put("LINKING_FEATURE_PENALTIES", feature_penalties)
    tracker_settings.put("GAP_CLOSING_FEATURE_PENALTIES", feature_penalties)
    settings.trackerSettings = tracker_settings
    settings.addAllAnalyzers()

    trackmate = TrackMate(trackmate_model, settings)
    if not trackmate.checkInput():
        raise RuntimeError(f"TrackMate input rejected: {trackmate.getErrorMessage()}")
    if not trackmate.process():
        raise RuntimeError(f"TrackMate processing failed: {trackmate.getErrorMessage()}")

    track_model = trackmate_model.getTrackModel()

    spot_info_by_id = {}
    spot_records = []
    for track_id in track_model.trackIDs(True):
        for spot in sorted(track_model.trackSpots(track_id), key=lambda spot: float(spot.getFeature("FRAME"))):
            spot_id = int(spot.ID())
            spot_record = {"spot_id": spot_id, "track_id": int(track_id), "t": int(float(spot.getFeature("FRAME"))), "y": float(spot.getFeature("POSITION_Y")), "x": float(spot.getFeature("POSITION_X")), "quality": float(spot.getFeature("QUALITY"))}
            spot_info_by_id[spot_id] = spot_record
            spot_records.append(spot_record)

    edges = []
    for track_id in track_model.trackIDs(True):
        for edge in track_model.trackEdges(track_id):
            source_spot_id = int(track_model.getEdgeSource(edge).ID())
            target_spot_id = int(track_model.getEdgeTarget(edge).ID())
            if spot_info_by_id[source_spot_id]["t"] > spot_info_by_id[target_spot_id]["t"]:
                source_spot_id, target_spot_id = target_spot_id, source_spot_id
            edges.append((source_spot_id, target_spot_id))

    lineage_dataframe = build_lineage_dataframe(spot_records, edges)
    if allow_track_splitting and lineage_dataframe["parent_track_id"].notna().sum() == 0:
        LOGGER.warning("Track splitting was enabled but TrackMate produced no division edges; lineage_id will be trivial.")
    elif not allow_track_splitting:
        LOGGER.warning("Track splitting is disabled; mother/daughter lineage will not be inferred. Enable allow_track_splitting to populate lineage_id and parent_track_id.")

    tracks_dataframe = lineage_dataframe[["track_id", "t", "y", "x", "quality", "lineage_id", "parent_track_id", "generation"]].copy()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    tracks_csv_path = output_directory / "trackmate_tracks.csv"
    tracks_dataframe.to_csv(tracks_csv_path, index=False)

    lineage_summary = (tracks_dataframe.groupby("track_id").agg(lineage_id=("lineage_id", "first"), parent_track_id=("parent_track_id", "first"), generation=("generation", "first"), first_frame=("t", "min"), last_frame=("t", "max"), n_frames=("t", "count")).reset_index().sort_values("lineage_id"))
    lineage_summary.to_csv(output_directory / "lineage_summary.csv", index=False)

    masks_stack = tifffile.imread(str(masks_path))
    linked_labels_stack = np.zeros_like(masks_stack, dtype=np.uint32)
    num_frames, num_y, num_x = masks_stack.shape
    for track_id_value, frame_value, y_value, x_value in tracks_dataframe[["track_id", "t", "y", "x"]].to_numpy():
        frame_index = int(frame_value)
        y_index = int(np.clip(round(float(y_value)), 0, num_y - 1))
        x_index = int(np.clip(round(float(x_value)), 0, num_x - 1))
        if 0 <= frame_index < num_frames:
            label_at_centroid = int(masks_stack[frame_index, y_index, x_index])
            if label_at_centroid > 0:
                linked_labels_stack[frame_index, masks_stack[frame_index] == label_at_centroid] = int(track_id_value) + 1

    linked_labels_path = output_directory / "linked_labels_trackmate.tiff"
    tifffile.imwrite(str(linked_labels_path), linked_labels_stack)
    LOGGER.info("TrackMate: wrote %d tracks to %s", tracks_dataframe["track_id"].nunique(), linked_labels_path)

    return {"imagej_instance": imagej_instance, "trackmate_tracks_df": tracks_dataframe, "linked_labels": linked_labels_stack, "linked_labels_path": linked_labels_path, "tracks_csv": tracks_csv_path}
