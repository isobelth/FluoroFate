"""FluoroFate — napari + magicgui GUI.

Pipeline: Cellpose segmentation -> TrackMate tracking -> fluorescence
thresholding -> fate assignment (persistent / snapshot) -> percentage
trajectories.

Two workflows:
  - Analyse Single Image : full pipeline, all outputs shown in napari.
  - Analyse All in Folder : batch every TIFF in a folder -> CSV.
"""

import logging
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import tifffile

from magicgui import magicgui
from magicgui.widgets import TextEdit
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from utils import running_in_notebook
from colours import assign_colours, get_fluor_base_colour, add_coloured_labels
from measurement import compute_cell_positivity
from segmentation import cellpose_live_segmentation, segment_fluorescence
from tracking import generate_trackmate_labels
from fate_assignment import (
    assign_persistent_fates,
    assign_snapshot_fates,
    compute_persistent_percentages,
    compute_snapshot_percentages,
    filter_by_frame_presence,
    filter_persistent_by_frame_presence,
)
from plotting import (
    plot_persistent_percentages,
    plot_snapshot_percentages,
    plot_snapshot_trajectories,
    plot_snapshot_cell_timelines,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger("fluorofate")

# Cellpose defaults. ``None`` for the diameter triggers per-image
# auto-estimation in ``cellpose_live_segmentation``. The flow and
# cell-probability thresholds are the Cellpose library defaults at the
# time of writing; tighten ``flow_threshold`` to reject more uncertain
# boundaries and raise ``cellprob_threshold`` to drop dim cells.
CELLPOSE_DEFAULT_DIAMETER = None
CELLPOSE_DEFAULT_FLOW_THRESHOLD = 0.4
CELLPOSE_DEFAULT_CELLPROB_THRESHOLD = 0.0

# Cell frame-presence cutoffs (percent of frames a cell must appear in)
# used to generate filtered analysis outputs.
FRAME_PRESENCE_CUTOFF_PERCENTAGES = (30, 40, 50, 60)


# ---------------------------------------------------------------------------
#  GUI Application
# ---------------------------------------------------------------------------

class FluoroFateApp:
    """Napari-based GUI that wraps the full cell-death analysis pipeline."""

    # -- placeholder methods for config panels (never called directly) ------

    @staticmethod
    def _file_placeholder(
        single_tiff: Path = Path(),
        batch_folder: Path = Path(),
        output_directory: Path = Path(),
    ):
        pass

    @staticmethod
    def _channel_placeholder(
        brightfield_channel: int = -1,
        fluor_1_name: str = "Green",
        fluor_1_channel: int = 0,
        fluor_1_threshold: str = "otsu",
        fluor_2_name: str = "Red",
        fluor_2_channel: int = 1,
        fluor_2_threshold: str = "otsu",
        fluor_3_name: str = "",
        fluor_3_channel: int = 2,
        fluor_3_threshold: str = "otsu",
    ):
        pass

    @staticmethod
    def _analysis_placeholder(
        analysis_mode: str = "persistent",
        blur_sigma: float = 1.0,
        custom_model_file: Path = Path(),
    ):
        pass

    @staticmethod
    def _cellpose_placeholder(
        model_type: str = "cpsam",
        min_size: int = 15,
        use_gpu: bool = True,
    ):
        pass

    @staticmethod
    def _trackmate_placeholder(
        initial_search_radius: float = 30.0,
        search_radius: float = 150.0,
        max_frame_gap: int = 2,
        allow_splitting: bool = True,
        splitting_max_distance: float = 15.0,
        allow_merging: bool = False,
    ):
        pass

    # -- init ---------------------------------------------------------------

    def __init__(self):
        self.viewer = napari.Viewer(title="FluoroFate")

        qt_app = QApplication.instance()
        if qt_app is not None:
            qt_app.setQuitOnLastWindowClosed(not running_in_notebook())

        # -- state --
        # PyImageJ instance, lazy-initialised on first TrackMate call and
        # then reused for every subsequent file (re-initialising the JVM
        # for every file is extremely slow). The same instance is shared
        # across batch processing.
        self._imagej_instance = None
        self._results: List[dict] = []
        self._results_df: Optional[pd.DataFrame] = None

        # -- progress bar --
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("Idle")

        # -- log --
        self._log = TextEdit(value="")
        try:
            self._log.native.setReadOnly(True)
        except Exception:
            pass
        self._log.min_height = 100
        self._log.max_height = 300

        # -- file panel --
        self.file_panel = magicgui(
            self._file_placeholder,
            single_tiff={
                "label": "Single TIFF (.tif/.tiff)",
                "mode": "r",
                "filter": "*.tif *.tiff",
            },
            batch_folder={"label": "Batch folder (optional for single)", "mode": "d"},
            output_directory={"label": "Output directory", "mode": "d"},
            call_button=False,
        )

        # -- channel panel --
        self.channel_panel = magicgui(
            self._channel_placeholder,
            brightfield_channel={
                "label": "Brightfield ch (-1=last)",
                "min": -1,
                "max": 20,
            },
            fluor_1_name={"label": "Fluorophore 1 name"},
            fluor_1_channel={
                "label": "Fluorophore 1 channel",
                "min": 0,
                "max": 20,
            },
            fluor_1_threshold={
                "label": "Fluorophore 1 threshold",
                "choices": ["mean", "minimum", "yen", "otsu", "triangle"],
            },
            fluor_2_name={"label": "Fluorophore 2 name"},
            fluor_2_channel={
                "label": "Fluorophore 2 channel",
                "min": 0,
                "max": 20,
            },
            fluor_2_threshold={
                "label": "Fluorophore 2 threshold",
                "choices": ["mean", "minimum", "yen", "otsu", "triangle"],
            },
            fluor_3_name={"label": "Fluorophore 3 (blank=none)"},
            fluor_3_channel={
                "label": "Fluorophore 3 channel",
                "min": 0,
                "max": 20,
            },
            fluor_3_threshold={
                "label": "Fluorophore 3 threshold",
                "choices": ["mean", "minimum", "yen", "otsu", "triangle"],
            },
            call_button=False,
        )

        # -- analysis panel --
        self.analysis_panel = magicgui(
            self._analysis_placeholder,
            analysis_mode={
                "label": "Analysis mode",
                "choices": ["persistent", "snapshot"],
            },
            blur_sigma={
                "label": "Blur sigma",
                "min": 0.1,
                "max": 20.0,
                "step": 0.1,
            },
            custom_model_file={
                "label": "Select file (optional custom model)",
                "mode": "r",
                "filter": "*.pt *.pth",
            },
            call_button=False,
        )
        self.analysis_panel.custom_model_file.changed.connect(
            self._on_custom_model_file_changed
        )

        # -- cellpose panel --
        self.cellpose_panel = magicgui(
            self._cellpose_placeholder,
            model_type={
                "label": "Model",
                "choices": ["cpsam", "cyto3", "cyto2", "cyto", "nuclei"],
            },
            min_size={"label": "Min cell size (px)", "min": 0, "max": 2000},
            use_gpu={"label": "Use GPU"},
            call_button=False,
        )
        self._on_custom_model_file_changed()

        # -- trackmate panel --
        self.trackmate_panel = magicgui(
            self._trackmate_placeholder,
            initial_search_radius={
                "label": "Init search radius",
                "min": 1.0,
                "max": 500.0,
                "step": 1.0,
            },
            search_radius={
                "label": "Search radius",
                "min": 1.0,
                "max": 2000.0,
                "step": 5.0,
            },
            max_frame_gap={"label": "Max frame gap", "min": 0, "max": 50},
            allow_splitting={"label": "Allow splitting"},
            splitting_max_distance={
                "label": "Splitting max distance",
                "min": 1.0,
                "max": 2000.0,
                "step": 5.0,
            },
            allow_merging={"label": "Allow merging"},
            call_button=False,
        )

        # -- action buttons --
        self._btn_segment = magicgui(
            self._run_segmentation_only, call_button="1) Run Segmentation"
        )
        self._btn_track = magicgui(
            self._run_tracking_only, call_button="2) Run Tracking"
        )
        self._btn_persistent = magicgui(
            self._run_persistent_analysis,
            call_button="3a) Run Persistent Analysis",
        )
        self._btn_snapshot = magicgui(
            self._run_snapshot_analysis,
            call_button="3b) Run Snapshot Analysis",
        )
        self._btn_single = magicgui(
            self._run_all_single_image, call_button="Run All (Single Image)"
        )
        self._btn_folder = magicgui(
            self._analyse_folder, call_button="Analyse All in Folder"
        )
        self._btn_save = magicgui(
            self._save_results, call_button="Save Results"
        )

        # -- dock widgets --
        self.viewer.window.add_dock_widget(
            self.file_panel, name="Files", area="right"
        )
        self.viewer.window.add_dock_widget(
            self.channel_panel, name="Channels", area="right"
        )
        self.viewer.window.add_dock_widget(
            self.analysis_panel, name="Analysis", area="right"
        )
        self.viewer.window.add_dock_widget(
            self.cellpose_panel, name="Cellpose", area="right"
        )
        self.viewer.window.add_dock_widget(
            self.trackmate_panel, name="TrackMate", area="right"
        )

        # Run panel: buttons + progress + log
        run_widget = QWidget()
        run_layout = QVBoxLayout(run_widget)
        run_layout.addWidget(self._btn_segment.native)
        run_layout.addWidget(self._btn_track.native)
        run_layout.addWidget(self._btn_persistent.native)
        run_layout.addWidget(self._btn_snapshot.native)
        run_layout.addWidget(self._btn_single.native)
        run_layout.addWidget(self._btn_folder.native)
        run_layout.addWidget(self._btn_save.native)
        run_layout.addWidget(self._progress)
        run_layout.addWidget(self._log.native)
        self.viewer.window.add_dock_widget(
            run_widget, name="Run & Log", area="right"
        )

    # -- helpers ------------------------------------------------------------

    def _append_log(self, msg: str):
        self._log.value = (
            (self._log.value.rstrip() + "\n" + msg)
            if self._log.value
            else msg
        )

    def _set_progress(self, value: int, maximum: int = 100, fmt: str = ""):
        self._progress.setRange(0, maximum)
        self._progress.setValue(min(value, maximum) if maximum > 0 else 0)
        if fmt:
            self._progress.setFormat(fmt)
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _set_buttons_enabled(self, enabled: bool):
        self._btn_segment.call_button.enabled = enabled
        self._btn_track.call_button.enabled = enabled
        self._btn_persistent.call_button.enabled = enabled
        self._btn_snapshot.call_button.enabled = enabled
        self._btn_single.call_button.enabled = enabled
        self._btn_folder.call_button.enabled = enabled
        self._btn_save.call_button.enabled = enabled

    def _custom_model_is_selected(self) -> bool:
        custom_value = str(self.analysis_panel.custom_model_file.value).strip()
        return bool(custom_value and custom_value != ".")

    def _on_custom_model_file_changed(self, *event_args):
        use_custom = self._custom_model_is_selected()
        self.cellpose_panel.model_type.enabled = not use_custom
        self.cellpose_panel.model_type.tooltip = (
            "Disabled because a custom model file is selected in Analysis."
            if use_custom
            else ""
        )

    def _read_fluorophore_config(self):
        """Read channel panel and return fluorophore names, per-name channel/threshold maps and brightfield channel.

        Raises ``ValueError`` if no fluorophore name is given or if any
        named fluorophore shares a channel with another fluorophore or
        with the brightfield channel.
        """
        brightfield_channel = int(self.channel_panel.brightfield_channel.value)
        fluorophore_names: List[str] = []
        fluorophore_channels: Dict[str, int] = {}
        fluorophore_thresholds: Dict[str, str] = {}
        for name_attr, channel_attr, threshold_attr in [("fluor_1_name", "fluor_1_channel", "fluor_1_threshold"), ("fluor_2_name", "fluor_2_channel", "fluor_2_threshold"), ("fluor_3_name", "fluor_3_channel", "fluor_3_threshold")]:
            fluorophore_name = str(getattr(self.channel_panel, name_attr).value).strip()
            channel_index = int(getattr(self.channel_panel, channel_attr).value)
            threshold_method = str(getattr(self.channel_panel, threshold_attr).value)
            if fluorophore_name:
                fluorophore_names.append(fluorophore_name)
                fluorophore_channels[fluorophore_name] = channel_index
                fluorophore_thresholds[fluorophore_name] = threshold_method
        if not fluorophore_names:
            raise ValueError("At least one fluorophore name must be provided.")
        seen_channels = {}
        for fluorophore_name, channel_index in fluorophore_channels.items():
            if channel_index in seen_channels:
                raise ValueError(f"Fluorophore '{fluorophore_name}' uses channel {channel_index}, already used by '{seen_channels[channel_index]}'.")
            if brightfield_channel >= 0 and channel_index == brightfield_channel:
                raise ValueError(f"Fluorophore '{fluorophore_name}' uses channel {channel_index}, which is also the brightfield channel.")
            seen_channels[channel_index] = fluorophore_name
        return fluorophore_names, fluorophore_channels, brightfield_channel, fluorophore_thresholds

    def _resolve_single_run_paths(self):
        """Resolve and validate single-image input/output paths."""
        tiff_path = Path(str(self.file_panel.single_tiff.value))
        if not tiff_path.exists() or not tiff_path.is_file():
            raise ValueError("Select a valid TIFF file in the Files panel.")
        if tiff_path.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError("Single image must be a .tif or .tiff file.")

        out_dir = Path(str(self.file_panel.output_directory.value))
        if not str(out_dir).strip() or str(out_dir) == ".":
            out_dir = tiff_path.parent / "cell_death_output"
        work_dir = out_dir / tiff_path.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        return tiff_path, out_dir, work_dir

    def _load_image_channels(self, tiff_path: Path):
        """Load a 4-D TIFF and split it into brightfield and fluorophore stacks.

        Returns ``(brightfield_stack, fluorophore_stacks, fluorophore_names, num_channels)``
        where ``fluorophore_stacks`` is ``{name: (n_frames, H, W) array}``.
        Raises ``ValueError`` if the file is not 4-D ``(T, C, Y, X)`` or if
        a configured channel index is out of range.
        """
        fluorophore_names, fluorophore_channels, brightfield_channel, _ = self._read_fluorophore_config()
        image = tifffile.imread(str(tiff_path))
        if image.ndim != 4:
            raise ValueError(f"Expected 4-D TIFF (T, C, Y, X), got {image.ndim}-D shape {image.shape}.")
        num_channels = image.shape[1]
        for fluorophore_name, channel_index in fluorophore_channels.items():
            if channel_index >= num_channels:
                raise ValueError(f"Channel {channel_index} for '{fluorophore_name}' is out of range (image has {num_channels} channels).")
        if brightfield_channel >= num_channels:
            raise ValueError(f"Brightfield channel {brightfield_channel} is out of range (image has {num_channels} channels).")
        brightfield_stack = image[:, brightfield_channel, :, :]
        fluorophore_stacks = {fluorophore_name: image[:, fluorophore_channels[fluorophore_name], :, :] for fluorophore_name in fluorophore_names}
        return brightfield_stack, fluorophore_stacks, fluorophore_names, num_channels

    def _resolve_cellpose_model_config(self):
        """Read model panel and resolve built-in/custom model selection."""
        model_type = str(self.cellpose_panel.model_type.value)
        custom_value = str(self.analysis_panel.custom_model_file.value).strip()
        custom_model_path = None
        if self._custom_model_is_selected():
            custom_model_path = Path(custom_value)
            if not custom_model_path.exists():
                raise ValueError("Custom model path does not exist.")
            self._append_log(
                f"Using custom Cellpose model path: {custom_model_path}"
            )
        else:
            self._append_log(f"Using built-in Cellpose model: {model_type}")
        return model_type, custom_model_path

    def _run_segmentation_stage(self, tiff_path: Path, work_dir: Path):
        """Run Cellpose segmentation on the brightfield channel and save the masks stack.

        Returns a dict with the loaded brightfield + fluorophore stacks,
        the resulting masks stack and the path it was written to.
        """
        cellpose_model_type, cellpose_custom_model_path = self._resolve_cellpose_model_config()
        cellpose_min_size = int(self.cellpose_panel.min_size.value)
        cellpose_use_gpu = bool(self.cellpose_panel.use_gpu.value)

        self._set_progress(2, fmt="Loading image...")
        self._append_log(f"Loading: {tiff_path.name}")
        brightfield_stack, fluorophore_stacks, _, num_channels = self._load_image_channels(tiff_path)
        self._append_log(f"  Shape: ({brightfield_stack.shape[0]}, {num_channels}, {brightfield_stack.shape[1]}, {brightfield_stack.shape[2]})  ({brightfield_stack.shape[0]} frames, {num_channels} channels)")

        self._set_progress(5, fmt="Cellpose: starting...")
        self._append_log("Running Cellpose segmentation...")

        masks_stack = cellpose_live_segmentation(brightfield_stack, diameter=CELLPOSE_DEFAULT_DIAMETER, flow_threshold=CELLPOSE_DEFAULT_FLOW_THRESHOLD, cellprob_threshold=CELLPOSE_DEFAULT_CELLPROB_THRESHOLD, min_size=cellpose_min_size, model_type=cellpose_model_type, custom_model_path=cellpose_custom_model_path, gpu=cellpose_use_gpu, progress_callback=lambda current_frame, total_frames: self._set_progress(5 + int(90 * current_frame / total_frames), fmt=f"Cellpose: frame {current_frame}/{total_frames}"))

        masks_path = work_dir / "masks_stack.tiff"
        tifffile.imwrite(str(masks_path), masks_stack.astype(np.uint16))
        self._set_progress(100, fmt="Segmentation saved")
        self._append_log(f"[OK] Saved segmentation -> {masks_path}")

        return {"bf_image": brightfield_stack, "fluor_images": fluorophore_stacks, "masks_stack": masks_stack, "masks_path": masks_path}

    def _run_tracking_stage(self, work_dir: Path):
        """Run TrackMate on the saved Cellpose mask stack and save the linked-label TIFF.

        Reuses ``self._imagej_instance`` across calls so the JVM is
        initialised exactly once per FluoroFate session.
        """
        initial_search_radius = float(self.trackmate_panel.initial_search_radius.value)
        search_radius = float(self.trackmate_panel.search_radius.value)
        max_frame_gap = int(self.trackmate_panel.max_frame_gap.value)
        allow_track_splitting = bool(self.trackmate_panel.allow_splitting.value)
        splitting_max_distance = float(self.trackmate_panel.splitting_max_distance.value)
        allow_track_merging = bool(self.trackmate_panel.allow_merging.value)

        masks_path = work_dir / "masks_stack.tiff"
        if not masks_path.exists():
            raise FileNotFoundError("No saved segmentation found. Run segmentation first.")

        self._set_progress(0, maximum=0, fmt="Running TrackMate...")
        self._append_log("Running TrackMate (UI may be unresponsive)...")
        if not allow_track_splitting:
            self._append_log("[WARN] TrackMate splitting is disabled \u2014 mother/daughter lineage will not be inferred.")
        trackmate_output = generate_trackmate_labels(masks_path=masks_path, output_directory=work_dir, initial_search_radius=initial_search_radius, search_radius=search_radius, max_frame_gap=max_frame_gap, allow_track_splitting=allow_track_splitting, splitting_max_distance=splitting_max_distance, allow_track_merging=allow_track_merging, imagej_instance=self._imagej_instance)
        self._imagej_instance = trackmate_output["imagej_instance"]
        tracks_dataframe = trackmate_output["trackmate_tracks_df"]
        linked_labels = trackmate_output["linked_labels"]
        num_tracks = int(tracks_dataframe["track_id"].nunique())
        num_division_events = int(tracks_dataframe.drop_duplicates("track_id")["parent_track_id"].notna().sum()) if "parent_track_id" in tracks_dataframe.columns else 0

        self._set_progress(100, fmt="Tracking saved")
        self._append_log(f"[OK] Saved tracking -> {trackmate_output['linked_labels_path']}")
        self._append_log(f"  Tracked {num_tracks} cells, {len(tracks_dataframe)} points, {num_division_events} division event(s)")

        return {"tracks_df": tracks_dataframe, "linked_labels": linked_labels, "linked_labels_path": trackmate_output["linked_labels_path"], "tracks_csv": trackmate_output["tracks_csv"]}

    def _run_analysis_stage(self, tiff_path: Path, work_dir: Path, mode: str):
        """Run either the persistent or the snapshot analysis from saved tracking outputs.

        Loads the linked-label TIFF and the original fluorescence channels,
        thresholds each fluorescence channel, computes per-cell positivity,
        assigns fates and writes all per-cell, per-frame, per-cutoff and
        plot outputs to ``work_dir``. Returns a one-row summary dict
        suitable for batch aggregation plus a ``run_data`` dict that the
        viewer can render.
        """
        _, _, _, fluorophore_threshold_methods = self._read_fluorophore_config()
        blur_sigma = float(self.analysis_panel.blur_sigma.value)

        linked_labels_path = work_dir / "linked_labels_trackmate.tiff"
        if not linked_labels_path.exists():
            raise FileNotFoundError("No saved tracking found. Run tracking first.")

        brightfield_stack, fluorophore_stacks, fluorophore_names, _ = self._load_image_channels(tiff_path)
        linked_labels = tifffile.imread(str(linked_labels_path)).astype(np.uint32)

        self._set_progress(10, fmt="Fluorescence segmentation...")
        self._append_log("Segmenting fluorescence channels...")
        fluorescence_segmentation = segment_fluorescence(fluorophore_stacks, blur_sigma=blur_sigma, threshold_method=fluorophore_threshold_methods)
        positive_label_stacks = {fluorophore_name: fluorescence_segmentation[fluorophore_name]["positive_labels"] for fluorophore_name in fluorophore_names}
        frame_cell_positive_area, positive_cell_labels = compute_cell_positivity(linked_labels, positive_label_stacks, fluorophore_names)

        self._set_progress(45, fmt=f"Assigning fates ({mode})...")
        self._append_log(f"Assigning fates (mode={mode})...")
        locked_labels = None
        assignments_dataframe = None
        snapshot_dataframe = None

        # Load lineage info from the tracks CSV (if it exists). This is
        # used to enrich the per-cell output tables with mother/daughter
        # relationships.
        tracks_csv_path = work_dir / "trackmate_tracks.csv"
        lineage_columns = ["track_id", "lineage_id", "parent_track_id", "generation"]
        if tracks_csv_path.exists():
            tracks_raw = pd.read_csv(tracks_csv_path)
            lineage_lookup = tracks_raw.drop_duplicates("track_id")[lineage_columns].copy() if "lineage_id" in tracks_raw.columns else pd.DataFrame(columns=lineage_columns)
        else:
            lineage_lookup = pd.DataFrame(columns=lineage_columns)

        if mode == "persistent":
            assignments_dataframe, locked_labels, persistent_per_frame = assign_persistent_fates(linked_labels, frame_cell_positive_area)
            assignments_dataframe = assignments_dataframe.sort_values("label_id").reset_index(drop=True)
            assignments_dataframe.to_csv(work_dir / "assignments_persistent.csv", index=False)
            if len(persistent_per_frame) > 0:
                persistent_per_frame_path = work_dir / "persistent_per_frame.csv"
                persistent_per_frame.to_csv(persistent_per_frame_path, index=False)
                self._append_log(f"  Per-frame metrics -> {persistent_per_frame_path.name}")
            persistent_by_cell = assignments_dataframe.rename(columns={"label_id": "cell_id"}).copy()
            persistent_by_cell["track_id"] = persistent_by_cell["cell_id"].astype(int) - 1
            if len(lineage_lookup) > 0:
                persistent_by_cell = persistent_by_cell.merge(lineage_lookup, on="track_id", how="left")
            persistent_by_cell.to_csv(work_dir / "persistent_by_cell.csv", index=False)
            self._append_log(f"  Fates: {dict(assignments_dataframe['fate'].value_counts())}")
        else:
            snapshot_dataframe = assign_snapshot_fates(linked_labels, frame_cell_positive_area)
            snapshot_dataframe = snapshot_dataframe.sort_values(["label_id", "frame"]).reset_index(drop=True)
            snapshot_dataframe.to_csv(work_dir / "snapshot.csv", index=False)
            snapshot_by_cell_long = snapshot_dataframe.rename(columns={"label_id": "cell_id"}).copy()
            snapshot_by_cell_long["track_id"] = snapshot_by_cell_long["cell_id"].astype(int) - 1
            if len(lineage_lookup) > 0:
                snapshot_by_cell_long = snapshot_by_cell_long.merge(lineage_lookup, on="track_id", how="left")
            snapshot_by_cell_long.to_csv(work_dir / "snapshot_by_cell_long.csv", index=False)

            # Wide-format snapshot table: one row per cell, one column per frame.
            snapshot_wide = snapshot_by_cell_long.pivot(index="cell_id", columns="frame", values="category").reset_index().sort_values("cell_id")
            snapshot_wide.columns = ["cell_id" if column_name == "cell_id" else f"frame_{int(column_name)}_category" for column_name in snapshot_wide.columns]
            if "lineage_id" in snapshot_by_cell_long.columns:
                lineage_per_cell = snapshot_by_cell_long.drop_duplicates("cell_id")[["cell_id", "lineage_id", "parent_track_id", "generation"]]
                snapshot_wide = snapshot_wide.merge(lineage_per_cell, on="cell_id", how="left")
                front_columns = ["cell_id", "lineage_id", "parent_track_id", "generation"]
                snapshot_wide = snapshot_wide[front_columns + [column_name for column_name in snapshot_wide.columns if column_name not in front_columns]]
            snapshot_wide.to_csv(work_dir / "snapshot_by_cell_wide.csv", index=False)
            self._append_log(f"  Categories: {sorted(snapshot_dataframe['category'].unique())}")

        self._set_progress(75, fmt="Computing percentages...")
        num_frames = linked_labels.shape[0]
        file_stem = tiff_path.stem
        if mode == "persistent":
            summary_dataframe = compute_persistent_percentages(assignments_dataframe, num_frames, fluorophore_names)
            summary_figure, _ = plot_persistent_percentages(summary_dataframe, fluorophore_names, title=file_stem)
            summary_csv_path = work_dir / "percentages_persistent.csv"
            summary_pdf_path = work_dir / "percentages_persistent.pdf"
            for cutoff_percentage in FRAME_PRESENCE_CUTOFF_PERCENTAGES:
                filtered_assignments = filter_persistent_by_frame_presence(assignments_dataframe, linked_labels, num_frames, cutoff_percentage)
                num_filtered_cells = len(filtered_assignments)
                if num_filtered_cells == 0:
                    self._append_log(f"  No cells at \u2265{cutoff_percentage}% frame presence, skipping")
                    continue
                filtered_summary = compute_persistent_percentages(filtered_assignments, num_frames, fluorophore_names)
                cutoff_figure, _ = plot_persistent_percentages(filtered_summary, fluorophore_names, title=f"{file_stem} \u2014 \u2265{cutoff_percentage}% frames (n={num_filtered_cells} cells)")
                cutoff_figure.savefig(str(work_dir / f"percentages_persistent_{cutoff_percentage}pct.pdf"), bbox_inches="tight")
                plt.close(cutoff_figure)
                filtered_summary.to_csv(work_dir / f"percentages_persistent_{cutoff_percentage}pct.csv", index=False)
        else:
            summary_dataframe, snapshot_categories = compute_snapshot_percentages(snapshot_dataframe, num_frames)
            summary_figure, _ = plot_snapshot_percentages(summary_dataframe, snapshot_categories, title=file_stem)
            summary_csv_path = work_dir / "percentages_snapshot.csv"
            summary_pdf_path = work_dir / "percentages_snapshot.pdf"
            tracks_for_plot = pd.read_csv(tracks_csv_path) if tracks_csv_path.exists() else pd.DataFrame(columns=["track_id", "t", "y", "x", "quality"])
            for cutoff_percentage in FRAME_PRESENCE_CUTOFF_PERCENTAGES:
                filtered_tracks, filtered_snapshot = filter_by_frame_presence(tracks_for_plot, snapshot_dataframe, num_frames, cutoff_percentage)
                num_filtered_cells = filtered_snapshot["label_id"].nunique() if len(filtered_snapshot) > 0 else 0
                filtered_summary, filtered_categories = compute_snapshot_percentages(filtered_snapshot, num_frames)
                cutoff_figure, _ = plot_snapshot_percentages(filtered_summary, filtered_categories, title=f"{file_stem} \u2014 \u2265{cutoff_percentage}% frames (n={num_filtered_cells} cells)")
                cutoff_figure.savefig(str(work_dir / f"percentages_snapshot_{cutoff_percentage}pct.pdf"), bbox_inches="tight")
                plt.close(cutoff_figure)
                filtered_summary.to_csv(work_dir / f"percentages_snapshot_{cutoff_percentage}pct.csv", index=False)

                trajectory_figure, _ = plot_snapshot_trajectories(filtered_tracks, filtered_snapshot, title=f"{file_stem} \u2014 trajectories \u2265{cutoff_percentage}% frames (n={num_filtered_cells} cells)")
                trajectory_figure.savefig(str(work_dir / f"snapshot_trajectories_{cutoff_percentage}pct.pdf"), bbox_inches="tight")
                plt.close(trajectory_figure)

                timeline_figure, _ = plot_snapshot_cell_timelines(filtered_snapshot, tracks_dataframe=filtered_tracks, title=f"{file_stem} \u2014 cell timelines \u2265{cutoff_percentage}% frames (n={num_filtered_cells} cells)")
                timeline_figure.savefig(str(work_dir / f"snapshot_timelines_{cutoff_percentage}pct.pdf"), bbox_inches="tight")
                plt.close(timeline_figure)

        summary_dataframe.to_csv(summary_csv_path, index=False)
        summary_figure.savefig(str(summary_pdf_path), bbox_inches="tight")
        plt.close(summary_figure)
        self._set_progress(100, fmt=f"{mode.capitalize()} analysis saved")
        self._append_log(f"[OK] Saved {mode} outputs -> {summary_csv_path.name}, {summary_pdf_path.name}, plus cutoff plots at 30/40/50/60%")

        tracks_dataframe = pd.read_csv(tracks_csv_path) if tracks_csv_path.exists() else pd.DataFrame(columns=["track_id", "t", "y", "x", "quality"])
        if len(tracks_dataframe) > 0:
            tracks_by_cell = tracks_dataframe.sort_values(["track_id", "t"]).copy()
            tracks_by_cell["cell_id"] = tracks_by_cell["track_id"].astype(int) + 1
            tracks_by_cell["frame"] = tracks_by_cell["t"].astype(int)
            tracks_by_cell_path = work_dir / "trackmate_tracks_by_cell.csv"
            tracks_by_cell.to_csv(tracks_by_cell_path, index=False)
            self._append_log(f"[OK] Saved plot-ready tracks -> {tracks_by_cell_path.name}")

        summary_record = {"filename": tiff_path.name, "analysis_mode": mode, "n_frames": num_frames, "n_tracked_cells": int(tracks_dataframe["track_id"].nunique()) if len(tracks_dataframe) > 0 else 0}
        if mode == "persistent":
            for fluorophore_name in fluorophore_names:
                summary_record[f"n_{fluorophore_name}"] = int((assignments_dataframe["fate"] == fluorophore_name).sum())
                summary_record[f"final_pct_{fluorophore_name}"] = float(summary_dataframe[f"{fluorophore_name}_pct"].iloc[-1])
            summary_record["n_negative"] = int((assignments_dataframe["fate"] == "negative").sum())
            summary_record["final_total_pct"] = float(summary_dataframe["total_positive_pct"].iloc[-1])
        else:
            last_frame_snapshot = snapshot_dataframe[snapshot_dataframe["frame"] == num_frames - 1]
            for category in sorted(snapshot_dataframe["category"].unique()):
                summary_record[f"final_pct_{category}"] = 100.0 * (last_frame_snapshot["category"] == category).sum() / max(len(last_frame_snapshot), 1)

        masks_stack_path = work_dir / "masks_stack.tiff"
        viewer_data = {"bf_image": brightfield_stack, "fluor_images": fluorophore_stacks, "masks_stack": tifffile.imread(str(masks_stack_path)).astype(np.uint16) if masks_stack_path.exists() else np.zeros_like(linked_labels, dtype=np.uint16), "linked_labels": linked_labels, "positive_cell_labels": positive_cell_labels, "locked_labels": locked_labels, "tracks_df": tracks_dataframe, "mode": mode}
        return summary_record, viewer_data

    def _run_segmentation_only(self):
        """Run only Cellpose segmentation and write masks for the active image."""
        self._set_buttons_enabled(False)
        try:
            tiff_path, _, work_dir = self._resolve_single_run_paths()
            segmentation_outputs = self._run_segmentation_stage(tiff_path, work_dir)
            self._show_in_viewer({
                "bf_image": segmentation_outputs["bf_image"],
                "fluor_images": segmentation_outputs["fluor_images"],
                "masks_stack": segmentation_outputs["masks_stack"],
                "linked_labels": np.zeros_like(segmentation_outputs["masks_stack"], dtype=np.uint32),
                "positive_cell_labels": {},
                "locked_labels": None,
                "tracks_df": pd.DataFrame(columns=["track_id", "t", "y", "x", "quality"]),
                "mode": "snapshot",
            })
        except Exception as exception:
            self._set_progress(0, fmt="Error")
            LOGGER.exception("Segmentation-only stage failed")
            self._append_log(f"[ERROR] {type(exception).__name__}: {exception}\n{traceback.format_exc()}")
        finally:
            self._set_buttons_enabled(True)

    def _run_tracking_only(self):
        """Run only TrackMate tracking using masks already saved in the working directory."""
        self._set_buttons_enabled(False)
        try:
            tiff_path, _, work_dir = self._resolve_single_run_paths()
            brightfield_stack, fluorophore_stacks, _, _ = self._load_image_channels(tiff_path)
            tracking_outputs = self._run_tracking_stage(work_dir)
            masks_path = work_dir / "masks_stack.tiff"
            masks_stack = tifffile.imread(str(masks_path)).astype(np.uint16) if masks_path.exists() else np.zeros_like(tracking_outputs["linked_labels"], dtype=np.uint16)
            self._show_in_viewer({
                "bf_image": brightfield_stack,
                "fluor_images": fluorophore_stacks,
                "masks_stack": masks_stack,
                "linked_labels": tracking_outputs["linked_labels"],
                "positive_cell_labels": {},
                "locked_labels": None,
                "tracks_df": tracking_outputs["tracks_df"],
                "mode": "snapshot",
            })
        except Exception as exception:
            self._set_progress(0, fmt="Error")
            LOGGER.exception("Tracking-only stage failed")
            self._append_log(f"[ERROR] {type(exception).__name__}: {exception}\n{traceback.format_exc()}")
        finally:
            self._set_buttons_enabled(True)

    def _run_persistent_analysis(self):
        """Run persistent fate assignment from previously saved tracking outputs."""
        self._set_buttons_enabled(False)
        try:
            tiff_path, _, work_dir = self._resolve_single_run_paths()
            summary_record, viewer_data = self._run_analysis_stage(tiff_path, work_dir, mode="persistent")
            self._results.append(summary_record)
            self._results_df = pd.DataFrame(self._results)
            self._show_in_viewer(viewer_data)
        except Exception as exception:
            self._set_progress(0, fmt="Error")
            LOGGER.exception("Persistent-analysis stage failed")
            self._append_log(f"[ERROR] {type(exception).__name__}: {exception}\n{traceback.format_exc()}")
        finally:
            self._set_buttons_enabled(True)

    def _run_snapshot_analysis(self):
        """Run snapshot fate assignment from previously saved tracking outputs."""
        self._set_buttons_enabled(False)
        try:
            tiff_path, _, work_dir = self._resolve_single_run_paths()
            summary_record, viewer_data = self._run_analysis_stage(tiff_path, work_dir, mode="snapshot")
            self._results.append(summary_record)
            self._results_df = pd.DataFrame(self._results)
            self._show_in_viewer(viewer_data)
        except Exception as exception:
            self._set_progress(0, fmt="Error")
            LOGGER.exception("Snapshot-analysis stage failed")
            self._append_log(f"[ERROR] {type(exception).__name__}: {exception}\n{traceback.format_exc()}")
        finally:
            self._set_buttons_enabled(True)

    def _run_all_single_image(self):
        """Run segmentation, tracking and both analyses on the active image."""
        self._set_buttons_enabled(False)
        try:
            tiff_path, _, work_dir = self._resolve_single_run_paths()
            self._append_log("[INFO] Running full pipeline (all stages)...")
            self._run_segmentation_stage(tiff_path, work_dir)
            self._run_tracking_stage(work_dir)
            persistent_summary, persistent_viewer_data = self._run_analysis_stage(tiff_path, work_dir, mode="persistent")
            snapshot_summary, _ = self._run_analysis_stage(tiff_path, work_dir, mode="snapshot")
            self._results.extend([persistent_summary, snapshot_summary])
            self._results_df = pd.DataFrame(self._results)
            self._show_in_viewer(persistent_viewer_data)
            self._append_log(f"[OK] Full pipeline complete: {tiff_path.name}")
        except Exception as exception:
            self._set_progress(0, fmt="Error")
            LOGGER.exception("Full single-image pipeline failed")
            self._append_log(f"[ERROR] {type(exception).__name__}: {exception}\n{traceback.format_exc()}")
        finally:
            self._set_buttons_enabled(True)

    # -- viewer -------------------------------------------------------------

    def _show_in_viewer(self, viewer_data: dict):
        """Populate the napari viewer with the pipeline outputs from one image."""
        self.viewer.layers.clear()
        self.viewer.add_image(viewer_data["bf_image"], name="Brightfield", colormap="gray", blending="translucent", opacity=0.7)

        colour_assignments = assign_colours(list(viewer_data["fluor_images"].keys()))
        for fluorophore_name, fluorophore_image in viewer_data["fluor_images"].items():
            napari_colormap = colour_assignments[fluorophore_name]["napari"]
            self.viewer.add_image(fluorophore_image, name=f"{fluorophore_name} fluorescence", colormap=napari_colormap, blending="additive")

        for fluorophore_name, positive_label_image in viewer_data["positive_cell_labels"].items():
            add_coloured_labels(self.viewer, positive_label_image, name=f"{fluorophore_name} positive cells", base_colour=get_fluor_base_colour(fluorophore_name), opacity=0.35)

        add_coloured_labels(self.viewer, viewer_data["linked_labels"], name="Linked labels", base_colour="gold", opacity=0.20)

        if viewer_data["mode"] == "persistent" and viewer_data["locked_labels"] is not None:
            for fluorophore_name, locked_label_image in viewer_data["locked_labels"].items():
                add_coloured_labels(self.viewer, locked_label_image, name=f"Persistent {fluorophore_name}", base_colour=get_fluor_base_colour(fluorophore_name), opacity=0.60)

        add_coloured_labels(self.viewer, viewer_data["masks_stack"].astype(np.uint32), name="Cellpose masks", base_colour="slategray", opacity=0.15)

        tracks_dataframe = viewer_data["tracks_df"]
        if len(tracks_dataframe) > 0:
            tracks_array = tracks_dataframe[["track_id", "t", "y", "x"]].sort_values(["track_id", "t"]).to_numpy(dtype=float)
            self.viewer.add_tracks(tracks_array, name="Tracks", tail_length=50, opacity=0.8)

    # -- button handlers ----------------------------------------------------

    def _analyse_folder(self):
        """Batch-process every TIFF in a folder, save combined CSV."""
        folder = Path(str(self.file_panel.batch_folder.value))
        if not folder.exists() or not folder.is_dir():
            self._append_log(
                "[ERROR] Select a valid batch folder in the Files panel."
            )
            return
        out_dir = Path(str(self.file_panel.output_directory.value))
        if not str(out_dir).strip() or str(out_dir) == ".":
            out_dir = folder / "cell_death_output"

        tiff_files = sorted(
            f
            for f in folder.iterdir()
            if f.suffix.lower() in (".tif", ".tiff") and f.is_file()
        )
        if not tiff_files:
            self._append_log("[WARN] No .tif / .tiff files found in folder.")
            return

        num_files = len(tiff_files)
        self._append_log(f"[INFO] Batch: {num_files} files in {folder.name}")
        batch_results: List[dict] = []

        self._set_buttons_enabled(False)
        try:
            for file_index, tiff_file_path in enumerate(tiff_files):
                self._append_log(f"--- File {file_index + 1}/{num_files}: {tiff_file_path.name} ---")
                try:
                    work_dir = out_dir / tiff_file_path.stem
                    work_dir.mkdir(parents=True, exist_ok=True)
                    self._run_segmentation_stage(tiff_file_path, work_dir)
                    self._run_tracking_stage(work_dir)
                    analysis_mode = str(self.analysis_panel.analysis_mode.value)
                    summary_record, _ = self._run_analysis_stage(tiff_file_path, work_dir, analysis_mode)
                    batch_results.append(summary_record)
                except Exception as exception:
                    batch_results.append({"filename": tiff_file_path.name, "error": str(exception)})
                    LOGGER.exception("Batch file %s failed", tiff_file_path.name)
                    self._append_log(f"  FAILED: {type(exception).__name__}: {exception}\n{traceback.format_exc()}")
                self._set_progress(file_index + 1, maximum=num_files, fmt=f"Files: {file_index + 1}/{num_files}")

            self._results.extend(batch_results)
            self._results_df = pd.DataFrame(self._results)
            out_dir.mkdir(parents=True, exist_ok=True)
            summary_path = out_dir / "batch_summary.csv"
            self._results_df.to_csv(summary_path, index=False)
            self._append_log(f"[OK] Batch done. Summary -> {summary_path}")
        finally:
            self._set_buttons_enabled(True)

    def _save_results(self):
        """Save accumulated results to a user-chosen CSV file."""
        if self._results_df is None or self._results_df.empty:
            self._append_log("[WARN] No results to save yet.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            None,
            "Save results CSV",
            str(Path.home()),
            "CSV files (*.csv)",
        )
        if not filename:
            self._append_log("[INFO] Save cancelled.")
            return
        if not filename.lower().endswith(".csv"):
            filename += ".csv"

        self._results_df.to_csv(filename, index=False)
        self._append_log(f"[OK] Saved results -> {filename}")


# ---------------------------------------------------------------------------
#  Launch helpers
# ---------------------------------------------------------------------------

def launch():
    """Create the GUI and return the app instance (for notebook use)."""
    return FluoroFateApp()


def main():
    """Launch the standalone GUI."""
    launch()
    napari.run()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
