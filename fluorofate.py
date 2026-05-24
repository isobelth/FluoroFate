"""FluoroFate — napari + magicgui GUI.

Pipeline: Cellpose segmentation -> TrackMate tracking -> fluorescence
thresholding -> fate assignment (persistent AND snapshot, both always
run) -> percentage trajectories.

Two input modes, one Run button:
  * Single image - full pipeline, all layers shown in napari.
  * Folder      - batch every TIFF in a folder -> ``batch_summary.csv``.

Conventions
-----------
``cell_id == label_id == track_id + 1`` because label 0 is reserved
for background in label images.
"""

import functools
import json
import logging
import platform
import subprocess
import sys
import traceback
import warnings
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import tifffile
from magicgui import magicgui
from magicgui.widgets import TextEdit
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication, QHBoxLayout, QProgressBar, QPushButton, QVBoxLayout, QWidget

from utils import running_in_notebook
from colours import assign_colours, get_fluor_base_colour, add_coloured_labels
from measurement import compute_cell_positivity, compute_per_cell_intensity_area
from segmentation import cellpose_live_segmentation, segment_fluorescence
from tracking import generate_trackmate_labels
from fate_assignment import (assign_persistent_fates, assign_snapshot_fates,
                             compute_persistent_percentages, compute_snapshot_percentages)
from plotting import (plot_persistent_percentages, plot_snapshot_percentages,
                      plot_snapshot_trajectories, plot_snapshot_cell_timelines)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger("fluorofate")

CELLPOSE_DEFAULT_DIAMETER = None
CELLPOSE_DEFAULT_FLOW_THRESHOLD = 0.4
CELLPOSE_DEFAULT_CELLPROB_THRESHOLD = 0.0
THRESHOLD_CHOICES = ["mean", "minimum", "yen", "otsu", "triangle"]
MODEL_CHOICES = ["cpsam", "cyto3", "cyto2", "cyto", "nuclei"]

# Output filenames (single source of truth). Used everywhere a file is read
# or written, and used to build outputs_README.md.
OUTPUT_MASKS = "masks_stack.tiff"
OUTPUT_LINKED = "linked_labels_trackmate.tiff"
OUTPUT_TRACKS = "trackmate_tracks.csv"
OUTPUT_PER_FRAME_CELLS = "per_frame_cells.csv"
OUTPUT_PCT_PERSISTENT_PDF = "percentages_persistent.pdf"
OUTPUT_PCT_SNAPSHOT_PDF = "percentages_snapshot.pdf"
OUTPUT_SNAPSHOT_TRAJECTORIES = "snapshot_trajectories.pdf"
OUTPUT_SNAPSHOT_TIMELINES = "snapshot_timelines.pdf"
OUTPUT_RUN_CONFIG = "run_config.json"
OUTPUT_RUN_LOG = "run.log"
OUTPUT_README = "outputs_README.md"
OUTPUT_BATCH_SUMMARY = "batch_summary.csv"

OUTPUT_DESCRIPTIONS: Dict[str, str] = {
    OUTPUT_MASKS: "Cellpose segmentation masks (uint16).",
    OUTPUT_LINKED: "TrackMate-linked labels (cell ID stable across frames).",
    OUTPUT_TRACKS: "TrackMate spot-level output (intermediate; required for staged Tracking \u2192 Analysis runs).",
    OUTPUT_PER_FRAME_CELLS: "Per-cell, per-frame analysis output: cell area, raw fluorescence sums, thresholded positive areas, persistent fate flags, snapshot positivity flags, and lineage columns.",
    OUTPUT_PCT_PERSISTENT_PDF: "Cumulative % positive over time (persistent mode).",
    OUTPUT_PCT_SNAPSHOT_PDF: "Per-frame category % (snapshot mode).",
    OUTPUT_SNAPSHOT_TRAJECTORIES: "XY trajectories coloured by snapshot category.",
    OUTPUT_SNAPSHOT_TIMELINES: "Per-cell horizontal-bar timeline coloured by category.",
    OUTPUT_RUN_CONFIG: "All parameters used for this run (full provenance).",
    OUTPUT_RUN_LOG: "Full log for this run.",
    OUTPUT_README: "This file: description of every output.",
}


class LogSignal(QObject):
    message = Signal(str)


class QtLogHandler(logging.Handler):
    """Route log records to a Qt TextEdit via a Qt signal (thread-safe)."""

    def __init__(self, append_callback: Callable[[str], None]):
        super().__init__()
        self.bridge = LogSignal()
        self.bridge.message.connect(append_callback)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.bridge.message.emit(self.format(record))
        except Exception:
            self.handleError(record)


def gui_action(label: str) -> Callable:
    """Wrap a stage handler in disable-buttons + progress reset + try/except."""
    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapped(self, *args, **kwargs):
            self.set_buttons_enabled(False)
            self.set_progress(0, fmt=f"{label}: starting...")
            LOGGER.info("=== %s ===", label)
            try:
                return method(self, *args, **kwargs)
            except Exception as exception:
                self.set_progress(0, fmt="Error")
                LOGGER.error("%s failed: %s: %s", label, type(exception).__name__, exception)
                LOGGER.debug(traceback.format_exc())
            finally:
                self.set_buttons_enabled(True)
        return wrapped
    return decorator


class FluoroFateApp:
    """Napari-based GUI wrapping the full FluoroFate pipeline."""

    @staticmethod
    def inputs_signature_template(input_mode: str = "Single image", single_tiff: Path = Path(),
                                  batch_folder: Path = Path(), output_directory: Path = Path(),
                                  brightfield_channel: int = 0,
                                  fluor_1_name: str = "Green", fluor_1_channel: int = 0, fluor_1_threshold: str = "otsu",
                                  fluor_2_name: str = "", fluor_2_channel: int = 0, fluor_2_threshold: str = "otsu",
                                  fluor_3_name: str = "", fluor_3_channel: int = 0, fluor_3_threshold: str = "otsu"):
        pass

    @staticmethod
    def params_signature_template(cellpose_model: str = "cpsam", custom_model_file: Path = Path(),
                                  min_cell_size: int = 15, use_gpu: bool = True,
                                  initial_search_radius: float = 30.0, search_radius: float = 150.0,
                                  max_frame_gap: int = 2, allow_splitting: bool = True,
                                  splitting_max_distance: float = 15.0, allow_merging: bool = False,
                                  blur_sigma: float = 1.0):
        pass

    def __init__(self):
        self.viewer = napari.Viewer(title="FluoroFate")
        qt_app = QApplication.instance()
        if qt_app is not None:
            qt_app.setQuitOnLastWindowClosed(not running_in_notebook())

        self.imagej_instance = None
        self.image_cache: Dict[Path, np.ndarray] = {}
        self.workdir_file_handler: Optional[logging.FileHandler] = None
        self.results: List[dict] = []
        self.last_refreshed_tiff_path: Optional[Path] = None

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Idle")
        self.log_widget = TextEdit(value="")
        try:
            self.log_widget.native.setReadOnly(True)
        except Exception:
            pass
        self.log_widget.min_height = 140
        self.log_widget.max_height = 400

        self.gui_log_handler = QtLogHandler(self.append_log)
        self.gui_log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self.gui_log_handler)

        self.inputs_panel = magicgui(
            self.inputs_signature_template,
            input_mode={"label": "Input mode", "choices": ["Single image", "Folder of images"], "widget_type": "RadioButtons"},
            single_tiff={"label": "Single TIFF", "mode": "r", "filter": "*.tif *.tiff"},
            batch_folder={"label": "Batch folder", "mode": "d"},
            output_directory={"label": "Output directory", "mode": "d"},
            brightfield_channel={"label": "Brightfield channel", "widget_type": "ComboBox", "choices": [0]},
            fluor_1_name={"label": "Fluorophore 1 name"},
            fluor_1_channel={"label": "Fluorophore 1 channel", "widget_type": "ComboBox", "choices": [0]},
            fluor_1_threshold={"label": "Fluorophore 1 threshold", "choices": THRESHOLD_CHOICES},
            fluor_2_name={"label": "Fluorophore 2 (blank=none)"},
            fluor_2_channel={"label": "Fluorophore 2 channel", "widget_type": "ComboBox", "choices": [0]},
            fluor_2_threshold={"label": "Fluorophore 2 threshold", "choices": THRESHOLD_CHOICES},
            fluor_3_name={"label": "Fluorophore 3 (blank=none)"},
            fluor_3_channel={"label": "Fluorophore 3 channel", "widget_type": "ComboBox", "choices": [0]},
            fluor_3_threshold={"label": "Fluorophore 3 threshold", "choices": THRESHOLD_CHOICES},
            call_button=False,
        )
        self.params_panel = magicgui(
            self.params_signature_template,
            cellpose_model={"label": "Cellpose model", "choices": MODEL_CHOICES},
            custom_model_file={"label": "Custom model (optional)", "mode": "r", "filter": "*.pt *.pth"},
            min_cell_size={"label": "Min cell size (px)", "min": 0, "max": 2000},
            use_gpu={"label": "Use GPU"},
            initial_search_radius={"label": "TM init search radius", "min": 1.0, "max": 500.0, "step": 1.0},
            search_radius={"label": "TM search radius", "min": 1.0, "max": 2000.0, "step": 5.0},
            max_frame_gap={"label": "TM max frame gap", "min": 0, "max": 50},
            allow_splitting={"label": "TM allow splitting"},
            splitting_max_distance={"label": "TM splitting max distance", "min": 1.0, "max": 2000.0, "step": 5.0},
            allow_merging={"label": "TM allow merging"},
            blur_sigma={"label": "Fluor. blur sigma", "min": 0.1, "max": 20.0, "step": 0.1},
            call_button=False,
        )

        self.inputs_panel.input_mode.changed.connect(self.on_input_mode_changed)
        self.inputs_panel.single_tiff.changed.connect(self.on_single_tiff_changed)
        self.inputs_panel.batch_folder.changed.connect(self.on_batch_folder_changed)
        self.params_panel.custom_model_file.changed.connect(self.on_custom_model_file_changed)
        self.on_input_mode_changed()
        self.on_custom_model_file_changed()

        self.action_buttons: List[QPushButton] = []
        run_all_button = self.make_button("Run All", self.on_run_all_clicked)
        clear_log_button = self.make_button("Clear log", self.clear_log, is_action_button=False)

        run_widget = QWidget()
        run_layout = QVBoxLayout(run_widget)
        run_layout.addWidget(run_all_button)
        run_layout.addWidget(self.progress_bar)
        log_row = QHBoxLayout()
        log_row.addStretch(1)
        log_row.addWidget(clear_log_button)
        run_layout.addLayout(log_row)
        run_layout.addWidget(self.log_widget.native)

        self.viewer.window.add_dock_widget(self.inputs_panel, name="Inputs", area="right")
        self.viewer.window.add_dock_widget(self.params_panel, name="Parameters", area="right")
        self.viewer.window.add_dock_widget(run_widget, name="Run & Log", area="right")
        LOGGER.info("FluoroFate ready.")

    # ---- Widget plumbing ---------------------------------------------------

    def make_button(self, label: str, callback: Callable, *, is_action_button: bool = True) -> QPushButton:
        button = QPushButton(label)
        button.clicked.connect(lambda _checked=False: callback())
        if is_action_button:
            self.action_buttons.append(button)
        return button

    def set_buttons_enabled(self, enabled: bool) -> None:
        for button in self.action_buttons:
            button.setEnabled(enabled)

    def append_log(self, message: str) -> None:
        self.log_widget.value = (self.log_widget.value.rstrip() + "\n" + message) if self.log_widget.value else message
        try:
            cursor = self.log_widget.native.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.log_widget.native.setTextCursor(cursor)
        except Exception:
            pass

    def clear_log(self) -> None:
        self.log_widget.value = ""

    def set_progress(self, value: int, maximum: int = 100, fmt: str = "") -> None:
        self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(min(value, maximum) if maximum > 0 else 0)
        if fmt:
            self.progress_bar.setFormat(fmt)
        qt_app = QApplication.instance()
        if qt_app is not None:
            qt_app.processEvents()

    # ---- Dynamic widget hooks ---------------------------------------------

    def on_input_mode_changed(self, *event_args) -> None:
        is_single = str(self.inputs_panel.input_mode.value) == "Single image"
        self.inputs_panel.single_tiff.enabled = is_single
        self.inputs_panel.batch_folder.enabled = not is_single

    def on_custom_model_file_changed(self, *event_args) -> None:
        use_custom = self.custom_model_is_selected()
        self.params_panel.cellpose_model.enabled = not use_custom
        self.params_panel.cellpose_model.tooltip = "Disabled because a custom model file is selected." if use_custom else ""

    def on_single_tiff_changed(self, *event_args) -> None:
        self.refresh_channel_choices_from(self.current_single_tiff_path())

    def on_batch_folder_changed(self, *event_args) -> None:
        folder = Path(str(self.inputs_panel.batch_folder.value))
        if folder.exists() and folder.is_dir():
            first_tiff = next((p for p in sorted(folder.iterdir()) if p.suffix.lower() in (".tif", ".tiff") and p.is_file()), None)
            if first_tiff is not None:
                self.refresh_channel_choices_from(first_tiff)

    def refresh_channel_choices_from(self, tiff_path: Optional[Path]) -> None:
        if tiff_path is None or not tiff_path.exists():
            return
        if self.last_refreshed_tiff_path is not None and tiff_path.resolve() == self.last_refreshed_tiff_path:
            return  # No-op guard: avoids resetting user-picked channels on spurious widget events.
        try:
            with tifffile.TiffFile(str(tiff_path)) as tif:
                shape = tif.series[0].shape
        except Exception as exception:
            LOGGER.warning("Could not read shape of %s: %s", tiff_path.name, exception)
            return
        num_channels = shape[1] if len(shape) >= 4 else (shape[0] if len(shape) == 3 else 1)
        choices = list(range(int(num_channels)))
        for widget_name in ("brightfield_channel", "fluor_1_channel", "fluor_2_channel", "fluor_3_channel"):
            widget = getattr(self.inputs_panel, widget_name)
            current = widget.value if widget.value in choices else choices[0]
            widget.choices = choices
            widget.value = current
        # Sensible default: brightfield = last channel (the common Incucyte/microscope layout).
        if num_channels >= 1:
            self.inputs_panel.brightfield_channel.value = num_channels - 1
        self.last_refreshed_tiff_path = tiff_path.resolve()
        LOGGER.info("Detected %d channel(s) in %s.", num_channels, tiff_path.name)

    # ---- Path + config resolution -----------------------------------------

    def current_single_tiff_path(self) -> Optional[Path]:
        candidate = Path(str(self.inputs_panel.single_tiff.value))
        return candidate if candidate.exists() and candidate.is_file() else None

    def custom_model_is_selected(self) -> bool:
        value = str(self.params_panel.custom_model_file.value).strip()
        return bool(value and value != ".")

    def resolve_paths(self) -> tuple:
        """Return ``(mode, tiff_path_or_None, folder_or_None, out_dir)``."""
        input_mode = str(self.inputs_panel.input_mode.value)
        out_dir_raw = str(self.inputs_panel.output_directory.value).strip()
        if not out_dir_raw or out_dir_raw == ".":
            raise ValueError("Select an output directory in the Inputs panel.")
        out_dir = Path(out_dir_raw)
        if input_mode == "Single image":
            tiff_path = self.current_single_tiff_path()
            if tiff_path is None:
                raise ValueError("Select a valid single TIFF file in the Inputs panel.")
            if tiff_path.suffix.lower() not in (".tif", ".tiff"):
                raise ValueError("Single image must be a .tif or .tiff file.")
            return "single", tiff_path, None, out_dir
        folder = Path(str(self.inputs_panel.batch_folder.value))
        if not folder.exists() or not folder.is_dir():
            raise ValueError("Select a valid batch folder in the Inputs panel.")
        return "folder", None, folder, out_dir

    def read_channel_config(self, num_channels_in_image: int) -> tuple:
        """Read inputs panel; return (fluor_names, channel_map, brightfield_channel, threshold_map)."""
        brightfield_channel = int(self.inputs_panel.brightfield_channel.value)
        fluorophore_names: List[str] = []
        channel_of: Dict[str, int] = {}
        threshold_of: Dict[str, str] = {}
        for name_attr, channel_attr, threshold_attr in (("fluor_1_name", "fluor_1_channel", "fluor_1_threshold"),
                                                        ("fluor_2_name", "fluor_2_channel", "fluor_2_threshold"),
                                                        ("fluor_3_name", "fluor_3_channel", "fluor_3_threshold")):
            fluorophore_name = str(getattr(self.inputs_panel, name_attr).value).strip()
            if not fluorophore_name:
                continue
            channel_index = int(getattr(self.inputs_panel, channel_attr).value)
            fluorophore_names.append(fluorophore_name)
            channel_of[fluorophore_name] = channel_index
            threshold_of[fluorophore_name] = str(getattr(self.inputs_panel, threshold_attr).value)
        if not fluorophore_names:
            raise ValueError("At least one fluorophore name must be provided.")
        seen_channels: Dict[int, str] = {}
        for fluorophore_name, channel_index in channel_of.items():
            if channel_index >= num_channels_in_image:
                raise ValueError(f"Channel {channel_index} for '{fluorophore_name}' is out of range (image has {num_channels_in_image} channels).")
            if channel_index in seen_channels:
                raise ValueError(f"Fluorophore '{fluorophore_name}' shares channel {channel_index} with '{seen_channels[channel_index]}'.")
            if channel_index == brightfield_channel:
                raise ValueError(f"Fluorophore '{fluorophore_name}' uses channel {channel_index}, which is also the brightfield channel.")
            seen_channels[channel_index] = fluorophore_name
        if brightfield_channel >= num_channels_in_image or brightfield_channel < 0:
            raise ValueError(f"Brightfield channel {brightfield_channel} is out of range (image has {num_channels_in_image} channels).")
        return fluorophore_names, channel_of, brightfield_channel, threshold_of

    def load_image_split(self, tiff_path: Path) -> tuple:
        """Load (cached) 4-D TIFF and split into brightfield + fluorophore stacks."""
        if tiff_path not in self.image_cache:
            self.image_cache[tiff_path] = tifffile.imread(str(tiff_path))
        image = self.image_cache[tiff_path]
        if image.ndim != 4:
            raise ValueError(f"Expected 4-D TIFF (T, C, Y, X), got {image.ndim}-D shape {image.shape}.")
        num_channels = image.shape[1]
        fluorophore_names, channel_of, brightfield_channel, threshold_of = self.read_channel_config(num_channels)
        brightfield_stack = image[:, brightfield_channel, :, :]
        fluorophore_stacks = {name: image[:, channel_of[name], :, :] for name in fluorophore_names}
        return brightfield_stack, fluorophore_stacks, fluorophore_names, threshold_of, brightfield_channel

    def gather_run_config(self, tiff_path: Path, work_dir: Path, fluor_names: List[str],
                          channel_of: Dict[str, int], threshold_of: Dict[str, str], brightfield_channel: int) -> dict:
        custom_model = self.custom_model_is_selected()
        try:
            git_output = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(Path(__file__).parent),
                                        capture_output=True, text=True, timeout=2)
            git_commit = git_output.stdout.strip() or None if git_output.returncode == 0 else None
        except Exception:
            git_commit = None
        package_versions: Dict[str, str] = {}
        for package_name in ("napari", "magicgui", "tifffile", "cellpose", "scikit-image", "pyimagej", "scyjava", "pandas", "numpy"):
            try:
                package_versions[package_name] = version(package_name)
            except PackageNotFoundError:
                package_versions[package_name] = "unknown"
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "input_image": str(tiff_path),
            "work_dir": str(work_dir),
            "brightfield_channel": brightfield_channel,
            "fluorophores": [{"name": name, "channel": channel_of[name], "threshold": threshold_of[name]} for name in fluor_names],
            "blur_sigma": float(self.params_panel.blur_sigma.value),
            "cellpose": {"model": "custom" if custom_model else str(self.params_panel.cellpose_model.value),
                         "custom_model_path": str(self.params_panel.custom_model_file.value) if custom_model else None,
                         "min_cell_size_px": int(self.params_panel.min_cell_size.value),
                         "use_gpu": bool(self.params_panel.use_gpu.value),
                         "diameter": CELLPOSE_DEFAULT_DIAMETER,
                         "flow_threshold": CELLPOSE_DEFAULT_FLOW_THRESHOLD,
                         "cellprob_threshold": CELLPOSE_DEFAULT_CELLPROB_THRESHOLD},
            "trackmate": {"initial_search_radius": float(self.params_panel.initial_search_radius.value),
                          "search_radius": float(self.params_panel.search_radius.value),
                          "max_frame_gap": int(self.params_panel.max_frame_gap.value),
                          "allow_splitting": bool(self.params_panel.allow_splitting.value),
                          "splitting_max_distance": float(self.params_panel.splitting_max_distance.value),
                          "allow_merging": bool(self.params_panel.allow_merging.value)},
            "environment": {"platform": platform.platform(), "python": sys.version.split()[0],
                            "packages": package_versions, "git_commit": git_commit},
        }

    # ---- Work-dir setup / log routing -------------------------------------

    def begin_workdir(self, tiff_path: Path, out_dir: Path) -> Path:
        work_dir = out_dir / tiff_path.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        self.end_workdir()
        file_handler = logging.FileHandler(work_dir / OUTPUT_RUN_LOG, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(file_handler)
        self.workdir_file_handler = file_handler
        LOGGER.info("Working directory: %s", work_dir)
        readme_lines = ["# FluoroFate outputs", "", "Files written into this directory:", ""]
        readme_lines += [f"- `{filename}` — {description}" for filename, description in OUTPUT_DESCRIPTIONS.items()]
        (work_dir / OUTPUT_README).write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
        return work_dir

    def end_workdir(self) -> None:
        if self.workdir_file_handler is not None:
            logging.getLogger().removeHandler(self.workdir_file_handler)
            try:
                self.workdir_file_handler.close()
            except Exception:
                pass
            self.workdir_file_handler = None

    # ---- Pipeline stages — each returns (data_dict, layer_specs) ----------

    def run_stage_segmentation(self, tiff_path: Path, work_dir: Path) -> tuple:
        custom_model = self.custom_model_is_selected()
        if custom_model:
            custom_model_path = Path(str(self.params_panel.custom_model_file.value).strip())
            if not custom_model_path.exists():
                raise ValueError("Custom model path does not exist.")
            model_type = None
        else:
            custom_model_path = None
            model_type = str(self.params_panel.cellpose_model.value)
        min_cell_size = int(self.params_panel.min_cell_size.value)
        use_gpu = bool(self.params_panel.use_gpu.value)

        self.set_progress(2, fmt="Loading image...")
        brightfield_stack, fluorophore_stacks, fluorophore_names, threshold_of, brightfield_channel = self.load_image_split(tiff_path)
        LOGGER.info("Loaded %s: %d frame(s), %d fluorophore(s).", tiff_path.name, brightfield_stack.shape[0], len(fluorophore_names))

        self.set_progress(5, fmt="Cellpose: starting...")
        masks_stack = cellpose_live_segmentation(
            brightfield_stack,
            diameter=CELLPOSE_DEFAULT_DIAMETER,
            flow_threshold=CELLPOSE_DEFAULT_FLOW_THRESHOLD,
            cellprob_threshold=CELLPOSE_DEFAULT_CELLPROB_THRESHOLD,
            min_size=min_cell_size,
            model_type=model_type or "cpsam",
            custom_model_path=custom_model_path,
            gpu=use_gpu,
            progress_callback=lambda current_frame, total_frames: self.set_progress(
                5 + int(90 * current_frame / total_frames), fmt=f"Cellpose: {current_frame}/{total_frames}"),
        )
        masks_path = work_dir / OUTPUT_MASKS
        tifffile.imwrite(str(masks_path), masks_stack.astype(np.uint16))
        self.set_progress(100, fmt="Segmentation saved")
        LOGGER.info("Saved %s", masks_path.name)

        layer_specs: List[dict] = [{"kind": "image", "data": brightfield_stack, "name": "raw/Brightfield",
                                    "colormap": "gray", "blending": "translucent", "opacity": 0.7}]
        colour_assignments = assign_colours(fluorophore_names)
        for name, stack in fluorophore_stacks.items():
            layer_specs.append({"kind": "image", "data": stack, "name": f"raw/{name}",
                                "colormap": colour_assignments[name]["napari"], "blending": "additive"})
        layer_specs.append({"kind": "labels_coloured", "data": masks_stack.astype(np.uint32),
                            "name": "cells/Cellpose masks", "base_colour": "slategray", "opacity": 0.15, "visible": False})
        return {"brightfield": brightfield_stack, "fluorophore_stacks": fluorophore_stacks,
                "fluorophore_names": fluorophore_names, "masks_stack": masks_stack}, layer_specs

    def run_stage_tracking(self, tiff_path: Path, work_dir: Path) -> tuple:
        masks_path = work_dir / OUTPUT_MASKS
        if not masks_path.exists():
            raise FileNotFoundError(f"Segmentation output missing: {masks_path}")
        if not bool(self.params_panel.allow_splitting.value):
            LOGGER.warning("TrackMate splitting is disabled — mother/daughter lineage will not be inferred.")

        self.set_progress(0, maximum=0, fmt="TrackMate running...")
        trackmate_output = generate_trackmate_labels(
            masks_path=masks_path, output_directory=work_dir,
            initial_search_radius=float(self.params_panel.initial_search_radius.value),
            search_radius=float(self.params_panel.search_radius.value),
            max_frame_gap=int(self.params_panel.max_frame_gap.value),
            allow_track_splitting=bool(self.params_panel.allow_splitting.value),
            splitting_max_distance=float(self.params_panel.splitting_max_distance.value),
            allow_track_merging=bool(self.params_panel.allow_merging.value),
            imagej_instance=self.imagej_instance,
        )
        self.imagej_instance = trackmate_output["imagej_instance"]
        tracks_dataframe = trackmate_output["trackmate_tracks_df"]
        linked_labels = trackmate_output["linked_labels"]

        if len(tracks_dataframe) > 0:
            tracks_dataframe = tracks_dataframe.sort_values(["track_id", "t"]).copy()
            tracks_dataframe.insert(0, "cell_id", tracks_dataframe["track_id"].astype(int) + 1)
            tracks_dataframe.insert(2, "frame", tracks_dataframe["t"].astype(int))
            tracks_dataframe.to_csv(work_dir / OUTPUT_TRACKS, index=False)

        num_tracks = int(tracks_dataframe["track_id"].nunique()) if len(tracks_dataframe) else 0
        num_division_events = int(tracks_dataframe.drop_duplicates("track_id")["parent_track_id"].notna().sum()) if "parent_track_id" in tracks_dataframe.columns else 0
        self.set_progress(100, fmt="Tracking saved")
        LOGGER.info("Tracked %d cell(s), %d point(s), %d division event(s).", num_tracks, len(tracks_dataframe), num_division_events)

        layer_specs: List[dict] = [{"kind": "labels_coloured", "data": linked_labels, "name": "cells/Linked labels",
                                    "base_colour": "gold", "opacity": 0.20}]
        if len(tracks_dataframe) > 0:
            tracks_array = tracks_dataframe[["track_id", "t", "y", "x"]].sort_values(["track_id", "t"]).to_numpy(dtype=float)
            layer_specs.append({"kind": "tracks", "data": tracks_array, "name": "cells/Tracks",
                                "tail_length": 50, "opacity": 0.8})
        return {"tracks_df": tracks_dataframe, "linked_labels": linked_labels}, layer_specs

    def run_stage_analysis(self, tiff_path: Path, work_dir: Path) -> tuple:
        """Run BOTH persistent and snapshot fate analyses. Returns (summary_record, layer_specs)."""
        linked_labels_path = work_dir / OUTPUT_LINKED
        if not linked_labels_path.exists():
            raise FileNotFoundError(f"Tracking output missing: {linked_labels_path}")

        brightfield_stack, fluorophore_stacks, fluorophore_names, threshold_of, brightfield_channel = self.load_image_split(tiff_path)
        channel_of = {name: int(getattr(self.inputs_panel, f"fluor_{i + 1}_channel").value) for i, name in enumerate(fluorophore_names)}
        linked_labels = tifffile.imread(str(linked_labels_path)).astype(np.uint32)
        num_frames = linked_labels.shape[0]
        file_stem = tiff_path.stem

        run_config = self.gather_run_config(tiff_path, work_dir, fluorophore_names, channel_of, threshold_of, brightfield_channel)
        (work_dir / OUTPUT_RUN_CONFIG).write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

        tracks_csv_path = work_dir / OUTPUT_TRACKS
        if tracks_csv_path.exists():
            tracks_dataframe = pd.read_csv(tracks_csv_path)
        else:
            tracks_dataframe = pd.DataFrame(columns=["track_id", "t", "y", "x", "quality"])
        lineage_columns = ["track_id", "lineage_id", "parent_track_id", "generation"]
        if "lineage_id" in tracks_dataframe.columns:
            lineage_lookup = tracks_dataframe.drop_duplicates("track_id")[lineage_columns].copy()
        else:
            lineage_lookup = pd.DataFrame(columns=lineage_columns)

        self.set_progress(10, fmt="Thresholding fluorescence...")
        blur_sigma = float(self.params_panel.blur_sigma.value)
        fluor_segmentation = segment_fluorescence(fluorophore_stacks, blur_sigma=blur_sigma, threshold_method=threshold_of)
        positive_label_stacks = {name: fluor_segmentation[name]["positive_labels"] for name in fluorophore_names}
        self.set_progress(30, fmt="Assigning blobs to cells...")
        frame_cell_positive_area, positive_cell_labels = compute_cell_positivity(linked_labels, positive_label_stacks, fluorophore_names)
        for name, snap_stack in positive_cell_labels.items():
            unique_positive_ids = np.unique(snap_stack)
            unique_positive_ids = unique_positive_ids[unique_positive_ids != 0]
            LOGGER.info("Snapshot positive cells [%s]: %d unique cell IDs across stack", name, len(unique_positive_ids))

        # ---- Persistent mode ----
        self.set_progress(50, fmt="Persistent fate assignment...")
        persistent_fates_df, locked_labels, _persistent_per_frame = assign_persistent_fates(linked_labels, frame_cell_positive_area)
        persistent_fates_df = persistent_fates_df.sort_values("label_id").reset_index(drop=True)
        persistent_summary_df = compute_persistent_percentages(persistent_fates_df, num_frames, fluorophore_names)
        persistent_figure, persistent_axes = plot_persistent_percentages(persistent_summary_df, fluorophore_names, title=file_stem)
        persistent_figure.savefig(str(work_dir / OUTPUT_PCT_PERSISTENT_PDF), bbox_inches="tight")
        plt.close(persistent_figure)
        LOGGER.info("Persistent fates: %s", dict(persistent_fates_df["fate"].value_counts()))

        # ---- Snapshot mode ----
        self.set_progress(75, fmt="Snapshot fate assignment...")
        snapshot_df = assign_snapshot_fates(linked_labels, frame_cell_positive_area).sort_values(["label_id", "frame"]).reset_index(drop=True)
        snapshot_summary_df, snapshot_categories = compute_snapshot_percentages(snapshot_df, num_frames)
        snapshot_figure, snapshot_axes = plot_snapshot_percentages(snapshot_summary_df, snapshot_categories, title=file_stem)
        snapshot_figure.savefig(str(work_dir / OUTPUT_PCT_SNAPSHOT_PDF), bbox_inches="tight")
        plt.close(snapshot_figure)
        trajectory_figure, trajectory_axes = plot_snapshot_trajectories(tracks_dataframe, snapshot_df, title=f"{file_stem} — snapshot trajectories")
        trajectory_figure.savefig(str(work_dir / OUTPUT_SNAPSHOT_TRAJECTORIES), bbox_inches="tight")
        plt.close(trajectory_figure)
        timeline_figure, timeline_axes = plot_snapshot_cell_timelines(snapshot_df, tracks_dataframe=tracks_dataframe, title=f"{file_stem} — cell timelines")
        timeline_figure.savefig(str(work_dir / OUTPUT_SNAPSHOT_TIMELINES), bbox_inches="tight")
        plt.close(timeline_figure)
        LOGGER.info("Snapshot categories: %s", sorted(snapshot_df["category"].unique()))

        # ---- Consolidated per-(frame, cell) CSV ----
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
            per_frame_cells_df[f"Snapshot {name}?"] = np.where(
                per_frame_cells_df[f"Thresholded {name} Area (Pixels)"] > 0, "Y", "N"
            )
        # Merge lineage info (track_id, lineage_id, parent_track_id, generation)
        per_frame_cells_df["track_id"] = per_frame_cells_df["cell_id"].astype(int) - 1
        if len(lineage_lookup) > 0:
            per_frame_cells_df = per_frame_cells_df.merge(lineage_lookup, on="track_id", how="left")
        else:
            for lineage_column in ("lineage_id", "parent_track_id", "generation"):
                per_frame_cells_df[lineage_column] = pd.NA
        # Final rename to user-facing column headers
        rename_map: Dict[str, str] = {
            "frame": "Frame ID",
            "cell_id": "Cell ID",
            "area_px": "Cell Area (pixels)",
            "track_id": "Track ID",
            "lineage_id": "Lineage ID",
            "parent_track_id": "Parent Track ID",
            "generation": "Generation",
        }
        for name in fluorophore_names:
            rename_map[f"{name}_total_intensity"] = f"{name} Fluorescence (Sum)"
        per_frame_cells_df = per_frame_cells_df.rename(columns=rename_map)
        column_order = [
            "Frame ID", "Cell ID", "Track ID", "Lineage ID", "Parent Track ID", "Generation",
            "Cell Area (pixels)",
        ]
        column_order += [f"{name} Fluorescence (Sum)" for name in fluorophore_names]
        column_order += [f"Thresholded {name} Area (Pixels)" for name in fluorophore_names]
        column_order += [f"Persistently {name}?" for name in fluorophore_names]
        column_order += [f"Snapshot {name}?" for name in fluorophore_names]
        per_frame_cells_df = per_frame_cells_df[column_order]
        per_frame_cells_df.to_csv(work_dir / OUTPUT_PER_FRAME_CELLS, index=False)
        LOGGER.info("Saved %s (%d rows).", OUTPUT_PER_FRAME_CELLS, len(per_frame_cells_df))

        # ---- Summary record ----
        summary_record: Dict[str, Any] = {
            "filename": tiff_path.name, "n_frames": num_frames,
            "n_tracked_cells": int(tracks_dataframe["track_id"].nunique()) if len(tracks_dataframe) > 0 else 0,
            "n_negative_persistent": int((persistent_fates_df["fate"] == "negative").sum()),
            "final_total_pct_persistent": float(persistent_summary_df["total_positive_pct"].iloc[-1]),
        }
        for name in fluorophore_names:
            summary_record[f"persistent_n_{name}"] = int((persistent_fates_df["fate"] == name).sum())
            summary_record[f"persistent_final_pct_{name}"] = float(persistent_summary_df[f"{name}_pct"].iloc[-1])
        last_frame_index = num_frames - 1
        last_frame_snapshot = snapshot_df[snapshot_df["frame"] == last_frame_index]
        last_frame_total = max(len(last_frame_snapshot), 1)
        for category in sorted(snapshot_df["category"].unique()):
            summary_record[f"snapshot_final_pct_{category}"] = 100.0 * (last_frame_snapshot["category"] == category).sum() / last_frame_total

        # ---- Viewer layers ----
        layer_specs: List[dict] = []
        if locked_labels is not None:
            for name, locked_label_image in locked_labels.items():
                layer_specs.append({"kind": "labels_coloured", "data": locked_label_image, "name": f"Persistent: {name} positive",
                                    "base_colour": get_fluor_base_colour(name), "opacity": 0.6, "visible": True})
        for name, positive_label_image in positive_cell_labels.items():
            layer_specs.append({"kind": "labels_coloured", "data": positive_label_image, "name": f"Snapshot: {name} positive",
                                "base_colour": get_fluor_base_colour(name), "opacity": 0.6, "visible": True})
        self.set_progress(100, fmt="Analysis saved")
        return summary_record, layer_specs

    # ---- Helpers ----------------------------------------------------------

    @staticmethod
    def attach_lineage(dataframe: pd.DataFrame, lineage_lookup: pd.DataFrame) -> pd.DataFrame:
        """Add ``track_id`` and lineage columns to a cell-id-indexed dataframe."""
        out = dataframe.copy()
        out["track_id"] = out["cell_id"].astype(int) - 1
        front = ["cell_id", "track_id"]
        if len(lineage_lookup) > 0:
            out = out.merge(lineage_lookup, on="track_id", how="left")
            front = ["cell_id", "track_id", "lineage_id", "parent_track_id", "generation"]
        return out[front + [c for c in out.columns if c not in front]]

    def show_in_viewer(self, layer_specs: List[dict]) -> None:
        """Render a list of layer specs in napari (clears existing layers first)."""
        self.viewer.layers.clear()
        for spec in layer_specs:
            kind = spec.pop("kind")
            if kind == "image":
                self.viewer.add_image(**spec)
            elif kind == "labels":
                self.viewer.add_labels(**spec)
            elif kind == "labels_coloured":
                add_coloured_labels(self.viewer, spec.pop("data"), name=spec.pop("name"),
                                    base_colour=spec.pop("base_colour"), opacity=spec.pop("opacity", 0.5), **spec)
            elif kind == "tracks":
                self.viewer.add_tracks(**spec)
            else:
                LOGGER.warning("Unknown layer kind: %s", kind)

    # ---- Button handlers --------------------------------------------------

    @gui_action("Full pipeline")
    def on_run_all_clicked(self) -> None:
        mode, tiff_path, folder, out_dir = self.resolve_paths()
        if mode == "single":
            self.run_full_pipeline(tiff_path, out_dir, render_in_viewer=True)
            return
        tiff_files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff") and p.is_file())
        if not tiff_files:
            LOGGER.warning("No .tif/.tiff files found in %s", folder)
            return
        LOGGER.info("Batch: %d file(s) in %s", len(tiff_files), folder.name)
        batch_results: List[dict] = []
        for file_index, tiff_file_path in enumerate(tiff_files, start=1):
            LOGGER.info("--- File %d/%d: %s ---", file_index, len(tiff_files), tiff_file_path.name)
            try:
                summary_record = self.run_full_pipeline(tiff_file_path, out_dir, render_in_viewer=(file_index == len(tiff_files)))
                if summary_record is not None:
                    batch_results.append(summary_record)
            except Exception as exception:
                LOGGER.error("File %s failed: %s: %s", tiff_file_path.name, type(exception).__name__, exception)
                LOGGER.debug(traceback.format_exc())
                batch_results.append({"filename": tiff_file_path.name, "error": str(exception)})
            self.set_progress(file_index, maximum=len(tiff_files), fmt=f"Files: {file_index}/{len(tiff_files)}")
            self.image_cache.clear()
        self.results.extend(batch_results)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / OUTPUT_BATCH_SUMMARY
        pd.DataFrame(self.results).to_csv(summary_path, index=False)
        LOGGER.info("Batch complete. Summary -> %s", summary_path)

    def run_full_pipeline(self, tiff_path: Path, out_dir: Path, *, render_in_viewer: bool) -> Optional[dict]:
        work_dir = self.begin_workdir(tiff_path, out_dir)
        try:
            seg_data_ignored, seg_specs = self.run_stage_segmentation(tiff_path, work_dir)
            track_data_ignored, track_specs = self.run_stage_tracking(tiff_path, work_dir)
            summary_record, analysis_specs = self.run_stage_analysis(tiff_path, work_dir)
            if render_in_viewer:
                self.show_in_viewer(seg_specs + track_specs + analysis_specs)
            LOGGER.info("Pipeline complete: %s", tiff_path.name)
            return summary_record
        finally:
            self.end_workdir()


def launch() -> FluoroFateApp:
    """Create the GUI and return the app instance (for notebook use)."""
    return FluoroFateApp()


def main() -> None:
    """Launch the standalone GUI."""
    launch()
    napari.run()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
