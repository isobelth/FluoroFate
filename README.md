# FluoroFate

**A graphical user interface for time-resolved single-cell analysis of fluorescent reporter dynamics in multi-channel timelapse microscopy data.**

FluoroFate is a point-and-click desktop application that takes multi-channel timelapse TIFF images and performs automated cell segmentation, tracking, fluorescence thresholding, and temporal fate classification — all without requiring the user to write any code. The interface is built on [napari](https://napari.org/), providing interactive visualisation of every stage of the analysis so that results can be inspected and verified before export.

The tool was developed to enable researchers without programming experience to perform rigorous, reproducible single-cell fluorescence quantification. It integrates established image analysis tools — [Cellpose](https://www.cellpose.org/) for segmentation and [TrackMate](https://imagej.net/plugins/trackmate/) for tracking — into a unified workflow with a consistent graphical interface, removing the need to move data between different software packages or write custom analysis scripts.

FluoroFate is generalisable to any combination of fluorescence channels and cell fate question. While the work accompanying this tool used Annexin V and propidium iodide (PI) to classify apoptotic and necroptotic cell death, the tool is equally applicable to FUCCI cell-cycle reporter analysis or any experiment in which fluorescent signals must be assigned to individual tracked cells over time.

---

## Overview of the Analysis Pipeline

The pipeline integrates brightfield-based cell segmentation, cell tracking, fluorescence thresholding, and temporal fate classification into a single analysis framework. Each stage can be run independently or as a complete end-to-end analysis.

![Pipeline overview — segment, track, and identify positive cells](images/image1.png)

### Stage 1: Cell Segmentation

Brightfield images are segmented frame-by-frame using Cellpose. Cell diameter is estimated automatically from the first frame, and a lower-bound cell area can be set to exclude debris or small artefacts. Either a pretrained Cellpose model or a user-supplied custom model can be used. GPU acceleration is supported when available. Segmentation outputs are saved as labelled mask stacks for downstream tracking.

### Stage 2: Cell Tracking

Segmented objects are linked across time using TrackMate, which is run headlessly through PyImageJ — there is no need to open FIJI or interact with Java directly. Cell tracks are linked using the Cellpose-identified cell masks and the Advanced Kalman Tracker. Configurable tracking parameters include the initial linking radius, Kalman search radius, and maximum frame gap (the number of frames a cell can disappear before its track is closed). Optional track splitting can be enabled to support lineage-aware analyses, in which mother-daughter relationships are inferred from TrackMate's edge structure and converted into hierarchical lineage identifiers. A relabelled stack with consistent cell identities across time is generated from the segmentation masks and TrackMate output.

### Stage 3: Fluorescence Thresholding and Fate Assignment

Fluorescence analysis can be performed on up to three user-defined channels. Each fluorescence channel is Gaussian-blurred and thresholded independently using one of several selectable global thresholding methods. Binary positive masks are generated for each frame and channel, and each fluorescence-positive connected component is assigned to whichever tracked cell label covers the most of its pixels (majority-overlap voting). The number of positive pixels assigned to each cell at each frame is recorded alongside the cell's area and summed raw fluorescence.

Each tracked cell is then classified using two complementary analysis modes (both always run; see [Analysis Modes](#analysis-modes) below), and the results are exported as a single consolidated CSV file and publication-ready PDF plots.

![Quantify positive cells and generate percentage curves](images/image2.png)

---

## Installation (No Coding Required)

> **For experienced Python users:** create the conda environment from `fluorofate_environment.yml`, activate it, and run `fluorofate.ipynb` or `python fluorofate.py` in your preferred editor. The detailed instructions below are for users who have not done this before.

FluoroFate has been designed so that users with no prior coding experience can install and run it by following the steps below. All dependencies are handled automatically by the provided environment file — there is nothing to configure manually.

### 1. Install Miniconda

Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html). This is a lightweight package manager that will handle all of FluoroFate's software dependencies. Run the installer and accept the default settings. Once installed, you should be able to open **Anaconda Prompt** from the Start menu (Windows) or run `conda` in any terminal (Mac/Linux).

### 2. Install Visual Studio Code

Download and install [Visual Studio Code](https://code.visualstudio.com/) (VS Code). This is a free code editor that can run Jupyter notebooks — which is how FluoroFate is launched. Once VS Code is open:

1. Open the **Extensions** panel (the square icon on the left sidebar, or press `Ctrl+Shift+X`).
2. Search for and install the **Python** extension (published by Microsoft).
3. Search for and install the **Jupyter** extension (published by Microsoft).

### 3. Create the FluoroFate Environment

Open a terminal inside VS Code (**Terminal → New Terminal**, or press `` Ctrl+` ``). Navigate to the folder containing this README and run:

```bash
cd path/to/FluoroFate
conda env create -f fluorofate_environment.yml
```

Replace `path/to/FluoroFate` with the actual path on your computer. This command installs Python, napari, Cellpose, PyImageJ, Java, and all required scientific libraries into a self-contained environment called `fluorofate`. This step only needs to be performed once.

### 4. Select the Environment in VS Code

1. Open the file `fluorofate.ipynb` in VS Code (File → Open File, or drag the file into the VS Code window).
2. In the top-right corner of the notebook, click the **kernel picker** (it may display "Select Kernel" or a Python version).
3. Select **Python Environments → fluorofate**. If it does not appear immediately, click "Refresh" or restart VS Code.

### 5. Launch FluoroFate

With `fluorofate.ipynb` open and the `fluorofate` kernel selected, click the **Run All** button at the top of the notebook (the double-play ▶▶ icon). The FluoroFate graphical interface will open in a napari window.

### Input Data Format

FluoroFate accepts **4-D TIFF** files with shape `(T, C, Y, X)` — time, channels, height, width. At least one brightfield channel and one fluorescence channel are required. Supported file extensions: `.tif`, `.tiff`.

---

## Training a Custom Cellpose Model (Optional)

Depending on the cell type, one of the [standard Cellpose models](https://cellpose.readthedocs.io/en/latest/models.html) may perform well without modification. However, if the default models do not segment your cells accurately, training a custom model is straightforward and typically produces a substantial improvement.

![Training a custom Cellpose model from a subset of brightfield images](images/image.png)

To train a custom model:

1. Select a small subset of representative brightfield images from your dataset.
2. Open them in the [Cellpose GUI](https://www.cellpose.org/) and iteratively train a custom model. A [video tutorial](https://www.youtube.com/watch?v=5qANHWoubZU) is available that walks through this process.
3. Save the trained model file (`.pt` or `.pth` format).

An example custom model is included in the `Example_Custom_Cellpose_Model` folder for reference.

To use a custom model in FluoroFate: in the **Parameters** panel, click the file picker next to **Custom model (optional)** and select your model file. The built-in model dropdown will be disabled automatically.

---

## Graphical Interface

The FluoroFate interface is organised into two configuration dock widgets (**Inputs** and **Parameters**) plus a **Run & Log** dock widget on the right side of the napari viewer. All parameters have sensible defaults and can be adjusted without writing any code.

### Inputs Panel

| Parameter | Default | Description |
|---|---|---|
| Input mode | Single image | `Single image` runs on one TIFF; `Folder of images` batches every TIFF in a folder |
| Single TIFF | *(empty)* | Path to one multi-channel timelapse TIFF (used in Single image mode) |
| Batch folder | *(empty)* | Folder of TIFFs (used in Folder mode) |
| Output directory | *(empty)* | **Required.** Where results are saved (one subfolder per TIFF, named after the file) |
| Brightfield channel | last channel | Channel index for brightfield. Defaults to the last channel the first time a TIFF is loaded; subsequent loads preserve your selection |
| Fluorophore 1 name | Green | Label used in output columns and filenames |
| Fluorophore 1 channel | 0 | Channel index for fluorophore 1 |
| Fluorophore 1 threshold | otsu | Thresholding method for fluorophore 1 (see [Thresholding Methods](#thresholding-methods)) |
| Fluorophore 2 name | *(blank)* | Leave blank to analyse only one fluorescence channel |
| Fluorophore 2 channel | 0 | Channel index for fluorophore 2 |
| Fluorophore 2 threshold | otsu | Thresholding method for fluorophore 2 |
| Fluorophore 3 name | *(blank)* | Leave blank to analyse fewer than three fluorescence channels |
| Fluorophore 3 channel | 0 | Channel index for fluorophore 3 |
| Fluorophore 3 threshold | otsu | Thresholding method for fluorophore 3 |

> Channel-index dropdowns are auto-populated from the selected TIFF (or the first TIFF in the selected batch folder). Between one and three fluorescence channels are supported — leave the fluorophore 2 and/or 3 name fields blank to skip them.

### Parameters Panel

| Parameter | Default | Description |
|---|---|---|
| Cellpose model | cpsam | Built-in Cellpose model (disabled when a custom model file is selected) |
| Custom model (optional) | *(empty)* | Optional `.pt`/`.pth` Cellpose model; overrides the built-in model |
| Min cell size (px) | 15 | Minimum object area in pixels |
| Use GPU | True | Enable GPU acceleration for Cellpose |
| TM init search radius | 30.0 | TrackMate initial linking distance (pixels) |
| TM search radius | 150.0 | TrackMate Kalman search radius (pixels) |
| TM max frame gap | 2 | Maximum frames a cell can be absent before the track is terminated |
| TM allow splitting | True | Enable detection of cell division events (required for lineage inference) |

### Run & Log Panel

| Button | Function |
|---|---|
| **Run All** | Single image: full pipeline (segmentation → tracking → analysis). Folder: batch every TIFF and write `batch_summary.csv` |
| **Threshold & Analyse Again** | Re-runs only the thresholding + fate-assignment stage on the last successfully processed image, using the current fluorophore threshold-method selections. Greyed out until **Run All** has completed at least once. Lets you iterate on threshold choices without re-running Cellpose / TrackMate. |
| **Clear log** | Empties the in-GUI log panel (the `run.log` file on disk is preserved) |

The progress bar and the live log are below the button. Each run also appends to `run.log` in the per-image output folder.

---

## Analysis Modes

FluoroFate **always runs both** of the following analysis modes when the analysis stage executes. They answer different biological questions, and producing both lets you compare them without re-running the pipeline.

### Persistent Mode

In persistent mode, each tracked cell is assigned a permanent fate according to the first fluorophore signal detected during the timelapse. Cells are defined by the first colour that they become positive for, regardless of fluorescence intensities at later time points. For each fluorophore, the first positive frame and total positive area across the track are recorded.

This mode is appropriate when the order of fluorescence appearance determines the biological outcome. For example, in cell-death assays:
- A cell that becomes Annexin V-positive first → **apoptosis**
- A cell that becomes PI-positive first (without prior Annexin V positivity) → **necroptosis**

Persistent mode generates cumulative percentage curves showing the proportion of cells assigned to each fate over time. These curves can only increase, since fate assignments are irreversible.

### Snapshot Mode

In snapshot mode, cells are classified independently at each frame according to the set of fluorophores active at that time point. Categories are formed by combining the names of currently active fluorophores (e.g. `Green`, `Red`, `Green+Red`, `negative`).

This mode is appropriate for analyses in which cell states change dynamically over time, such as FUCCI-based cell-cycle reporter datasets where cells transition between G1 (red) and S/G2/M (green) phases. The percentage curves in snapshot mode can both increase and decrease as cells move between states.

---

## Thresholding Methods

Each fluorophore channel can use a different global thresholding method, selected in the Inputs panel. A light Gaussian blur is applied before thresholding to reduce noise.

| Method | Description |
|---|---|
| **mean** | Pixels above the image mean intensity are classified as positive |
| **otsu** | Otsu's method; minimises intra-class variance (suitable for bimodal intensity distributions) |
| **yen** | Yen's method; maximises correlation between original and thresholded images |
| **triangle** | Triangle algorithm; suitable for unimodal histograms with an extended tail |
| **minimum** | Histogram-based minimum method; assumes a bimodal distribution |

The thresholding method can be changed in the Inputs panel and the pipeline re-run.

---

## Output Files

All per-image outputs are saved in a per-image subfolder (named after the TIFF stem) under the output directory.

### Segmentation and Tracking Outputs

| File | Description |
|---|---|
| `masks_stack.tiff` | Cellpose segmentation masks (T, Y, X), uint16 |
| `linked_labels_trackmate.tiff` | Tracked label stack with consistent cell IDs across frames |
| `trackmate_tracks.csv` | TrackMate spot-level output (intermediate; required for staged Tracking → Analysis runs) |

### Per-Cell, Per-Frame Output

| File | Description |
|---|---|
| `per_frame_cells.csv` | **The single analysis output.** One row per (frame, cell), consolidating area, raw fluorescence, thresholded positive area, persistent + snapshot fate flags, and lineage info |

Columns (in order):

- `Frame ID`, `Cell ID`, `Track ID`, `Lineage ID`, `Parent Track ID`, `Generation`
- `Cell Area (pixels)`
- `{Fluorophore} Fluorescence (Sum)` — summed raw intensity inside the cell mask (no thresholding)
- `Thresholded {Fluorophore} Area (Pixels)` — pixels of thresholded positive blobs assigned to this cell at this frame
- `{Fluorophore} Threshold Method` — the thresholding method used for this fluorophore in this run (constant per file; e.g. `otsu`, `mean`)
- `Persistently {Fluorophore}?` — `Y`/`N`, constant per cell (the cell's assigned persistent fate). A cell can be persistently `Y` for at most one fluorophore.
- `Snapshot {Fluorophore}?` — `Y`/`N`, per frame; `Y` iff `Thresholded {Fluorophore} Area (Pixels) > 0`

### Plot Outputs

For each plot type below, FluoroFate writes one "all cells" PDF plus three frame-presence-filtered variants restricted to cells appearing in at least 40%, 60% and 80% of frames (suffixes `_min40pct.pdf`, `_min60pct.pdf`, `_min80pct.pdf`). The filtered variants exclude cells that are tracked for only a small fraction of the timelapse, which are often segmentation artefacts or cells entering/leaving the field of view.

| File (all cells) | Filtered variants | Description |
|---|---|---|
| `percentages_persistent.pdf` | `percentages_persistent_min{40,60,80}pct.pdf` | Cumulative percentage of persistently-positive cells per frame |
| `percentages_snapshot.pdf` | `percentages_snapshot_min{40,60,80}pct.pdf` | Per-frame snapshot-category percentages |
| `snapshot_trajectories.pdf` | `snapshot_trajectories_min{40,60,80}pct.pdf` | XY trajectory plots coloured by snapshot category |
| `snapshot_timelines.pdf` | `snapshot_timelines_min{40,60,80}pct.pdf` | Per-cell horizontal-bar timeline coloured by category |

### Run-Level Outputs

| File | Description |
|---|---|
| `run_config.json` | Every parameter used for this run (Cellpose, TrackMate, fluorophores, thresholds, brightfield), package versions, platform, and git commit |
| `run.log` | Full run log (also streamed live in the Run & Log dock widget) |

### Batch-Level Output

In folder mode, a single `batch_summary.csv` is written at the top of the output directory. It contains one row per processed image with both persistent (per-fluorophore final percentages, fate counts) and snapshot (per-category final percentages) summary columns.

### Notes on Output Data

- **`per_frame_cells.csv` contains every tracked cell** regardless of track length. To restrict downstream analyses to longer-lived cells, filter on the `Cell ID` / `Frame ID` columns yourself; the `_min{40,60,80}pct.pdf` plots already provide pre-filtered visual summaries.
- **Thresholded area columns** report the number of thresholded positive pixels overlapping each cell mask — per fluorophore per frame.
- **Lineage columns** (`Lineage ID`, `Parent Track ID`, `Generation`) are populated when TrackMate splitting is enabled (the default). Lineage IDs use dotted notation (e.g. `1`, `1.1`, `1.2`) to group mother and daughter cells.

---

## Troubleshooting

- **GPU acceleration** is recommended for Cellpose segmentation. If a CUDA-capable GPU is not available, segmentation will still run on the CPU.
- **Java not found:** The environment file includes Java dependencies. If TrackMate reports a missing JVM, ensure that the `fluorofate` conda environment is active.
- **Re-running analysis with different thresholds:** Use the **Threshold & Analyse Again** button to re-run only the thresholding and fate-assignment stage on the most recently processed image, reusing the cached Cellpose masks and TrackMate links.
- **Batch processing** applies the same settings to every TIFF in a folder and produces a combined `batch_summary.csv` alongside per-image outputs.
- **Colour assignment** is automatic: fluorophore names containing colour keywords (e.g. "Red", "Green") are matched to corresponding plot colours. Names without a recognised colour keyword are assigned unused colours from a default palette.
