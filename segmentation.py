"""Cellpose and fluorescence segmentation functions.

Two top-level helpers:

* :func:`cellpose_live_segmentation` — segment a brightfield time-lapse
  frame-by-frame using Cellpose, with optional GUI progress callback.
* :func:`segment_fluorescence` — Gaussian blur + per-frame thresholding
  + connected-component labelling for one or more fluorescence channels.

Status messages are emitted via the standard :mod:`logging` module so
they can be routed to a GUI log widget or a log file.
"""

import logging
from pathlib import Path

import numpy as np
from cellpose import io, models
from skimage.filters import gaussian, threshold_mean, threshold_minimum, threshold_otsu, threshold_triangle, threshold_yen
from skimage.measure import label

LOGGER = logging.getLogger(__name__)

THRESHOLD_FUNCTIONS = {"mean": threshold_mean, "minimum": threshold_minimum, "yen": threshold_yen, "otsu": threshold_otsu, "triangle": threshold_triangle}


def cellpose_live_segmentation(brightfield_stack, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0, min_size=15, model_type="cpsam", custom_model_path=None, gpu=True, progress_callback=None):
    """Segment a brightfield image stack frame-by-frame with Cellpose.

    If ``diameter`` is ``None`` it is estimated from the cells detected
    in the first frame; if the first frame contains no cells the value
    falls back to 30 pixels and a warning is logged.

    Parameters
    ----------
    brightfield_stack : numpy.ndarray, str, or pathlib.Path
        Brightfield image stack of shape ``(n_frames, H, W)``, a single
        2-D image of shape ``(H, W)``, or a file path that Cellpose can
        read.
    diameter : float or None, default None
        Expected cell diameter in pixels. If ``None``, estimated
        automatically from the first frame.
    flow_threshold : float, default 0.4
        Cellpose flow-error threshold (lower = stricter cell boundaries).
    cellprob_threshold : float, default 0.0
        Cellpose cell-probability threshold (higher = fewer cells kept).
    min_size : int, default 15
        Minimum cell area in pixels; smaller objects are removed.
    model_type : str, default "cpsam"
        Built-in Cellpose model name (e.g. ``"cpsam"``, ``"cyto3"``,
        ``"nuclei"``). Ignored when ``custom_model_path`` is provided.
    custom_model_path : str, pathlib.Path or None, default None
        Path to a user-trained Cellpose model file. Overrides
        ``model_type`` when given.
    gpu : bool, default True
        Whether to use GPU acceleration.
    progress_callback : callable or None, default None
        Called as ``progress_callback(current_frame, total_frames)``
        after each frame so a GUI can update its progress bar.

    Returns
    -------
    numpy.ndarray, dtype uint16
        Segmentation masks with the same spatial shape as the input. If
        the input was a single 2-D image, a 2-D array is returned;
        otherwise a 3-D stack of shape ``(n_frames, H, W)``.
    """
    if isinstance(brightfield_stack, (str, Path)):
        brightfield_stack = io.imread(brightfield_stack)
    if brightfield_stack.ndim == 2:
        brightfield_stack = brightfield_stack[np.newaxis, :, :]

    if custom_model_path:
        LOGGER.info("Cellpose: loading custom model from %s", custom_model_path)
        cellpose_model = models.CellposeModel(gpu=gpu, pretrained_model=str(custom_model_path))
    else:
        LOGGER.info("Cellpose: loading built-in model %s", model_type)
        cellpose_model = models.CellposeModel(gpu=gpu, model_type=model_type)

    if diameter is None:
        first_frame_masks, _, _ = cellpose_model.eval(brightfield_stack[0], diameter=30, flow_threshold=flow_threshold, min_size=min_size)
        if first_frame_masks.max() > 0:
            cell_areas = [int(np.sum(first_frame_masks == cell_id)) for cell_id in range(1, min(first_frame_masks.max() + 1, 50))]
            diameter = 2 * np.sqrt((np.median(cell_areas) if cell_areas else 700) / np.pi)
            LOGGER.info("Cellpose: estimated diameter from first frame = %.1f px", diameter)
        else:
            diameter = 30.0
            LOGGER.warning("Cellpose: no cells detected in first frame; falling back to default diameter = %.1f px", diameter)

    num_frames = brightfield_stack.shape[0]
    masks_stack = np.zeros(brightfield_stack.shape, dtype=np.uint16)
    LOGGER.info("Cellpose: segmenting %d frame(s) (diameter=%.1f, flow_threshold=%.2f, cellprob_threshold=%.2f, min_size=%d)", num_frames, diameter, flow_threshold, cellprob_threshold, min_size)
    for frame_index in range(num_frames):
        frame_masks, _, _ = cellpose_model.eval(brightfield_stack[frame_index], diameter=diameter, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, min_size=min_size, resample=True)
        masks_stack[frame_index] = frame_masks.astype(np.uint16)
        LOGGER.info("Cellpose: frame %d/%d — %d cell(s) detected", frame_index + 1, num_frames, int(frame_masks.max()))
        if progress_callback is not None:
            progress_callback(frame_index + 1, num_frames)

    LOGGER.info("Cellpose: segmentation complete (%d frame(s))", num_frames)
    return masks_stack[0] if num_frames == 1 else masks_stack


def segment_fluorescence(fluorescence_stacks, blur_sigma=1.0, threshold_method="otsu"):
    """Segment fluorescence channels by Gaussian blur + per-frame thresholding.

    Each fluorophore stack is processed independently: a Gaussian blur is
    applied, then a separate threshold is computed for each frame
    (handles signal drift over time), pixels above that threshold are
    flagged as positive, and connected components of positive pixels are
    labelled.

    Parameters
    ----------
    fluorescence_stacks : dict[str, numpy.ndarray]
        ``{fluorophore_name: array of shape (n_frames, H, W)}``.
    blur_sigma : float, default 1.0
        Sigma for Gaussian smoothing before thresholding.
    threshold_method : str or dict[str, str], default "otsu"
        Thresholding algorithm. A single string applies the same method
        to every channel; a dict maps each channel name to its own
        method. Supported: ``"mean", "minimum", "yen", "otsu", "triangle"``.

    Returns
    -------
    dict
        Top-level keys ``"threshold_methods"`` (the resolved per-channel
        method map) and ``"blur_sigma"``, plus one key per fluorophore.
        Each fluorophore entry is a dict with:

        * ``"blurred"``              — ndarray (n_frames, H, W) smoothed stack
        * ``"thresholds_per_frame"`` — ndarray (n_frames,) threshold per frame
        * ``"positive"``             — ndarray (n_frames, H, W) bool binary mask
        * ``"positive_labels"``      — ndarray (n_frames, H, W) uint32 blob labels
    """
    if isinstance(threshold_method, str):
        threshold_method_per_channel = {fluorophore_name: threshold_method.lower() for fluorophore_name in fluorescence_stacks}
    else:
        threshold_method_per_channel = {fluorophore_name: str(threshold_method.get(fluorophore_name, "otsu")).lower() for fluorophore_name in fluorescence_stacks}

    result = {"threshold_methods": threshold_method_per_channel, "blur_sigma": blur_sigma}
    for fluorophore_name, fluorescence_stack in fluorescence_stacks.items():
        method_name = threshold_method_per_channel[fluorophore_name]
        if method_name not in THRESHOLD_FUNCTIONS:
            raise ValueError(f"Unsupported threshold method '{method_name}' for channel '{fluorophore_name}'. Choose one of {sorted(THRESHOLD_FUNCTIONS)}.")
        threshold_function = THRESHOLD_FUNCTIONS[method_name]
        blurred_stack = np.stack([gaussian(frame, sigma=blur_sigma, preserve_range=True) for frame in fluorescence_stack], axis=0)
        num_frames = blurred_stack.shape[0]
        thresholds_per_frame = np.zeros(num_frames)
        positive_mask_stack = np.zeros_like(blurred_stack, dtype=bool)
        positive_label_stack = np.zeros(blurred_stack.shape, dtype=np.uint32)
        for frame_index in range(num_frames):
            frame_threshold = threshold_function(blurred_stack[frame_index])
            thresholds_per_frame[frame_index] = frame_threshold
            positive_mask_stack[frame_index] = blurred_stack[frame_index] > frame_threshold
            positive_label_stack[frame_index] = label(positive_mask_stack[frame_index]).astype(np.uint32)
        result[fluorophore_name] = {"blurred": blurred_stack, "thresholds_per_frame": thresholds_per_frame, "positive": positive_mask_stack, "positive_labels": positive_label_stack}
    return result
