"""Colour inference and category colour-mapping for plots and Napari."""

import colorsys

import matplotlib.colors as mcolors
import numpy as np

COLOUR_KEYWORDS = {"red": {"mpl": "tab:red", "napari": "red", "rgb": (1.0, 0.0, 0.0)}, "green": {"mpl": "tab:green", "napari": "green", "rgb": (0.0, 0.8, 0.0)}, "blue": {"mpl": "tab:blue", "napari": "blue", "rgb": (0.2, 0.4, 1.0)}, "yellow": {"mpl": "goldenrod", "napari": "yellow", "rgb": (0.9, 0.85, 0.0)}, "cyan": {"mpl": "tab:cyan", "napari": "cyan", "rgb": (0.0, 0.8, 0.8)}, "magenta": {"mpl": "tab:pink", "napari": "magenta", "rgb": (0.9, 0.0, 0.6)}}


def assign_colours(names):
    """Assign plot colours to a list of fluorophore or category names.

    Names containing a colour keyword (e.g. ``"Red"``, ``"Green"``) are
    matched to that colour. Unmatched names cycle through the remaining
    palette entries; if more names are provided than palette entries the
    palette wraps around (no error is raised).

    Parameters
    ----------
    names : list[str]
        Fluorophore or category names.

    Returns
    -------
    dict[str, dict]
        ``{name: {"mpl": ..., "napari": ..., "rgb": ...}}`` for each
        input name.
    """
    palette_entries = list(COLOUR_KEYWORDS.values())
    palette_keywords = list(COLOUR_KEYWORDS.keys())
    result = {}
    used_palette_indices = set()
    for name in names:
        lowercase_name = name.lower()
        for palette_index, keyword in enumerate(palette_keywords):
            if keyword in lowercase_name:
                result[name] = palette_entries[palette_index]
                used_palette_indices.add(palette_index)
                break
    available_indices = [palette_index for palette_index in range(len(palette_entries)) if palette_index not in used_palette_indices]
    fallback_counter = 0
    for name in names:
        if name not in result:
            if available_indices:
                result[name] = palette_entries[available_indices[fallback_counter % len(available_indices)]]
            else:
                result[name] = palette_entries[fallback_counter % len(palette_entries)]
            fallback_counter += 1
    return result


def build_category_colormap(categories):
    """Build an RGB colour mapping for snapshot categories.

    Single-fluorophore categories (e.g. ``"Red"``) get their inferred
    colour directly. Compound categories (e.g. ``"Red+Green"``) get the
    component-wise mean of their parts' RGB values, clamped to ``[0, 1]``.
    The category ``"negative"`` is always grey.

    Parameters
    ----------
    categories : list[str]
        Category names, e.g. ``["Red", "Green", "Red+Green", "negative"]``.

    Returns
    -------
    dict[str, tuple[float, float, float]]
        ``{category_name: (R, G, B)}`` with values in 0–1 range.
    """
    single_fluorophores = []
    for category in categories:
        if category == "negative":
            continue
        for part in category.split("+"):
            if part not in single_fluorophores:
                single_fluorophores.append(part)
    colour_assignments = assign_colours(single_fluorophores)
    single_rgb = {fluorophore_name: np.array(colour_assignments[fluorophore_name]["rgb"]) for fluorophore_name in single_fluorophores}
    category_colormap = {}
    for category in categories:
        if category == "negative":
            category_colormap[category] = (0.7, 0.7, 0.7)
            continue
        parts = category.split("+")
        rgb = np.clip(np.mean([single_rgb.get(part, np.array([0.5, 0.5, 0.5])) for part in parts], axis=0), 0, 1)
        category_colormap[category] = tuple(rgb.tolist())
    return category_colormap


def get_fluor_base_colour(name):
    """Map a fluorophore name to a matplotlib base colour string.

    Parameters
    ----------
    name : str
        Fluorophore name (e.g. ``"Green"``, ``"Red"``, ``"Annexin V"``).

    Returns
    -------
    str
        A matplotlib named colour (e.g. ``"limegreen"``, ``"red"``).
        Defaults to ``"cyan"`` when no colour keyword is found.
    """
    lowercase_name = name.lower()
    if "green" in lowercase_name:
        return "limegreen"
    if "red" in lowercase_name:
        return "red"
    if "blue" in lowercase_name:
        return "dodgerblue"
    return "cyan"


def build_direct_label_colormap(label_stack, base_colour, hue_span=0.02, sat_span=0.45, light_span=0.45, seed=0, alpha=1.0):
    """Build a Napari ``DirectLabelColormap`` with jittered shades of a base colour.

    Each non-zero label ID in ``label_stack`` gets a unique colour derived
    from ``base_colour`` by adding small uniform jitter to the hue,
    saturation and lightness. Background (label 0) is always transparent.

    Parameters
    ----------
    label_stack : numpy.ndarray
        Integer label image (2-D, 3-D, or 4-D). Background must be 0.
    base_colour : str
        Any matplotlib named colour (e.g. ``"red"``, ``"limegreen"``).
    hue_span, sat_span, light_span : float
        Half-widths of the uniform jitter applied to hue / saturation /
        lightness in HLS space.
    seed : int, default 0
        RNG seed for reproducibility.
    alpha : float, default 1.0
        Alpha channel value applied to every label colour.

    Returns
    -------
    napari.utils.colormaps.DirectLabelColormap
    """
    from napari.utils.colormaps import DirectLabelColormap

    unique_label_ids = np.unique(label_stack)
    unique_label_ids = unique_label_ids[unique_label_ids != 0]
    num_labels = len(unique_label_ids)
    label_to_rgba = {0: (0.0, 0.0, 0.0, 0.0)}
    if num_labels > 0:
        random_generator = np.random.default_rng(seed)
        base_red, base_green, base_blue = mcolors.to_rgb(base_colour)
        base_hue, base_lightness, base_saturation = colorsys.rgb_to_hls(base_red, base_green, base_blue)
        hues = (base_hue + random_generator.uniform(-hue_span, hue_span, num_labels)) % 1.0
        saturations = np.clip(base_saturation + random_generator.uniform(-sat_span, sat_span, num_labels), 0.20, 1.00)
        lightnesses = np.clip(base_lightness + random_generator.uniform(-light_span, light_span, num_labels), 0.18, 0.88)
        for label_index, label_id in enumerate(unique_label_ids):
            jittered_red, jittered_green, jittered_blue = colorsys.hls_to_rgb(float(hues[label_index]), float(lightnesses[label_index]), float(saturations[label_index]))
            label_to_rgba[int(label_id)] = (jittered_red, jittered_green, jittered_blue, float(alpha))
    return DirectLabelColormap(color_dict=label_to_rgba)


def add_coloured_labels(viewer, label_stack, name, base_colour, opacity=0.5, **napari_layer_kwargs):
    """Add a Napari labels layer with jittered shades of ``base_colour``.

    Convenience wrapper around :func:`build_direct_label_colormap` that
    creates the colormap and adds the layer in one call.

    Parameters
    ----------
    viewer : napari.Viewer
    label_stack : numpy.ndarray
    name : str
    base_colour : str
    opacity : float, default 0.5
    **napari_layer_kwargs
        Forwarded to :meth:`napari.Viewer.add_labels`.

    Returns
    -------
    napari.layers.Labels
    """
    label_colormap = build_direct_label_colormap(label_stack, base_colour)
    return viewer.add_labels(label_stack.astype(np.int32), name=name, opacity=opacity, colormap=label_colormap, **napari_layer_kwargs)
