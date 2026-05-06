"""Plotting utilities for raster and vector geodata."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.axes import Axes
from matplotlib.figure import Figure

DEFAULT_DSM_CMAP = "terrain"
DEFAULT_PERCENTILE_LOW = 2.0
DEFAULT_PERCENTILE_HIGH = 98.0
DEFAULT_FIGURE_SIZE = (10, 8)
DEFAULT_SLOPE_CMAP = "RdYlGn_r"
DEFAULT_ASPECT_CMAP = "hsv"
DEFAULT_HILLSHADE_CMAP = "gray"

_SLOPE_UNIT_STYLES: dict[str, tuple[float, float, str]] = {
    "degrees": (0.0, 90.0, "Slope (°)"),
    "radians": (0.0, math.pi / 2, "Slope (rad)"),
    "percent": (0.0, 100.0, "Slope (%)"),
}


def plot_slope(
    path: Path | str,
    *,
    units: str = "degrees",
    cmap: str = DEFAULT_SLOPE_CMAP,
    style: dict[str, tuple[float, float, str]] = _SLOPE_UNIT_STYLES,
    band: int = 1,
    ax: Axes | None = None,
    show_colorbar: bool = True,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a slope raster with a unit-aware fixed display range.

    The display range is fixed per unit type (degrees: 0–90, radians: 0–π/2,
    percent: 0–100) so comparisons across scenes are visually consistent.
    Nodata pixels are masked and shown as transparent.

    Args:
        path: Path to the slope GeoTIFF.
        units: Units of the slope values — "degrees", "radians", or "percent".
            Controls vmin, vmax, and the colorbar label. Defaults to "degrees".
        cmap: Matplotlib colourmap name. Defaults to "RdYlGn_r".
        band: Raster band index (1-based). Defaults to 1.
        ax: Existing Axes to draw into. Creates a new figure when None.
        show_colorbar: Attach a colourbar. Defaults to True.
        title: Axes title. Defaults to the file stem when None.

    Returns:
        Tuple of (Figure, Axes) for further customisation.

    Raises:
        ValueError: If units is not one of "degrees", "radians", "percent".
        FileNotFoundError: If the file does not exist.
        ValueError: If all pixels in the band are nodata.
    """
    if units not in style:
        raise ValueError(f"invalid units {units!r}; valid values: {sorted(style)}")

    slope_path = Path(path)
    if not slope_path.exists():
        raise FileNotFoundError(f"file not found: {slope_path}")

    with rasterio.open(slope_path) as src:
        raw = src.read(band)
        nodata = src.nodata
        bounds = src.bounds

    data = raw.astype(float)
    if nodata is not None:
        data = np.where(raw == nodata, np.nan, data)

    valid = data[np.isfinite(data)]
    if valid.size == 0:
        raise ValueError(f"no valid (non-nodata) pixels in band {band} of {slope_path}")

    vmin, vmax, colorbar_label = style[units]
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    else:
        fig = ax.get_figure()

    img = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="upper",
    )

    ax.set_title(title if title is not None else slope_path.stem)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")

    if show_colorbar:
        fig.colorbar(img, ax=ax, label=colorbar_label)

    return fig, ax


def plot_dsm(
    path: Path | str,
    *,
    percentile_low: float = DEFAULT_PERCENTILE_LOW,
    percentile_high: float = DEFAULT_PERCENTILE_HIGH,
    cmap: str = DEFAULT_DSM_CMAP,
    band: int = 1,
    ax: Axes | None = None,
    show_colorbar: bool = True,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a DSM GeoTIFF with percentile-based contrast stretching.

    Clips display range to the given percentiles so terrain detail is visible
    even when the data contains extreme outliers or a wide elevation range.
    Nodata pixels are masked and shown as transparent.

    Args:
        path: Path to the GeoTIFF file.
        percentile_low: Lower percentile for the display minimum. Pixels below
            this value are clamped to the minimum colour. Defaults to 2.
        percentile_high: Upper percentile for the display maximum. Pixels above
            this value are clamped to the maximum colour. Defaults to 98.
        cmap: Matplotlib colourmap name. Defaults to "terrain".
        band: Raster band index (1-based). Defaults to 1.
        ax: Existing Axes to draw into. Creates a new figure when None.
        show_colorbar: Attach a colourbar showing the elevation range.
        title: Axes title. Defaults to the file stem when None.

    Returns:
        Tuple of (Figure, Axes) for further customisation.

    Raises:
        FileNotFoundError: If the file does not exist.
        rasterio.errors.RasterioIOError: If the file cannot be opened.
    """
    tif_path = Path(path)
    if not tif_path.exists():
        raise FileNotFoundError(f"file not found: {tif_path}")

    with rasterio.open(tif_path) as src:
        raw = src.read(band)
        nodata = src.nodata
        bounds = src.bounds

    data = raw.astype(float)
    if nodata is not None:
        data = np.where(raw == nodata, np.nan, data)

    valid = data[np.isfinite(data)]
    if valid.size == 0:
        raise ValueError(f"no valid (non-nodata) pixels in band {band} of {tif_path}")

    vmin = float(np.percentile(valid, percentile_low))
    vmax = float(np.percentile(valid, percentile_high))

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    else:
        fig = ax.get_figure()

    img = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="upper",
    )

    ax.set_title(title if title is not None else tif_path.stem)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")

    if show_colorbar:
        fig.colorbar(img, ax=ax, label="Elevation (m)")

    return fig, ax


def plot_aspect(
    path: Path | str,
    *,
    cmap: str = DEFAULT_ASPECT_CMAP,
    band: int = 1,
    ax: Axes | None = None,
    show_colorbar: bool = True,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot an aspect raster with a circular colourmap (0–360°).

    Flat pixels (coded as -1 by whitebox-workflows) are masked and shown as
    transparent. The display range is fixed to 0–360 so north always maps to
    the same hue regardless of the scene.

    Args:
        path: Path to the aspect GeoTIFF.
        cmap: Matplotlib colourmap name. Defaults to "hsv".
        band: Raster band index (1-based). Defaults to 1.
        ax: Existing Axes to draw into. Creates a new figure when None.
        show_colorbar: Attach a colourbar. Defaults to True.
        title: Axes title. Defaults to the file stem when None.

    Returns:
        Tuple of (Figure, Axes) for further customisation.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If all pixels are nodata or flat.
    """
    aspect_path = Path(path)
    if not aspect_path.exists():
        raise FileNotFoundError(f"file not found: {aspect_path}")

    with rasterio.open(aspect_path) as src:
        raw = src.read(band)
        nodata = src.nodata
        bounds = src.bounds

    data = raw.astype(float)
    # mask nodata and flat-area sentinel (-1)
    mask = np.zeros(data.shape, dtype=bool)
    if nodata is not None:
        mask |= raw == nodata
    mask |= data < 0
    data = np.where(mask, np.nan, data)

    if not np.any(np.isfinite(data)):
        raise ValueError(f"no valid aspect pixels in band {band} of {aspect_path}")

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    else:
        fig = ax.get_figure()

    img = ax.imshow(
        data,
        cmap=cmap,
        vmin=0.0,
        vmax=360.0,
        extent=extent,
        origin="upper",
    )

    ax.set_title(title if title is not None else aspect_path.stem)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")

    if show_colorbar:
        fig.colorbar(img, ax=ax, label="Aspect (°)")

    return fig, ax


def plot_hillshade(
    source: Path | str | np.ndarray,
    *,
    cmap: str = DEFAULT_HILLSHADE_CMAP,
    band: int = 1,
    ax: Axes | None = None,
    title: str | None = None,
    extent: list[float] | None = None,
) -> tuple[Figure, Axes]:
    """Plot a hillshade from a file path or a numpy array.

    Accepts either a GeoTIFF path (output of ``hillshade_path``) or a 2-D
    float32 array (output of ``hillshade``). Values are normalised to [0, 1]
    so both sources display consistently.

    Args:
        source: Path to a hillshade GeoTIFF, or a 2-D float/uint array.
        cmap: Matplotlib colourmap name. Defaults to "gray".
        band: Raster band index (1-based); only used when source is a path.
        ax: Existing Axes to draw into. Creates a new figure when None.
        title: Axes title. Defaults to the file stem (path) or "Hillshade"
            (array).
        extent: ``[left, right, bottom, top]`` in map coordinates; only used
            when source is an array (ignored for path input where bounds come
            from the file).

    Returns:
        Tuple of (Figure, Axes) for further customisation.

    Raises:
        FileNotFoundError: If source is a path that does not exist.
        ValueError: If all pixels are nodata.
    """
    if isinstance(source, np.ndarray):
        data = source.astype(float)
        data_extent = extent
        default_title = "Hillshade"
    else:
        hs_path = Path(source)
        if not hs_path.exists():
            raise FileNotFoundError(f"file not found: {hs_path}")
        with rasterio.open(hs_path) as src:
            raw = src.read(band)
            nodata = src.nodata
            bounds = src.bounds
        data = raw.astype(float)
        if nodata is not None:
            data = np.where(raw == nodata, np.nan, data)
        data_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        default_title = hs_path.stem

    # normalise to [0, 1] — wbe outputs 0-255, LightSource outputs 0-1
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("no valid pixels in hillshade source")
    if finite.max() > 1.0:
        data = data / 255.0

    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    else:
        fig = ax.get_figure()

    ax.imshow(
        data,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=data_extent,
        origin="upper",
    )

    ax.set_title(title if title is not None else default_title)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")

    return fig, ax
