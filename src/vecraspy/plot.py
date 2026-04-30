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
    if units not in _SLOPE_UNIT_STYLES:
        raise ValueError(
            f"invalid units {units!r}; valid values: {sorted(_SLOPE_UNIT_STYLES)}"
        )

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
        raise ValueError(
            f"no valid (non-nodata) pixels in band {band} of {slope_path}"
        )

    vmin, vmax, colorbar_label = _SLOPE_UNIT_STYLES[units]
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
