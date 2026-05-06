"""Terrain analysis functions."""

from pathlib import Path

import numpy as np
import whitebox_workflows as wbw

VALID_SLOPE_UNITS: frozenset[str] = frozenset({"degrees", "radians", "percent"})

_wbe = wbw.WbEnvironment()


def slope(
    dem_path: Path | str,
    output_path: Path | str,
    *,
    units: str = "degrees",
    z_factor: float = 1.0,
) -> Path:
    """Compute slope from a DEM using whitebox-workflows.

    Args:
        dem_path: Path to the input DEM GeoTIFF.
        output_path: Path for the output slope raster.
        units: Output units — "degrees", "radians", or "percent". Defaults to
            "degrees".
        z_factor: Vertical exaggeration factor applied before slope calculation.
            Use values > 1 to amplify relief in flat terrain. Defaults to 1.0.

    Returns:
        The resolved output_path as a Path.

    Raises:
        FileNotFoundError: If dem_path does not exist.
        ValueError: If units is not one of "degrees", "radians", "percent".
    """
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")
    if units not in VALID_SLOPE_UNITS:
        raise ValueError(
            f"invalid units {units!r}; valid values: {sorted(VALID_SLOPE_UNITS)}"
        )

    dem = _wbe.read_raster(str(dem_path))
    result = _wbe.slope(dem, units=units, z_factor=z_factor)

    out_path = Path(output_path)
    _wbe.write_raster(result, str(out_path), compress=True)

    return out_path


def aspect(
    dem_path: Path | str,
    output_path: Path | str,
    *,
    z_factor: float = 1.0,
) -> Path:
    """Compute aspect from a DEM using whitebox-workflows.

    Args:
        dem_path: Path to the input DEM GeoTIFF.
        output_path: Path for the output aspect raster (degrees, 0–360, clockwise
            from north; flat areas are set to -1).
        z_factor: Vertical exaggeration factor applied before the calculation.
            Defaults to 1.0.

    Returns:
        The resolved output_path as a Path.

    Raises:
        FileNotFoundError: If dem_path does not exist.
    """
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    dem = _wbe.read_raster(str(dem_path))
    result = _wbe.aspect(dem, z_factor=z_factor)

    out_path = Path(output_path)
    _wbe.write_raster(result, str(out_path), compress=True)

    return out_path


def hillshade(
    elevation: np.ndarray,
    *,
    azimuth: float = 165.0,
    altitude: float = 55.0,
    z_factor: float = 1.0,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """Compute hillshade from a 2-D elevation array (e.g. from eo_art).

    Args:
        elevation: 2-D float array of elevation values.
        azimuth: Sun azimuth in degrees (0 = north, clockwise). Defaults to 165.
        altitude: Sun altitude above horizon in degrees. Defaults to 55.
        z_factor: Vertical exaggeration applied before shading. Defaults to 1.0.
        dx: Cell size in x direction (same units as elevation). Defaults to 1.0.
        dy: Cell size in y direction (same units as elevation). Defaults to 1.0.

    Returns:
        2-D float32 array in [0, 1] where 1 is fully lit and 0 is fully shadowed.
    """
    from matplotlib.colors import LightSource

    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    result = ls.hillshade(elevation, vert_exag=z_factor, dx=dx, dy=dy)
    return result.astype(np.float32)


def hillshade_path(
    dem_path: Path | str,
    output_path: Path | str,
    *,
    azimuth: float = 165.0,
    altitude: float = 55.0,
    z_factor: float = 1.0,
) -> Path:
    """Compute hillshade from a DEM file using whitebox-workflows.

    Args:
        dem_path: Path to the input DEM GeoTIFF.
        output_path: Path for the output hillshade raster.
        azimuth: Sun azimuth in degrees (0 = north, clockwise). Defaults to 165.
        altitude: Sun altitude above horizon in degrees. Defaults to 55.
        z_factor: Vertical exaggeration applied before shading. Defaults to 1.0.

    Returns:
        The resolved output_path as a Path.

    Raises:
        FileNotFoundError: If dem_path does not exist.
    """
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    dem = _wbe.read_raster(str(dem_path))
    result = _wbe.hillshade(dem, azimuth=azimuth, altitude=altitude, z_factor=z_factor)

    out_path = Path(output_path)
    _wbe.write_raster(result, str(out_path), compress=True)

    return out_path
