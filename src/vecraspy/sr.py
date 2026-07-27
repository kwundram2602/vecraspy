"""DTM super-resolution: optical-guided filtering and thermal erosion."""

import math
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from vecraspy.raster import align_raster_grid, clip_tif_by_aoi, tif_bounds_as_polygon

_VALID_RESAMPLING_NAMES: frozenset[str] = frozenset(r.name for r in Resampling)


def _box_filter(arr: np.ndarray, radius: int) -> np.ndarray:
    """Local mean of arr over a (2*radius+1) square window (edge-padded)."""
    height, width = arr.shape
    window = 2 * radius + 1
    padded = np.pad(arr, radius, mode="edge")
    cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cumsum = np.pad(cumsum, ((1, 0), (1, 0)), mode="constant")
    total = (
        cumsum[window : window + height, window : window + width]
        - cumsum[0:height, window : window + width]
        - cumsum[window : window + height, 0:width]
        + cumsum[0:height, 0:width]
    )
    return total / (window * window)


def _guided_filter(
    guide: np.ndarray, source: np.ndarray, radius: int, eps: float
) -> np.ndarray:
    mean_I = _box_filter(guide, radius)
    mean_p = _box_filter(source, radius)
    corr_I = _box_filter(guide * guide, radius)
    corr_Ip = _box_filter(guide * source, radius)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)

    return mean_a * guide + mean_b


def guided_upsample_dem(
    dem_path: Path | str,
    optical_path: Path | str,
    output_path: Path | str,
    *,
    band: int | None = None,
    radius: int = 8,
    eps: float = 1e-2,
    resampling: str = "cubic",
) -> Path:
    """Upsample a DEM to the resolution of a co-registered optical image.

    Bicubic-warps the DEM onto the optical image's grid, then refines the
    result with a guided image filter (He et al.) using the optical image
    as an edge/structure reference. Pixels near a nodata boundary in the
    DEM may be slightly smoothed by the filter before the original nodata
    mask is reapplied.

    Args:
        dem_path: Path to the source DEM GeoTIFF.
        optical_path: Path to the co-registered high-resolution optical GeoTIFF.
        output_path: Path for the super-resolved output GeoTIFF.
        band: 1-indexed optical band to use as the guide. Defaults to the
            mean of all bands when None.
        radius: Guided filter window radius, in pixels of the optical grid.
        eps: Guided filter regularization term; the guide is normalized to
            [0, 1] internally, so the default matches the literature.
        resampling: Rasterio resampling name for the bicubic baseline warp.

    Returns:
        The resolved output_path as a Path.

    Raises:
        FileNotFoundError: If dem_path or optical_path does not exist.
        ValueError: If radius <= 0, eps <= 0, band is out of range, or
            resampling is not a valid rasterio resampling algorithm.
    """
    dem_path = Path(dem_path)
    optical_path = Path(optical_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"file not found: {dem_path}")
    if not optical_path.exists():
        raise FileNotFoundError(f"file not found: {optical_path}")
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if resampling not in _VALID_RESAMPLING_NAMES:
        raise ValueError(
            f"invalid resampling {resampling!r}; "
            f"valid names: {sorted(_VALID_RESAMPLING_NAMES)}"
        )

    with rasterio.open(optical_path) as optical_src:
        band_count = optical_src.count
    if band is not None and not (1 <= band <= band_count):
        raise ValueError(f"band must be in [1, {band_count}], got {band}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        footprint = tif_bounds_as_polygon(dem_path)
        with rasterio.open(dem_path) as dem_src:
            dem_crs_wkt = dem_src.crs.to_wkt()

        optical_clip = tmp_dir_path / "optical_clip.tif"
        clip_tif_by_aoi(optical_path, optical_clip, footprint, aoi_crs=dem_crs_wkt)

        dem_upsampled = tmp_dir_path / "dem_upsampled.tif"
        align_raster_grid(optical_clip, dem_path, dem_upsampled, resampling=resampling)

        with rasterio.open(dem_upsampled) as up_src:
            baseline = up_src.read(1).astype(np.float64)
            profile = up_src.profile.copy()
            nodata = up_src.nodata
            valid_mask = up_src.read_masks(1) != 0

        with rasterio.open(optical_clip) as guide_src:
            if band is not None:
                guide_raw = guide_src.read(band).astype(np.float64)
            else:
                guide_raw = guide_src.read().astype(np.float64).mean(axis=0)

        guide_min = guide_raw.min()
        guide_max = guide_raw.max()
        if guide_max > guide_min:
            guide = (guide_raw - guide_min) / (guide_max - guide_min)
        else:
            guide = np.zeros_like(guide_raw)

        baseline_filled = baseline.copy()
        if not valid_mask.all():
            fill_value = baseline[valid_mask].mean() if valid_mask.any() else 0.0
            baseline_filled[~valid_mask] = fill_value

        result = _guided_filter(guide, baseline_filled, radius, eps)

        if nodata is not None:
            result[~valid_mask] = nodata

        profile.update(dtype="float32")
        out_path = Path(output_path)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(result.astype(np.float32), 1)

    return out_path


_NEIGHBOR_OFFSETS: list[tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return arr shifted by (dy, dx), replicating edge values (no wrap)."""
    height, width = arr.shape
    padded = np.pad(arr, 1, mode="edge")
    return padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]


def simulate_thermal_erosion(
    dem_path: Path | str,
    output_path: Path | str,
    *,
    iterations: int = 50,
    talus_angle: float = 35.0,
    transfer_rate: float = 0.5,
) -> Path:
    """Refine a DEM with a vectorized talus-angle thermal erosion simulation.

    Each iteration, every cell sheds material toward its steepest of 8
    neighbours if the slope to that neighbour exceeds talus_angle, moving a
    transfer_rate fraction of the excess toward the talus-stable level.
    Purely procedural, independent of any optical guide. Nodata cells do not
    participate in material transport.

    Args:
        dem_path: Path to the source DEM GeoTIFF.
        output_path: Path for the eroded output GeoTIFF.
        iterations: Number of simulation steps.
        talus_angle: Talus (repose) angle in degrees, in (0, 90).
        transfer_rate: Fraction (0, 1] of the excess above the talus-stable
            level moved per iteration.

    Returns:
        The resolved output_path as a Path.

    Raises:
        FileNotFoundError: If dem_path does not exist.
        ValueError: If iterations <= 0, talus_angle is outside (0, 90), or
            transfer_rate is outside (0, 1].
    """
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"file not found: {dem_path}")
    if iterations <= 0:
        raise ValueError(f"iterations must be > 0, got {iterations}")
    if not 0 < talus_angle < 90:
        raise ValueError(f"talus_angle must be in (0, 90), got {talus_angle}")
    if not 0 < transfer_rate <= 1:
        raise ValueError(f"transfer_rate must be in (0, 1], got {transfer_rate}")

    with rasterio.open(dem_path) as src:
        height_map = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        nodata = src.nodata
        valid_mask = src.read_masks(1) != 0
        pixel_size = (abs(src.transform.a) + abs(src.transform.e)) / 2

    talus_slope = math.tan(math.radians(talus_angle))
    rows, cols = np.indices(height_map.shape)

    for _ in range(iterations):
        best_diff = np.zeros_like(height_map)
        best_dy = np.zeros(height_map.shape, dtype=np.int8)
        best_dx = np.zeros(height_map.shape, dtype=np.int8)

        for dy, dx in _NEIGHBOR_OFFSETS:
            neighbor = _shift(height_map, dy, dx)
            neighbor_valid = _shift(valid_mask, dy, dx)
            distance = pixel_size * math.hypot(dy, dx)
            diff = height_map - neighbor - talus_slope * distance
            # A nodata neighbour must never be chosen as a transport target —
            # its raw sentinel value (e.g. -9999) would otherwise look like an
            # enormous drop and drain real elevation from valid cells at the
            # border, since the nodata mask is only restored once at the end.
            diff = np.where(neighbor_valid, diff, -np.inf)
            is_better = diff > best_diff
            best_diff = np.where(is_better, diff, best_diff)
            best_dy = np.where(is_better, dy, best_dy)
            best_dx = np.where(is_better, dx, best_dx)

        transportable = valid_mask & (best_diff > 0)
        amount = np.where(transportable, transfer_rate * best_diff / 2, 0.0)

        target_rows = np.clip(rows + best_dy, 0, height_map.shape[0] - 1)
        target_cols = np.clip(cols + best_dx, 0, height_map.shape[1] - 1)

        inflow = np.zeros_like(height_map)
        np.add.at(
            inflow,
            (target_rows[transportable], target_cols[transportable]),
            amount[transportable],
        )

        height_map = height_map - amount + inflow

    if nodata is not None:
        height_map[~valid_mask] = nodata

    profile.update(dtype="float32")
    out_path = Path(output_path)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(height_map.astype(np.float32), 1)

    return out_path


def super_resolve_dtm(
    dem_path: Path | str,
    optical_path: Path | str,
    output_path: Path | str,
    *,
    band: int | None = None,
    radius: int = 8,
    eps: float = 1e-2,
    apply_erosion: bool = False,
    erosion_kwargs: dict | None = None,
) -> Path:
    """Super-resolve a DEM: guided upsampling, plus optional thermal erosion.

    Chains guided_upsample_dem and, when apply_erosion is True,
    simulate_thermal_erosion.

    Args:
        dem_path: Path to the source DEM GeoTIFF.
        optical_path: Path to the co-registered high-resolution optical GeoTIFF.
        output_path: Path for the final output GeoTIFF.
        band: Forwarded to guided_upsample_dem.
        radius: Forwarded to guided_upsample_dem.
        eps: Forwarded to guided_upsample_dem.
        apply_erosion: If True, run simulate_thermal_erosion on the guided
            upsampling result before writing output_path.
        erosion_kwargs: Keyword arguments forwarded to simulate_thermal_erosion
            when apply_erosion is True (e.g. iterations, talus_angle,
            transfer_rate).

    Returns:
        The resolved output_path as a Path.
    """
    output_path = Path(output_path)

    if not apply_erosion:
        return guided_upsample_dem(
            dem_path, optical_path, output_path, band=band, radius=radius, eps=eps
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        upsampled = Path(tmp_dir) / "upsampled.tif"
        guided_upsample_dem(
            dem_path, optical_path, upsampled, band=band, radius=radius, eps=eps
        )
        simulate_thermal_erosion(upsampled, output_path, **(erosion_kwargs or {}))

    return output_path
