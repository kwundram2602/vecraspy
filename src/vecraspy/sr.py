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
