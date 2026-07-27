"""Tests for vecraspy.sr."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds


def _write_tif(
    path: Path,
    data: np.ndarray,
    *,
    crs: str = "EPSG:32632",
    west: float = 0.0,
    south: float = 0.0,
    east: float = 4.0,
    north: float = 4.0,
    nodata: float | None = None,
) -> None:
    height, width = data.shape
    transform = from_bounds(west, south, east, north, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=CRS.from_user_input(crs),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def _write_multiband_tif(
    path: Path,
    data: np.ndarray,
    *,
    crs: str = "EPSG:32632",
    west: float = 0.0,
    south: float = 0.0,
    east: float = 4.0,
    north: float = 4.0,
    nodata: float | None = None,
) -> None:
    count, height, width = data.shape
    transform = from_bounds(west, south, east, north, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=CRS.from_user_input(crs),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)


from vecraspy.sr import _box_filter


def test_box_filter_constant_array_unchanged():
    arr = np.full((10, 10), 5.0)
    result = _box_filter(arr, radius=2)
    np.testing.assert_allclose(result, arr)


def test_box_filter_matches_naive_reference():
    rng = np.random.default_rng(42)
    arr = rng.random((10, 10))
    radius = 2
    result = _box_filter(arr, radius)

    padded = np.pad(arr, radius, mode="edge")
    window = 2 * radius + 1
    expected = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            expected[i, j] = padded[i : i + window, j : j + window].mean()

    np.testing.assert_allclose(result, expected, rtol=1e-10)


from vecraspy.sr import guided_upsample_dem
from vecraspy.raster import align_raster_grid


def test_guided_upsample_dem_output_matches_optical_grid(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"
    _write_tif(
        dem,
        np.arange(16, dtype=np.float32).reshape(4, 4) + 10.0,
        west=0,
        south=0,
        east=40,
        north=40,
    )
    guide = np.stack([np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)] * 3)
    _write_multiband_tif(optical, guide, west=0, south=0, east=40, north=40)

    out = tmp_path / "out.tif"
    result = guided_upsample_dem(dem, optical, out)

    assert result == out
    with rasterio.open(optical) as opt_src, rasterio.open(out) as out_src:
        assert out_src.width == opt_src.width
        assert out_src.height == opt_src.height
        assert out_src.crs == opt_src.crs


def test_guided_upsample_dem_constant_guide_returns_box_filtered_baseline(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"
    dem_data = np.arange(16, dtype=np.float32).reshape(4, 4) + 10.0
    _write_tif(dem, dem_data, west=0, south=0, east=40, north=40)
    _write_multiband_tif(
        optical, np.ones((1, 16, 16), dtype=np.float32), west=0, south=0, east=40, north=40
    )

    out = tmp_path / "out.tif"
    guided_upsample_dem(dem, optical, out, radius=2)

    with rasterio.open(out) as ds:
        result = ds.read(1)

    with tempfile.TemporaryDirectory() as tmp:
        baseline = Path(tmp) / "baseline.tif"
        align_raster_grid(optical, dem, baseline, resampling="cubic")
        with rasterio.open(baseline) as ds:
            from vecraspy.sr import _box_filter

            # A constant guide carries no structure: var_I and cov_Ip are both
            # exactly zero everywhere, so a=0 and b=mean_p. The guided filter's
            # own final averaging step then box-filters b again (mean_b), so
            # the closed form is a box filter applied TWICE, not once.
            once = _box_filter(ds.read(1).astype(np.float64), 2)
            expected = _box_filter(once, 2)

    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_guided_upsample_dem_sharpens_edge_relative_to_baseline(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"

    dem_data = np.zeros((4, 4), dtype=np.float32)
    dem_data[:, 2:] = 100.0
    _write_tif(dem, dem_data, west=0, south=0, east=40, north=40)

    guide_data = np.zeros((1, 16, 16), dtype=np.float32)
    guide_data[:, :, 8:] = 1.0
    _write_multiband_tif(optical, guide_data, west=0, south=0, east=40, north=40)

    baseline = tmp_path / "baseline.tif"
    align_raster_grid(optical, dem, baseline, resampling="cubic")

    out = tmp_path / "out.tif"
    guided_upsample_dem(dem, optical, out, radius=3, eps=1e-4)

    with rasterio.open(baseline) as ds:
        baseline_row = ds.read(1)[8]
    with rasterio.open(out) as ds:
        guided_row = ds.read(1)[8]

    # The guided filter tracks the guide's edge strongly (small eps), producing
    # one steep jump right at the guide boundary, plus shallow decay on both
    # sides from the window averaging — a wider *span* of non-extreme pixels
    # than the bicubic baseline, but a much steeper *single-step* jump at the
    # edge itself. Steepest single-step gradient is the right sharpness metric.
    def steepest_gradient(row):
        return float(np.max(np.abs(np.diff(row))))

    assert steepest_gradient(guided_row) > steepest_gradient(baseline_row)


def test_guided_upsample_dem_preserves_nodata(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"

    dem_data = np.full((4, 4), 10.0, dtype=np.float32)
    dem_data[0, :] = -9999.0
    _write_tif(dem, dem_data, west=0, south=0, east=40, north=40, nodata=-9999.0)
    _write_multiband_tif(
        optical, np.ones((1, 16, 16), dtype=np.float32), west=0, south=0, east=40, north=40
    )

    out = tmp_path / "out.tif"
    guided_upsample_dem(dem, optical, out)

    with rasterio.open(out) as ds:
        assert ds.nodata == -9999.0
        data = ds.read(1)
        assert data[0, 0] == -9999.0
        assert data[-1, 0] != -9999.0
