"""Tests for vecraspy.sr."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from vecraspy.raster import align_raster_grid
from vecraspy.sr import (
    _box_filter,
    guided_upsample_dem,
    simulate_thermal_erosion,
    super_resolve_dtm,
)


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
        optical,
        np.ones((1, 16, 16), dtype=np.float32),
        west=0,
        south=0,
        east=40,
        north=40,
    )

    out = tmp_path / "out.tif"
    guided_upsample_dem(dem, optical, out, radius=2)

    with rasterio.open(out) as ds:
        result = ds.read(1)

    with tempfile.TemporaryDirectory() as tmp:
        baseline = Path(tmp) / "baseline.tif"
        align_raster_grid(optical, dem, baseline, resampling="cubic")
        with rasterio.open(baseline) as ds:
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
        optical,
        np.ones((1, 16, 16), dtype=np.float32),
        west=0,
        south=0,
        east=40,
        north=40,
    )

    out = tmp_path / "out.tif"
    guided_upsample_dem(dem, optical, out)

    with rasterio.open(out) as ds:
        assert ds.nodata == -9999.0
        data = ds.read(1)
        assert data[0, 0] == -9999.0
        assert data[-1, 0] != -9999.0


def test_guided_upsample_dem_missing_dem_raises(tmp_path):
    optical = tmp_path / "optical.tif"
    _write_multiband_tif(optical, np.ones((1, 4, 4), dtype=np.float32))
    with pytest.raises(FileNotFoundError, match="file not found"):
        guided_upsample_dem(tmp_path / "ghost.tif", optical, tmp_path / "out.tif")


def test_guided_upsample_dem_missing_optical_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    with pytest.raises(FileNotFoundError, match="file not found"):
        guided_upsample_dem(dem, tmp_path / "ghost.tif", tmp_path / "out.tif")


def test_guided_upsample_dem_invalid_radius_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    _write_multiband_tif(optical, np.ones((1, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="radius must be > 0"):
        guided_upsample_dem(dem, optical, tmp_path / "out.tif", radius=0)


def test_guided_upsample_dem_invalid_eps_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    _write_multiband_tif(optical, np.ones((1, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="eps must be > 0"):
        guided_upsample_dem(dem, optical, tmp_path / "out.tif", eps=0)


def test_guided_upsample_dem_invalid_band_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    _write_multiband_tif(optical, np.ones((1, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match=r"band must be in \[1, 1\]"):
        guided_upsample_dem(dem, optical, tmp_path / "out.tif", band=2)


def test_guided_upsample_dem_invalid_resampling_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    _write_multiband_tif(optical, np.ones((1, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="invalid resampling"):
        guided_upsample_dem(dem, optical, tmp_path / "out.tif", resampling="magic")


def test_simulate_thermal_erosion_flat_terrain_unchanged(tmp_path):
    dem = tmp_path / "dem.tif"
    data = np.full((10, 10), 50.0, dtype=np.float32)
    _write_tif(dem, data, west=0, south=0, east=10, north=10)

    out = tmp_path / "out.tif"
    simulate_thermal_erosion(dem, out, iterations=10)

    with rasterio.open(out) as ds:
        result = ds.read(1)
    np.testing.assert_allclose(result, data, atol=1e-6)


def test_simulate_thermal_erosion_reduces_spike(tmp_path):
    dem = tmp_path / "dem.tif"
    data = np.full((9, 9), 10.0, dtype=np.float32)
    data[4, 4] = 100.0
    _write_tif(dem, data, west=0, south=0, east=9, north=9)

    out = tmp_path / "out.tif"
    simulate_thermal_erosion(
        dem, out, iterations=20, talus_angle=45.0, transfer_rate=0.5
    )

    with rasterio.open(out) as ds:
        result = ds.read(1)

    assert result[4, 4] < 100.0
    assert result[4, 4] > 10.0


def test_simulate_thermal_erosion_conserves_total_mass(tmp_path):
    dem = tmp_path / "dem.tif"
    data = np.full((9, 9), 10.0, dtype=np.float32)
    data[4, 4] = 100.0
    _write_tif(dem, data, west=0, south=0, east=9, north=9)

    out = tmp_path / "out.tif"
    simulate_thermal_erosion(
        dem, out, iterations=20, talus_angle=45.0, transfer_rate=0.5
    )

    with rasterio.open(out) as ds:
        result = ds.read(1)

    assert result.sum() == pytest.approx(data.sum(), rel=1e-4)


def test_simulate_thermal_erosion_ignores_nodata_neighbors(tmp_path):
    dem = tmp_path / "dem.tif"
    data = np.full((9, 9), 10.0, dtype=np.float32)
    data[4, 4] = 100.0
    data[4, 5] = -9999.0  # nodata cell right next to the spike
    _write_tif(dem, data, west=0, south=0, east=9, north=9, nodata=-9999.0)

    out = tmp_path / "out.tif"
    simulate_thermal_erosion(
        dem, out, iterations=20, talus_angle=45.0, transfer_rate=0.5
    )

    with rasterio.open(out) as ds:
        result = ds.read(1)
        valid = ds.read_masks(1) != 0

    # The nodata cell must stay untouched, and no real elevation may drain
    # into it — total mass over the *valid* region alone is conserved.
    assert result[4, 5] == -9999.0
    assert result[valid].sum() == pytest.approx(data[valid].sum(), rel=1e-4)


def test_simulate_thermal_erosion_returns_path(tmp_path):
    dem = tmp_path / "dem.tif"
    _write_tif(dem, np.full((4, 4), 5.0, dtype=np.float32))
    out = tmp_path / "out.tif"
    result = simulate_thermal_erosion(dem, out)
    assert isinstance(result, Path)
    assert result == out


def test_simulate_thermal_erosion_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="file not found"):
        simulate_thermal_erosion(tmp_path / "ghost.tif", tmp_path / "out.tif")


def test_simulate_thermal_erosion_invalid_iterations_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="iterations must be > 0"):
        simulate_thermal_erosion(dem, tmp_path / "out.tif", iterations=0)


def test_simulate_thermal_erosion_invalid_talus_angle_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="talus_angle must be in"):
        simulate_thermal_erosion(dem, tmp_path / "out.tif", talus_angle=90.0)


def test_simulate_thermal_erosion_invalid_transfer_rate_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="transfer_rate must be in"):
        simulate_thermal_erosion(dem, tmp_path / "out.tif", transfer_rate=0.0)


def test_super_resolve_dtm_without_erosion_matches_guided_upsample(tmp_path):
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
    _write_multiband_tif(
        optical,
        np.ones((1, 16, 16), dtype=np.float32),
        west=0,
        south=0,
        east=40,
        north=40,
    )

    direct = tmp_path / "direct.tif"
    guided_upsample_dem(dem, optical, direct)

    via_pipeline = tmp_path / "pipeline.tif"
    result = super_resolve_dtm(dem, optical, via_pipeline)

    assert result == via_pipeline
    with rasterio.open(direct) as a, rasterio.open(via_pipeline) as b:
        np.testing.assert_allclose(a.read(1), b.read(1))


def test_super_resolve_dtm_with_erosion_applies_erosion(tmp_path):
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
    _write_multiband_tif(
        optical,
        np.ones((1, 16, 16), dtype=np.float32),
        west=0,
        south=0,
        east=40,
        north=40,
    )

    without_erosion = tmp_path / "without.tif"
    guided_upsample_dem(dem, optical, without_erosion, radius=2)

    with_erosion = tmp_path / "with.tif"
    super_resolve_dtm(
        dem,
        optical,
        with_erosion,
        radius=2,
        apply_erosion=True,
        erosion_kwargs={"iterations": 5, "talus_angle": 5.0, "transfer_rate": 1.0},
    )

    with rasterio.open(without_erosion) as a, rasterio.open(with_erosion) as b:
        data_a = a.read(1)
        data_b = b.read(1)
    assert not np.allclose(data_a, data_b)
