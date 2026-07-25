"""Tests for vecraspy.raster."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from vecraspy.raster import _collect_tif_paths, merge_tifs, reproject_raster


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


def test_collect_from_list(tmp_path):
    t1 = tmp_path / "a.tif"
    t2 = tmp_path / "b.tif"
    _write_tif(t1, np.ones((4, 4), dtype=np.float32))
    _write_tif(t2, np.ones((4, 4), dtype=np.float32))
    result = _collect_tif_paths([t1, t2])
    assert result == [t1, t2]


def test_collect_from_directory(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _write_tif(tiles / "a.tif", np.ones((4, 4), dtype=np.float32))
    _write_tif(tiles / "b.tiff", np.ones((4, 4), dtype=np.float32))
    result = _collect_tif_paths(tiles)
    assert len(result) == 2
    assert all(p.suffix.lower() in {".tif", ".tiff"} for p in result)


def test_collect_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="directory not found"):
        _collect_tif_paths(tmp_path / "nonexistent")


def test_collect_empty_directory_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no TIF files found"):
        _collect_tif_paths(empty)


def test_collect_missing_file_in_list_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="file not found"):
        _collect_tif_paths([tmp_path / "ghost.tif"])


def test_merge_tifs_from_list_stitches_horizontally(tmp_path):
    t1 = tmp_path / "tile1.tif"
    t2 = tmp_path / "tile2.tif"
    _write_tif(t1, np.ones((4, 4), dtype=np.float32), west=0, east=4, south=0, north=4)
    _write_tif(
        t2, np.full((4, 4), 2.0, dtype=np.float32), west=4, east=8, south=0, north=4
    )

    out = tmp_path / "merged.tif"
    result = merge_tifs([t1, t2], out)

    assert result == out
    assert out.exists()
    with rasterio.open(out) as ds:
        assert ds.width == 8
        assert ds.height == 4
        assert ds.count == 1


def test_merge_tifs_from_directory(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _write_tif(
        tiles / "a.tif",
        np.ones((4, 4), dtype=np.float32),
        west=0,
        east=4,
        south=0,
        north=4,
    )
    _write_tif(
        tiles / "b.tif",
        np.ones((4, 4), dtype=np.float32),
        west=4,
        east=8,
        south=0,
        north=4,
    )

    out = tmp_path / "merged.tif"
    result = merge_tifs(tiles, out)

    assert result == out
    with rasterio.open(out) as ds:
        assert ds.width == 8


def test_merge_tifs_returns_path(tmp_path):
    t1 = tmp_path / "t.tif"
    _write_tif(t1, np.ones((4, 4), dtype=np.float32))
    out = tmp_path / "out.tif"
    result = merge_tifs([t1], out)
    assert isinstance(result, Path)
    assert result == out


def test_merge_tifs_preserves_nodata_from_first_tif(tmp_path):
    t1 = tmp_path / "t1.tif"
    t2 = tmp_path / "t2.tif"
    data = np.ones((4, 4), dtype=np.float32)
    data[0, 0] = -9999.0
    _write_tif(t1, data, west=0, east=4, south=0, north=4, nodata=-9999.0)
    _write_tif(
        t2,
        np.ones((4, 4), dtype=np.float32),
        west=4,
        east=8,
        south=0,
        north=4,
        nodata=-9999.0,
    )

    out = tmp_path / "merged.tif"
    merge_tifs([t1, t2], out)

    with rasterio.open(out) as ds:
        assert ds.nodata == -9999.0


def test_merge_tifs_caller_overrides_nodata(tmp_path):
    t1 = tmp_path / "t.tif"
    _write_tif(t1, np.ones((4, 4), dtype=np.float32), nodata=-1.0)
    out = tmp_path / "out.tif"
    merge_tifs([t1], out, nodata=0.0)
    with rasterio.open(out) as ds:
        assert ds.nodata == 0.0


def test_merge_tifs_invalid_resampling_raises(tmp_path):
    t1 = tmp_path / "t.tif"
    _write_tif(t1, np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="invalid resampling"):
        merge_tifs([t1], tmp_path / "out.tif", resampling="notamethod")


def test_merge_tifs_missing_file_in_list_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="file not found"):
        merge_tifs([tmp_path / "ghost.tif"], tmp_path / "out.tif")


def test_merge_tifs_missing_source_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="directory not found"):
        merge_tifs(tmp_path / "nonexistent", tmp_path / "out.tif")


def test_merge_tifs_empty_directory_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no TIF files found"):
        merge_tifs(empty, tmp_path / "out.tif")


def test_merge_tifs_reprojects_to_target_crs(tmp_path):
    t1 = tmp_path / "utm.tif"
    t2 = tmp_path / "wgs84.tif"
    _write_tif(
        t1,
        np.ones((4, 4), dtype=np.float32),
        crs="EPSG:32632",
        west=500000,
        east=500004,
        south=5400000,
        north=5400004,
    )
    _write_tif(
        t2,
        np.ones((4, 4), dtype=np.float32),
        crs="EPSG:4326",
        west=9.0,
        east=9.001,
        south=48.7,
        north=48.701,
    )

    out = tmp_path / "merged.tif"
    merge_tifs([t1, t2], out, target_crs="EPSG:32632")

    with rasterio.open(out) as ds:
        assert ds.crs.to_epsg() == 32632


def test_merge_tifs_uses_target_resolution(tmp_path):
    t1 = tmp_path / "tile.tif"
    _write_tif(t1, np.ones((4, 4), dtype=np.float32), west=0, east=4, south=0, north=4)

    out = tmp_path / "out.tif"
    merge_tifs([t1], out, target_resolution=(2.0, 2.0))

    with rasterio.open(out) as ds:
        assert abs(ds.transform.a - 2.0) < 1e-6
        assert abs(abs(ds.transform.e) - 2.0) < 1e-6


def _write_multiband_tif(
    path: Path,
    data: np.ndarray,
    *,
    crs: str = "EPSG:32632",
    west: float = 500000.0,
    south: float = 5600000.0,
    east: float = 500400.0,
    north: float = 5600400.0,
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


def test_reproject_raster_changes_crs(tmp_path):
    src_tif = tmp_path / "src.tif"
    out = tmp_path / "out_4326.tif"
    _write_multiband_tif(src_tif, np.arange(400, dtype=np.float32).reshape(1, 20, 20))

    result = reproject_raster(src_tif, out, "EPSG:4326")

    assert result == out
    with rasterio.open(out) as ds:
        assert ds.crs == CRS.from_epsg(4326)
        assert ds.count == 1
        assert ds.dtypes[0] == "float32"
        assert ds.width > 0 and ds.height > 0


def test_reproject_raster_accepts_crs_object(tmp_path):
    src_tif = tmp_path / "src.tif"
    out = tmp_path / "out.tif"
    _write_multiband_tif(src_tif, np.ones((1, 10, 10), dtype=np.float32))

    reproject_raster(src_tif, out, CRS.from_epsg(25832))

    with rasterio.open(out) as ds:
        assert ds.crs == CRS.from_epsg(25832)


def test_reproject_raster_honours_target_resolution(tmp_path):
    src_tif = tmp_path / "src.tif"
    out = tmp_path / "out.tif"
    # 400 m extent over 20 px -> 20 m pixels
    _write_multiband_tif(src_tif, np.ones((1, 20, 20), dtype=np.float32))

    reproject_raster(src_tif, out, "EPSG:25832", resolution=40.0)

    with rasterio.open(out) as ds:
        assert ds.transform.a == pytest.approx(40.0)
        assert abs(ds.transform.e) == pytest.approx(40.0)


def test_reproject_raster_preserves_band_count(tmp_path):
    src_tif = tmp_path / "src.tif"
    out = tmp_path / "out.tif"
    data = np.stack(
        [
            np.full((10, 10), 1.0, dtype=np.float32),
            np.full((10, 10), 2.0, dtype=np.float32),
        ]
    )
    _write_multiband_tif(src_tif, data)

    reproject_raster(src_tif, out, "EPSG:4326", resampling="nearest")

    with rasterio.open(out) as ds:
        assert ds.count == 2
        assert np.nanmax(ds.read(1)) == pytest.approx(1.0)
        assert np.nanmax(ds.read(2)) == pytest.approx(2.0)


def test_reproject_raster_fills_edges_with_nodata(tmp_path):
    src_tif = tmp_path / "src.tif"
    out = tmp_path / "out.tif"
    # a 500 km wide tile picks up several degrees of rotation across UTM zones
    _write_multiband_tif(
        src_tif,
        np.ones((1, 40, 40), dtype=np.float32),
        crs="EPSG:32632",
        west=300000.0,
        south=5400000.0,
        east=800000.0,
        north=5900000.0,
        nodata=-9999.0,
    )

    reproject_raster(src_tif, out, "EPSG:32633")

    with rasterio.open(out) as ds:
        assert ds.nodata == pytest.approx(-9999.0)
        data = ds.read(1)
        # the rotated footprint leaves nodata in the corners
        assert np.any(data == -9999.0)
        assert np.any(data == 1.0)


def test_reproject_raster_nodata_override(tmp_path):
    src_tif = tmp_path / "src.tif"
    out = tmp_path / "out.tif"
    _write_multiband_tif(src_tif, np.ones((1, 10, 10), dtype=np.float32))

    reproject_raster(src_tif, out, "EPSG:4326", nodata=0.0)

    with rasterio.open(out) as ds:
        assert ds.nodata == pytest.approx(0.0)


def test_reproject_raster_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="file not found"):
        reproject_raster(tmp_path / "ghost.tif", tmp_path / "out.tif", "EPSG:4326")


def test_reproject_raster_invalid_resampling_raises(tmp_path):
    src_tif = tmp_path / "src.tif"
    _write_multiband_tif(src_tif, np.ones((1, 4, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="invalid resampling"):
        reproject_raster(src_tif, tmp_path / "out.tif", "EPSG:4326", resampling="magic")


def test_reproject_raster_invalid_resolution_raises(tmp_path):
    src_tif = tmp_path / "src.tif"
    _write_multiband_tif(src_tif, np.ones((1, 4, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="resolution must be > 0"):
        reproject_raster(src_tif, tmp_path / "out.tif", "EPSG:4326", resolution=0.0)


def test_reproject_raster_without_crs_raises(tmp_path):
    src_tif = tmp_path / "src.tif"
    out = tmp_path / "out.tif"
    with rasterio.open(
        src_tif,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        transform=from_bounds(0, 0, 4, 4, 4, 4),
    ) as dst:
        dst.write(np.ones((4, 4), dtype=np.float32), 1)

    with pytest.raises(rasterio.errors.CRSError, match="no CRS"):
        reproject_raster(src_tif, out, "EPSG:4326")
