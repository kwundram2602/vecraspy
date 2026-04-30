"""Tests for vecraspy.raster."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from vecraspy.raster import _collect_tif_paths, merge_tifs


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
    _write_tif(t2, np.full((4, 4), 2.0, dtype=np.float32), west=4, east=8, south=0, north=4)

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
    _write_tif(tiles / "a.tif", np.ones((4, 4), dtype=np.float32), west=0, east=4, south=0, north=4)
    _write_tif(tiles / "b.tif", np.ones((4, 4), dtype=np.float32), west=4, east=8, south=0, north=4)

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
    _write_tif(t2, np.ones((4, 4), dtype=np.float32), west=4, east=8, south=0, north=4, nodata=-9999.0)

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
