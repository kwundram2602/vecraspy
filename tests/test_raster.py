"""Tests for vecraspy.raster."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from vecraspy.raster import _collect_tif_paths


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
