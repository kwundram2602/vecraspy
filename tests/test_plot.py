"""Tests for vecraspy.plot."""

import math
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

matplotlib.use("Agg")  # non-interactive backend for tests

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _write_slope_tif(
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


import pytest

from vecraspy.plot import plot_slope


def test_plot_slope_invalid_units_raises(tmp_path):
    tif = tmp_path / "slope.tif"
    _write_slope_tif(tif, np.full((4, 4), 30.0, dtype=np.float32))
    with pytest.raises(ValueError, match="invalid units"):
        plot_slope(tif, units="gradians")


def test_plot_slope_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        plot_slope(tmp_path / "ghost.tif")


def test_plot_slope_all_nodata_raises(tmp_path):
    tif = tmp_path / "slope.tif"
    data = np.full((4, 4), -9999.0, dtype=np.float32)
    _write_slope_tif(tif, data, nodata=-9999.0)
    with pytest.raises(ValueError, match="no valid"):
        plot_slope(tif)
