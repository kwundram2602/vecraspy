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
