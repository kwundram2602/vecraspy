"""Tests for vecraspy.vector."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from vecraspy.vector import Trajectory, read_points


def _write_gpkg(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a minimal point GeoPackage from a list of row dicts (must include 'geometry')."""
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    path = tmp_path / "test.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def test_read_points_returns_geodataframe(tmp_path):
    path = _write_gpkg(tmp_path, [
        {"geometry": Point(1.0, 2.0)},
        {"geometry": Point(3.0, 4.0)},
    ])
    gdf = read_points(path)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 2


def test_read_points_preserves_crs(tmp_path):
    path = _write_gpkg(tmp_path, [{"geometry": Point(1.0, 2.0)}])
    gdf = read_points(path)
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326


def test_trajectory_dataclass():
    gdf = gpd.GeoDataFrame({"geometry": [Point(0.0, 0.0)]}, crs="EPSG:4326")
    traj = Trajectory(id="track_1", points=gdf)
    assert traj.id == "track_1"
    assert len(traj.points) == 1
