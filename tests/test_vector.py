"""Tests for vecraspy.vector."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from vecraspy.vector import Trajectory, read_points, group_by_id, build_trajectory, build_trajectories


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
    geom = Point(0.0, 0.0)
    gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")
    traj = Trajectory(id="track_1", points=gdf)
    assert traj.id == "track_1"
    assert isinstance(traj.points, gpd.GeoDataFrame)
    assert traj.points.geometry.iloc[0] == geom


def test_trajectory_rejects_non_geodataframe():
    import pandas as pd
    with pytest.raises(TypeError, match="GeoDataFrame"):
        Trajectory(id="x", points=pd.DataFrame({"a": [1]}))


def test_read_points_layer_kwarg(tmp_path):
    gdf_a = gpd.GeoDataFrame({"geometry": [Point(1.0, 1.0)]}, crs="EPSG:4326")
    gdf_b = gpd.GeoDataFrame({"geometry": [Point(9.0, 9.0)]}, crs="EPSG:4326")
    path = tmp_path / "multi.gpkg"
    gdf_a.to_file(path, driver="GPKG", layer="layer_a")
    gdf_b.to_file(path, driver="GPKG", layer="layer_b")

    result = read_points(path, layer="layer_b")
    assert len(result) == 1
    assert result.geometry.iloc[0].x == pytest.approx(9.0)


def test_group_by_id_splits_into_correct_groups(tmp_path):
    path = _write_gpkg(tmp_path, [
        {"geometry": Point(1.0, 1.0), "track_id": "a"},
        {"geometry": Point(2.0, 2.0), "track_id": "b"},
        {"geometry": Point(3.0, 3.0), "track_id": "a"},
    ])
    gdf = read_points(path)
    groups = group_by_id(gdf, "track_id")
    assert set(groups.keys()) == {"a", "b"}
    assert len(groups["a"]) == 2
    assert len(groups["b"]) == 1


def test_group_by_id_returns_geodataframes(tmp_path):
    path = _write_gpkg(tmp_path, [
        {"geometry": Point(0.0, 0.0), "id": 1},
        {"geometry": Point(1.0, 1.0), "id": 2},
    ])
    gdf = read_points(path)
    groups = group_by_id(gdf, "id")
    for val in groups.values():
        assert isinstance(val, gpd.GeoDataFrame)


def test_build_trajectory_preserves_points():
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(1.0, 1.0), Point(2.0, 2.0)]},
        crs="EPSG:4326",
    )
    traj = build_trajectory(gdf, id="x")
    assert isinstance(traj, Trajectory)
    assert len(traj.points) == 2
    assert traj.id == "x"


def test_build_trajectory_sorts_by_sort_col():
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [Point(3.0, 3.0), Point(1.0, 1.0), Point(2.0, 2.0)],
            "t": [3, 1, 2],
        },
        crs="EPSG:4326",
    )
    traj = build_trajectory(gdf, sort_col="t")
    xs = [geom.x for geom in traj.points.geometry]
    assert xs == [1.0, 2.0, 3.0]


def test_build_trajectories_no_id_col(tmp_path):
    path = _write_gpkg(tmp_path, [
        {"geometry": Point(0.0, 0.0)},
        {"geometry": Point(1.0, 1.0)},
    ])
    result = build_trajectories(path)
    assert len(result) == 1
    assert result[0].id is None
    assert len(result[0].points) == 2


def test_build_trajectories_with_id_col(tmp_path):
    path = _write_gpkg(tmp_path, [
        {"geometry": Point(0.0, 0.0), "tid": "a"},
        {"geometry": Point(1.0, 0.0), "tid": "b"},
        {"geometry": Point(2.0, 0.0), "tid": "a"},
    ])
    result = build_trajectories(path, id_col="tid")
    assert len(result) == 2
    ids = {t.id for t in result}
    assert ids == {"a", "b"}


def test_build_trajectories_sort_col(tmp_path):
    path = _write_gpkg(tmp_path, [
        {"geometry": Point(3.0, 0.0), "tid": "a", "t": 3},
        {"geometry": Point(1.0, 0.0), "tid": "a", "t": 1},
        {"geometry": Point(2.0, 0.0), "tid": "a", "t": 2},
    ])
    result = build_trajectories(path, id_col="tid", sort_col="t")
    xs = [geom.x for geom in result[0].points.geometry]
    assert xs == [1.0, 2.0, 3.0]
