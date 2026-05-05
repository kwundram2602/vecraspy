"""Point handling and trajectory building."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd


@dataclass
class Trajectory:
    """An ordered sequence of points representing a single track."""

    id: Any  # groupby key — intentionally unbounded
    points: gpd.GeoDataFrame

    def __post_init__(self) -> None:
        if not isinstance(self.points, gpd.GeoDataFrame):
            raise TypeError(f"points must be a GeoDataFrame, got {type(self.points)}")


def read_points(
    path: Path | str,
    layer: str | None = None,
) -> gpd.GeoDataFrame:
    """Read a point layer from any OGR-supported vector file.

    Args:
        path: Path to the vector file (GeoPackage, Shapefile, etc.).
        layer: Layer name or index. If None, reads the first/default layer.
    """
    kwargs: dict[str, Any] = {}
    if layer is not None:
        kwargs["layer"] = layer
    return gpd.read_file(path, **kwargs)


def group_by_id(
    gdf: gpd.GeoDataFrame,
    id_col: str,
) -> dict[Any, gpd.GeoDataFrame]:
    """Split a GeoDataFrame into groups by the values in id_col."""
    return {val: group.copy() for val, group in gdf.groupby(id_col)}


def build_trajectory(
    gdf: gpd.GeoDataFrame,
    sort_col: str | None = None,
    id: Any = None,
) -> Trajectory:
    """Build a Trajectory from an ordered (or sortable) GeoDataFrame of points."""
    pts = gdf.sort_values(sort_col) if sort_col is not None else gdf
    return Trajectory(id=id, points=pts.copy())


def build_trajectories(
    path: Path | str,
    id_col: str | None = None,
    sort_col: str | None = None,
    layer: str | None = None,
) -> list[Trajectory]:
    """Read a GeoPackage and return one Trajectory per unique id_col value.

    If id_col is None, the entire dataset is returned as a single Trajectory.
    """
    gdf = read_points(path, layer=layer)
    if id_col is None:
        return [build_trajectory(gdf, sort_col=sort_col)]
    groups = group_by_id(gdf, id_col)
    return [
        build_trajectory(pts, sort_col=sort_col, id=id_val)
        for id_val, pts in groups.items()
    ]
