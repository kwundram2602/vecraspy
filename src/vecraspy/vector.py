"""Point handling and trajectory building."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
from shapely import concave_hull
from shapely.geometry import MultiPoint
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box


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

def get_bounds_as_gdf(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with a single row containing the bounding box of gdf."""
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    bounds_gdf = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=gdf.crs)
    return bounds_gdf

def compute_aoi(
    gdf: gpd.GeoDataFrame,
    coverage: float,
    *,
    method: Literal["convex", "concave"] = "convex",
    concave_ratio: float = 0.0,
) -> BaseGeometry:
    """Return a polygon enclosing `coverage` fraction of the points.

    Sorts points by distance from the centroid, retains the closest
    ``coverage`` fraction, then builds the polygon using the chosen method.

    Args:
        gdf: GeoDataFrame of point geometries.
        coverage: Fraction of points to include, in (0, 1].
            E.g. 0.95 keeps the 95 % of points closest to the centroid.
        method: Hull algorithm. ``"convex"`` returns the Minimum Convex
            Polygon (MCP). ``"concave"`` follows the point cloud more
            tightly and is better for elongated or irregular distributions.
        concave_ratio: Only used when ``method="concave"``. Controls how
            closely the hull follows the points: 0.0 (default) = tightest
            fit, 1.0 = convex hull. Passed to ``shapely.concave_hull``.

    Returns:
        Hull geometry enclosing the retained points (typically a Polygon).

    Raises:
        ValueError: If coverage is not in (0, 1].
        ValueError: If gdf contains fewer than 3 points.
    """
    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1], got {coverage!r}")
    if len(gdf) < 3:
        raise ValueError(f"at least 3 points required, got {len(gdf)}")

    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    centroid = coords.mean(axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)

    n_keep = math.ceil(len(gdf) * coverage)
    retained = MultiPoint(coords[np.argsort(distances)[:n_keep]])

    if method == "concave":
        return concave_hull(retained, ratio=concave_ratio)
    return retained.convex_hull


def _safe_stem(value: Any, fallback: str) -> str:
    name = str(value) if value is not None else fallback
    name = name.strip()
    if not name:
        return fallback
    cleaned: list[str] = []
    for char in name:
        if char.isascii() and (char.isalnum() or char in {"-", "_", "."}):
            cleaned.append(char)
        else:
            cleaned.append("_")
    stem = "".join(cleaned).strip("._-")
    return stem or fallback


def write_trajectories(
    trajectories: list[Trajectory],
    output_dir: Path | str,
    *,
    driver: str = "GPKG",
    layer: str | None = None,
    overwrite: bool = True,
) -> list[Path]:
    """Write each trajectory to its own GeoPackage.

    Filenames are derived from trajectory ids, with unsafe characters replaced
    by underscores.

    Args:
        trajectories: Trajectories to write; each becomes its own file.
        output_dir: Directory where files are written.
        driver: Fiona driver name; defaults to "GPKG".
        layer: Optional layer name to write; uses the default if None.
        overwrite: Whether to overwrite existing files.

    Returns:
        Paths to the written files.

    Raises:
        FileExistsError: If overwrite is False and a target file exists.
        NotADirectoryError: If output_dir exists and is not a directory.
        ValueError: If two trajectories produce the same filename.
    """
    out_dir = Path(output_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise NotADirectoryError(f"output_dir must be a directory, got {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    used: set[str] = set()
    for index, traj in enumerate(trajectories, start=1):
        stem = _safe_stem(traj.id, fallback=f"trajectory_{index}")
        if stem in used:
            raise ValueError(f"duplicate trajectory id produces filename '{stem}.gpkg'")
        used.add(stem)
        path = out_dir / f"{stem}.gpkg"
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"output file already exists: {path}")
            path.unlink()
        kwargs: dict[str, Any] = {"driver": driver}
        if layer is not None:
            kwargs["layer"] = layer
        traj.points.to_file(path, **kwargs)
        outputs.append(path)
    return outputs
