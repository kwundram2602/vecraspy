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
