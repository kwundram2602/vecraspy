"""Point handling and trajectory building."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd


@dataclass
class Trajectory:
    """An ordered sequence of points representing a single track."""

    id: Any
    points: gpd.GeoDataFrame


def read_points(
    path: Path | str,
    layer: str | None = None,
) -> gpd.GeoDataFrame:
    """Read a point layer from a GeoPackage."""
    kwargs: dict[str, Any] = {}
    if layer is not None:
        kwargs["layer"] = layer
    return gpd.read_file(path, **kwargs)
