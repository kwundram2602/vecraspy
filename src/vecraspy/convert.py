"""Convert AOIs between GeoDataFrames, WKT, GeoJSON and bounding boxes.

Bounding boxes are always ``(minx, miny, maxx, maxy)``, the order STAC, GeoJSON
and shapely use. ``crs`` arguments take anything ``GeoDataFrame.to_crs`` accepts
(``"EPSG:4326"``, an EPSG code, WKT, a ``pyproj.CRS``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import shapely
from shapely.errors import ShapelyError
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry

BBox = tuple[float, float, float, float]
AoiLike = gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | Path | str


def _to_gdf(aoi: AoiLike, crs: Any = None) -> gpd.GeoDataFrame:
    """Normalise an AOI to a GeoDataFrame, reprojected to ``crs`` if given."""
    if isinstance(aoi, gpd.GeoDataFrame):
        gdf = aoi
    elif isinstance(aoi, gpd.GeoSeries):
        gdf = gpd.GeoDataFrame(geometry=aoi)
    elif isinstance(aoi, BaseGeometry):
        gdf = gpd.GeoDataFrame(geometry=[aoi])
    elif isinstance(aoi, (Path, str)):
        gdf = gpd.read_file(aoi)
    else:
        raise TypeError(
            "aoi must be a GeoDataFrame, GeoSeries, shapely geometry or a path, "
            f"got {type(aoi).__name__}"
        )
    if gdf.empty or gdf.geometry.is_empty.all():
        raise ValueError("AOI contains no geometry")
    if crs is not None:
        if gdf.crs is None:
            raise ValueError(f"AOI has no CRS defined; cannot reproject to {crs!r}.")
        gdf = gdf.to_crs(crs)
    return gdf


def _round(geom: BaseGeometry, precision: int | None) -> BaseGeometry:
    if precision is None:
        return geom
    return shapely.transform(geom, lambda coords: np.round(coords, precision))


def _parse_bbox(bbox: Sequence[float] | str) -> BBox:
    """Validate a bbox given as four numbers or a string like ``"1,2,3,4"``."""
    if isinstance(bbox, str):
        parts = [p for p in re.split(r"[,;\s]+", bbox.strip().strip("[]()")) if p]
    else:
        parts = list(bbox)
    if len(parts) != 4:
        raise ValueError(
            f"bbox needs four values (minx, miny, maxx, maxy), got {len(parts)}"
        )
    try:
        minx, miny, maxx, maxy = (float(p) for p in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bbox values must be numeric: {exc}") from exc
    if minx >= maxx or miny >= maxy:
        raise ValueError(f"bbox must satisfy minx < maxx and miny < maxy, got {bbox!r}")
    return minx, miny, maxx, maxy


def _from_wkt(wkt: str) -> BaseGeometry:
    try:
        return shapely.from_wkt(wkt.strip())
    except ShapelyError as exc:
        raise ValueError(f"invalid WKT: {exc}") from exc


def aoi_to_wkt(
    aoi: AoiLike,
    *,
    crs: Any = None,
    precision: int | None = 6,
) -> str:
    """Return the union of all AOI geometries as a WKT string.

    Args:
        aoi: GeoDataFrame, GeoSeries, shapely geometry or path to a vector file.
        crs: Output CRS. None keeps the AOI's own CRS.
        precision: Decimal places of the coordinates; None keeps full precision.
            6 is about 0.1 m in degrees; use fewer for projected CRSs.

    Raises:
        ValueError: If the AOI is empty, or ``crs`` is given but the AOI has none.
    """
    geom = _to_gdf(aoi, crs).geometry.union_all()
    return shapely.to_wkt(
        geom, rounding_precision=-1 if precision is None else precision
    )


def aoi_to_bbox(aoi: AoiLike, *, crs: Any = None) -> BBox:
    """Return the AOI's bounding box as ``(minx, miny, maxx, maxy)``.

    Args:
        aoi: GeoDataFrame, GeoSeries, shapely geometry or path to a vector file.
        crs: CRS of the returned bounds. None keeps the AOI's own CRS. The AOI is
            reprojected before taking the bounds, so the box covers it fully.
    """
    minx, miny, maxx, maxy = _to_gdf(aoi, crs).total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def aoi_to_geojson(
    aoi: AoiLike,
    *,
    crs: Any = "EPSG:4326",
    precision: int | None = 6,
) -> str:
    """Return the union of all AOI geometries as a GeoJSON geometry string.

    Args:
        aoi: GeoDataFrame, GeoSeries, shapely geometry or path to a vector file.
        crs: Output CRS. Defaults to EPSG:4326, the only CRS RFC 7946 allows;
            None keeps the AOI's own CRS.
        precision: Decimal places of the coordinates; None keeps full precision.
    """
    geom = _round(_to_gdf(aoi, crs).geometry.union_all(), precision)
    return json.dumps(mapping(geom))


def wkt_to_aoi(wkt: str, crs: Any = "EPSG:4326") -> gpd.GeoDataFrame:
    """Return a single-row GeoDataFrame from a WKT geometry in ``crs``.

    Raises:
        ValueError: If the WKT cannot be parsed.
    """
    return gpd.GeoDataFrame(geometry=[_from_wkt(wkt)], crs=crs)


def geojson_to_aoi(
    geojson: str | dict[str, Any],
    crs: Any = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame from a GeoJSON geometry, Feature or FeatureCollection.

    Features keep their properties as columns; a bare geometry gives one row.

    Raises:
        ValueError: If the string is not valid JSON or not a GeoJSON object.
    """
    if isinstance(geojson, str):
        try:
            obj = json.loads(geojson)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid GeoJSON: {exc}") from exc
    else:
        obj = geojson
    kind = obj.get("type") if isinstance(obj, dict) else None
    if kind == "FeatureCollection":
        return gpd.GeoDataFrame.from_features(obj["features"], crs=crs)
    if kind == "Feature":
        return gpd.GeoDataFrame.from_features([obj], crs=crs)
    if kind is None:
        raise ValueError("GeoJSON object needs a 'type' member")
    try:
        geom = shape(obj)
    except (AttributeError, KeyError, TypeError, ValueError, ShapelyError) as exc:
        raise ValueError(f"invalid GeoJSON geometry: {exc}") from exc
    return gpd.GeoDataFrame(geometry=[geom], crs=crs)


def bbox_to_aoi(
    bbox: Sequence[float] | str,
    crs: Any = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Return a single-row GeoDataFrame holding the bbox rectangle.

    Args:
        bbox: ``(minx, miny, maxx, maxy)`` as four numbers or a string such as
            ``"7.1,51.9,7.3,52.0"`` (commas, semicolons or spaces; brackets allowed).
        crs: CRS of the bbox coordinates.

    Raises:
        ValueError: If there are not four numbers or min >= max.
    """
    return gpd.GeoDataFrame(geometry=[box(*_parse_bbox(bbox))], crs=crs)


def wkt_to_bbox(wkt: str) -> BBox:
    """Return the bounds ``(minx, miny, maxx, maxy)`` of a WKT geometry."""
    geom = _from_wkt(wkt)
    if geom.is_empty:
        raise ValueError("WKT geometry is empty")
    minx, miny, maxx, maxy = geom.bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def bbox_to_wkt(bbox: Sequence[float] | str, *, precision: int | None = None) -> str:
    """Return the bbox rectangle as a WKT polygon.

    Args:
        bbox: ``(minx, miny, maxx, maxy)`` as four numbers or a string, see
            :func:`bbox_to_aoi`.
        precision: Decimal places of the coordinates; None keeps them as given.
    """
    geom = box(*_parse_bbox(bbox))
    return shapely.to_wkt(
        geom, rounding_precision=-1 if precision is None else precision
    )
