"""Raster utility functions."""

from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon


def _collect_tif_paths(source: list[Path | str] | Path | str) -> list[Path]:
    if isinstance(source, list):
        paths = [Path(p) for p in source]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"file not found: {p}")
        return paths

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"directory not found: {source_path}")

    paths = sorted(
        p for p in source_path.iterdir()
        if p.suffix.lower() in {".tif", ".tiff"}
    )
    if not paths:
        raise ValueError(f"no TIF files found in directory: {source_path}")

    return paths


def tif_bounds_as_polygon(path: Path | str) -> Polygon:
    """Return the bounding box of a GeoTIFF as a Shapely polygon.

    The polygon is in the CRS of the raster (no reprojection).

    Args:
        path: Path to the GeoTIFF file.

    Returns:
        A rectangular Shapely polygon covering the raster extent.

    Raises:
        FileNotFoundError: If the file does not exist.
        rasterio.errors.RasterioIOError: If the file cannot be opened.
    """
    tif_path = Path(path)
    if not tif_path.exists():
        raise FileNotFoundError(f"file not found: {tif_path}")

    with rasterio.open(tif_path) as dataset:
        bounds = dataset.bounds

    return box(bounds.left, bounds.bottom, bounds.right, bounds.top)


def filter_tifs_by_aoi(
    tif_paths: list[Path | str],
    aoi: BaseGeometry | gpd.GeoDataFrame | gpd.GeoSeries,
    aoi_crs: str = "EPSG:4326",
) -> list[Path]:
    """Return only the TIFs whose spatial extent intersects the AOI.

    Each TIF's bounds are reprojected to the AOI CRS before testing intersection,
    so TIFs with differing projections are handled correctly.

    Args:
        tif_paths: Paths to candidate GeoTIFF files.
        aoi: AOI as a Shapely geometry, GeoDataFrame, or GeoSeries in the CRS
            given by aoi_crs. A GeoDataFrame/GeoSeries is dissolved to a single
            geometry via union_all().
        aoi_crs: EPSG string or WKT for the AOI geometry's CRS. Defaults to WGS-84.

    Returns:
        Subset of tif_paths whose extents intersect aoi (fully or partly).

    Raises:
        rasterio.errors.RasterioIOError: If a file cannot be opened as a raster.
    """
    if isinstance(aoi, gpd.GeoDataFrame):
        aoi_geom: BaseGeometry = aoi.geometry.union_all()
    elif isinstance(aoi, gpd.GeoSeries):
        aoi_geom = aoi.union_all()
    else:
        aoi_geom = aoi

    target_crs = CRS.from_user_input(aoi_crs)
    matches: list[Path] = []

    for raw_path in tif_paths:
        path = Path(raw_path)
        with rasterio.open(path) as src:
            west, south, east, north = transform_bounds(src.crs, target_crs, *src.bounds)
        tif_bbox = box(west, south, east, north)
        if aoi_geom.intersects(tif_bbox):
            matches.append(path)

    return matches
