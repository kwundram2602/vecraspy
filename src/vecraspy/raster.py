"""Raster utility functions."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge as _rasterio_merge
from rasterio.vrt import WarpedVRT
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


_VALID_RESAMPLING_NAMES: frozenset[str] = frozenset(r.name for r in Resampling)


def merge_tifs(
    source: list[Path | str] | Path | str,
    output_path: Path | str,
    *,
    target_crs: str | None = None,
    target_resolution: tuple[float, float] | None = None,
    resampling: str = "bilinear",
    nodata: float | None = None,
) -> Path:
    """Stitch multiple GeoTIFFs into a single output GeoTIFF.

    Source TIFs with differing CRS or resolution are reprojected on-the-fly
    using WarpedVRT — no temporary files are written to disk.

    Args:
        source: A list of TIF file paths, or a directory whose *.tif/*.tiff
            files are collected automatically.
        output_path: Destination path for the merged GeoTIFF.
        target_crs: EPSG string or WKT for the output CRS. Defaults to the
            first TIF's CRS when None.
        target_resolution: (x_res, y_res) in target CRS units. Defaults to
            the first TIF's native resolution when None.
        resampling: Rasterio resampling algorithm name (e.g. "bilinear",
            "nearest", "cubic"). Defaults to "bilinear".
        nodata: Output nodata value. Defaults to the first TIF's nodata when
            None; proceeds without nodata if the first TIF has none.

    Returns:
        The resolved output_path as a Path.

    Raises:
        FileNotFoundError: If source directory or any listed file does not exist.
        ValueError: If source directory contains no TIFs, or resampling name
            is not a valid rasterio resampling algorithm.
    """
    if resampling not in _VALID_RESAMPLING_NAMES:
        raise ValueError(
            f"invalid resampling {resampling!r}; "
            f"valid names: {sorted(_VALID_RESAMPLING_NAMES)}"
        )

    tif_paths = _collect_tif_paths(source)
    resampling_enum = Resampling[resampling]

    with rasterio.open(tif_paths[0]) as ref:
        resolved_crs = CRS.from_user_input(target_crs) if target_crs is not None else ref.crs
        resolved_res = target_resolution if target_resolution is not None else (
            abs(ref.transform.a),
            abs(ref.transform.e),
        )
        resolved_nodata = nodata if nodata is not None else ref.nodata

    sources = [rasterio.open(p) for p in tif_paths]
    vrts: list[WarpedVRT] = []
    try:
        vrts = [
            WarpedVRT(
                src,
                crs=resolved_crs,
                resampling=resampling_enum,
                target_aligned_pixels=True,
                res=resolved_res,
                nodata=resolved_nodata,
            )
            for src in sources
        ]
        merged_array, merged_transform = _rasterio_merge(vrts, nodata=resolved_nodata)
    finally:
        for vrt in vrts:
            vrt.close()
        for src in sources:
            src.close()

    out_path = Path(output_path)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=merged_array.shape[1],
        width=merged_array.shape[2],
        count=merged_array.shape[0],
        dtype=merged_array.dtype,
        crs=resolved_crs,
        transform=merged_transform,
        nodata=resolved_nodata,
    ) as dst:
        dst.write(merged_array)

    return out_path


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
