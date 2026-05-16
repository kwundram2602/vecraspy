"""Raster utility functions."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import whitebox_workflows as wbw
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask as _rasterio_mask
from rasterio.merge import merge as _rasterio_merge
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds
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
    if not source_path.is_dir():
        raise ValueError(f"expected a directory, got a file: {source_path}")

    paths = sorted(
        p for p in source_path.iterdir() if p.suffix.lower() in {".tif", ".tiff"}
    )
    if not paths:
        raise ValueError(f"no TIF files found in directory: {source_path}")

    return paths


_VALID_RESAMPLING_NAMES: frozenset[str] = frozenset(r.name for r in Resampling)
_VALID_AGGREGATION_TYPES: frozenset[str] = frozenset(
    {"mean", "sum", "maximum", "minimum", "range"}
)

_wbe = wbw.WbEnvironment()


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
        resolved_crs = (
            CRS.from_user_input(target_crs) if target_crs is not None else ref.crs
        )
        resolved_res = (
            target_resolution
            if target_resolution is not None
            else (
                abs(ref.transform.a),
                abs(ref.transform.e),
            )
        )
        resolved_nodata = nodata if nodata is not None else ref.nodata

    sources = [rasterio.open(p) for p in tif_paths]
    vrts: list[WarpedVRT] = []
    try:
        for src in sources:
            vrts.append(
                WarpedVRT(
                    src,
                    crs=resolved_crs,
                    resampling=resampling_enum,
                    nodata=resolved_nodata,
                )
            )
        merged_array, merged_transform = _rasterio_merge(
            vrts,
            res=resolved_res,
            nodata=resolved_nodata,
        )
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


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute the Normalised Difference Vegetation Index.

    Formula: (NIR - Red) / (NIR + Red). Pixels where NIR + Red == 0 are set
    to 0 to avoid division by zero.

    Args:
        red: 2-D float array for the red band.
        nir: 2-D float array for the near-infrared band.

    Returns:
        2-D float32 array in [-1, 1].
    """
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    denom = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denom == 0, 0.0, (nir - red) / denom)
    return result.astype(np.float32)


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
            west, south, east, north = transform_bounds(
                src.crs, target_crs, *src.bounds
            )
        tif_bbox = box(west, south, east, north)
        if aoi_geom.intersects(tif_bbox):
            matches.append(path)

    return matches


def aggregate_raster(
    raster: wbw.Raster,
    aggregation_factor: int = 2,
    aggregation_type: str = "mean",
) -> wbw.Raster:
    """Aggregate a raster to a coarser grid using whitebox-workflows.

    Args:
        raster: Input raster to aggregate.
        aggregation_factor: Factor by which rows/columns are reduced.
        aggregation_type: One of "mean", "sum", "maximum", "minimum", "range".

    Returns:
        Aggregated raster.

    Raises:
        ValueError: If aggregation_factor is < 1 or aggregation_type is invalid.
    """
    if aggregation_factor < 1:
        raise ValueError(
            f"aggregation_factor must be >= 1, got {aggregation_factor}"
        )
    if aggregation_type not in _VALID_AGGREGATION_TYPES:
        raise ValueError(
            f"invalid aggregation_type {aggregation_type!r}; "
            f"valid values: {sorted(_VALID_AGGREGATION_TYPES)}"
        )

    return _wbe.aggregate_raster(
        raster,
        aggregation_factor=aggregation_factor,
        aggregation_type=aggregation_type,
    )


def scale_raster_to_gsd(
    input_path: Path | str,
    output_path: Path | str,
    target_gsd: float,
    *,
    resampling: str = "bilinear",
) -> Path:
    """Resample a raster to a target ground sample distance (GSD).

    Args:
        input_path: Path to the source raster.
        output_path: Path for the resampled raster.
        target_gsd: Desired pixel size in CRS units (e.g. meters).
        resampling: Rasterio resampling algorithm name (e.g. "bilinear",
            "nearest", "cubic"). Defaults to "bilinear".

    Returns:
        The resolved output_path as a Path.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If target_gsd <= 0 or resampling is invalid.
    """
    if target_gsd <= 0:
        raise ValueError(f"target_gsd must be > 0, got {target_gsd}")
    if resampling not in _VALID_RESAMPLING_NAMES:
        raise ValueError(
            f"invalid resampling {resampling!r}; "
            f"valid names: {sorted(_VALID_RESAMPLING_NAMES)}"
        )

    src_path = Path(input_path)
    if not src_path.exists():
        raise FileNotFoundError(f"file not found: {src_path}")

    resampling_enum = Resampling[resampling]
    out_path = Path(output_path)

    with rasterio.open(src_path) as src:
        src_gsd_x = src.transform.a
        src_gsd_y = abs(src.transform.e)

        scale_x = src_gsd_x / target_gsd
        scale_y = src_gsd_y / target_gsd

        new_width = max(1, round(src.width * scale_x))
        new_height = max(1, round(src.height * scale_y))

        new_transform = from_bounds(
            *src.bounds,
            width=new_width,
            height=new_height,
        )

        profile = src.profile.copy()
        profile.update(
            width=new_width,
            height=new_height,
            transform=new_transform,
        )

        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=resampling_enum,
        )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    return out_path


def clip_tif_by_aoi(
    input_path: Path | str,
    output_path: Path | str,
    aoi: BaseGeometry | gpd.GeoDataFrame | gpd.GeoSeries,
    *,
    aoi_crs: str = "EPSG:4326",
    nodata: float | None = None,
    crop: bool = True,
) -> Path:
    """Clip a GeoTIFF to an AOI and write the result to disk.

    The AOI is reprojected to the raster's CRS on-the-fly — no intermediate
    files are written. For GeoDataFrame and GeoSeries inputs the embedded CRS
    is used; for plain Shapely geometries ``aoi_crs`` applies.

    Args:
        input_path: Path to the source GeoTIFF.
        output_path: Path for the clipped output GeoTIFF.
        aoi: Area of interest as a Shapely geometry, GeoDataFrame, or
            GeoSeries. Multi-geometry inputs are dissolved to a single
            geometry via union_all().
        aoi_crs: CRS for plain Shapely geometry inputs (ignored when aoi
            is a GeoDataFrame or GeoSeries). Defaults to WGS-84.
        nodata: Nodata value for masked pixels. Falls back to the source
            raster's nodata, then 0 if neither is set.
        crop: If True (default), the output extent is cropped to the AOI
            bounding box. If False, the full raster extent is kept and only
            pixels outside the AOI are masked.

    Returns:
        The resolved output_path as a Path.

    Raises:
        FileNotFoundError: If input_path does not exist.
    """
    src_path = Path(input_path)
    if not src_path.exists():
        raise FileNotFoundError(f"file not found: {src_path}")

    if isinstance(aoi, gpd.GeoDataFrame):
        aoi_series = aoi.geometry
    elif isinstance(aoi, gpd.GeoSeries):
        aoi_series = aoi
    else:
        aoi_series = gpd.GeoSeries([aoi], crs=aoi_crs)

    with rasterio.open(src_path) as src:
        aoi_reprojected = aoi_series.to_crs(src.crs)
        aoi_geom = aoi_reprojected.union_all()

        resolved_nodata = nodata if nodata is not None else (src.nodata or 0)

        clipped, clipped_transform = _rasterio_mask(
            src,
            shapes=[aoi_geom],
            crop=crop,
            nodata=resolved_nodata,
        )

        profile = src.profile.copy()
        profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=clipped_transform,
            nodata=resolved_nodata,
        )

    out_path = Path(output_path)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(clipped)

    return out_path


def align_raster_grid(
    reference_path: Path | str,
    target_path: Path | str,
    output_path: Path | str,
    *,
    resampling: str = "bilinear",
) -> Path:
    """Align a raster to the grid of a reference raster.

    Reprojects, resamples, and clips the target raster to exactly match
    the CRS, pixel size, and spatial extent of the reference raster.
    Steps that are already identical are skipped implicitly by rasterio.

    Args:
        reference_path: Path to the reference GeoTIFF defining the target grid.
        target_path: Path to the GeoTIFF to align.
        output_path: Path for the aligned output GeoTIFF.
        resampling: Rasterio resampling algorithm name (e.g. "bilinear",
            "nearest", "cubic"). Defaults to "bilinear".

    Returns:
        The resolved output_path as a Path.

    Raises:
        FileNotFoundError: If reference_path or target_path does not exist.
        ValueError: If resampling is not a valid rasterio resampling algorithm.
    """
    if resampling not in _VALID_RESAMPLING_NAMES:
        raise ValueError(
            f"invalid resampling {resampling!r}; "
            f"valid names: {sorted(_VALID_RESAMPLING_NAMES)}"
        )

    ref_path = Path(reference_path)
    src_path = Path(target_path)

    if not ref_path.exists():
        raise FileNotFoundError(f"file not found: {ref_path}")
    if not src_path.exists():
        raise FileNotFoundError(f"file not found: {src_path}")

    resampling_enum = Resampling[resampling]
    out_path = Path(output_path)

    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

    with rasterio.open(src_path) as src:
        destination = np.zeros(
            (src.count, ref_height, ref_width), dtype=src.dtypes[0]
        )

        for band_idx in src.indexes:
            reproject(
                source=rasterio.band(src, band_idx),
                destination=destination[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling_enum,
                src_nodata=src.nodata,
                dst_nodata=src.nodata,
            )

        profile = src.profile.copy()
        profile.update(
            crs=ref_crs,
            transform=ref_transform,
            width=ref_width,
            height=ref_height,
        )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(destination)

    return out_path


def same_nodata_mask(
    reference_path: Path | str,
    target_path: Path | str,
    output_path: Path | str,
    *,
    nodata: float | None = None,
) -> Path:
    """Apply the nodata mask of a reference raster to a target raster.

    Every pixel that is nodata in any band of the reference is set to the
    nodata value in the output. All other pixels are copied from the target
    unchanged. Both rasters must share the same grid (height, width).

    Args:
        reference_path: Path to the reference GeoTIFF whose nodata mask is used.
        target_path: Path to the GeoTIFF to apply the mask to.
        output_path: Path for the masked output GeoTIFF.
        nodata: Nodata value written to masked pixels. Falls back to the target
            raster's nodata when None. Raises if neither is set.

    Returns:
        The resolved output_path as a Path.

    Raises:
        FileNotFoundError: If reference_path or target_path does not exist.
        ValueError: If the rasters have different shapes, or if no nodata value
            can be determined.
    """
    ref_path = Path(reference_path)
    src_path = Path(target_path)

    if not ref_path.exists():
        raise FileNotFoundError(f"file not found: {ref_path}")
    if not src_path.exists():
        raise FileNotFoundError(f"file not found: {src_path}")

    with rasterio.open(ref_path) as ref:
        # read_masks returns 0 where nodata, 255 where valid
        ref_nodata_mask = np.any(ref.read_masks() == 0, axis=0)

    with rasterio.open(src_path) as src:
        resolved_nodata = nodata if nodata is not None else src.nodata
        if resolved_nodata is None:
            raise ValueError(
                "no nodata value available; set one on the target raster or pass nodata="
            )
        if (src.height, src.width) != ref_nodata_mask.shape:
            raise ValueError(
                f"raster shapes do not match: reference is {ref_nodata_mask.shape}, "
                f"target is ({src.height}, {src.width})"
            )

        data = src.read()
        data[:, ref_nodata_mask] = resolved_nodata

        profile = src.profile.copy()
        profile.update(nodata=resolved_nodata)

    out_path = Path(output_path)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    return out_path


def extract_raster_values_at_points(
    rasters: list[wbw.Raster],
    points: wbw.Vector,
) -> tuple[wbw.Vector, str]:
    """Extract raster values at point locations using whitebox-workflows.

    Args:
        rasters: Input rasters to sample.
        points: Point vector to sample at.

    Returns:
        A tuple of (updated points vector, text report).

    Raises:
        ValueError: If rasters is empty.
    """
    if not rasters:
        raise ValueError("rasters must contain at least one Raster")

    return _wbe.extract_raster_values_at_points(rasters, points)


def summarize_extracted_raster_values(
    points: wbw.Vector | gpd.GeoDataFrame | tuple[wbw.Vector, str],
    *,
    value_fields: list[str] | None = None,
    value_prefix: str = "VALUE",
) -> dict[str, dict[str, float]]:
    """Summarize sampled raster values from extract_raster_values_at_points.

    Args:
        points: A WbW Vector, GeoDataFrame, or the (Vector, report) tuple
            returned by extract_raster_values_at_points.
        value_fields: Optional list of field names to summarize. If None,
            fields starting with value_prefix are used.
        value_prefix: Field-name prefix used for extracted values.

    Returns:
        Mapping of field name to summary stats.

    Raises:
        TypeError: If points cannot be converted to a GeoDataFrame.
        ValueError: If no value fields are found or values are non-numeric.
    """
    if isinstance(points, tuple):
        points = points[0]

    if isinstance(points, gpd.GeoDataFrame):
        gdf = points
    else:
        converter = None
        for name in ("to_geopandas", "to_geo_dataframe", "to_gdf"):
            if hasattr(points, name):
                converter = getattr(points, name)
                break
        if converter is None:
            raise TypeError(
                "points must be a GeoDataFrame or a WbW Vector with a GeoDataFrame converter"
            )
        gdf = converter()

    if value_fields is None:
        value_fields = [
            col for col in gdf.columns if str(col).startswith(value_prefix)
        ]
    if not value_fields:
        raise ValueError("no value fields found to summarize")

    stats: dict[str, dict[str, float]] = {}
    for field in value_fields:
        values = gdf[field].to_numpy()
        try:
            numeric = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"field {field!r} contains non-numeric values") from exc

        numeric = numeric[np.isfinite(numeric)]
        if numeric.size == 0:
            raise ValueError(f"field {field!r} has no finite values")

        stats[field] = {
            "count": float(numeric.size),
            "min": float(np.min(numeric)),
            "max": float(np.max(numeric)),
            "mean": float(np.mean(numeric)),
            "p25": float(np.percentile(numeric, 25)),
            "median": float(np.percentile(numeric, 50)),
            "p75": float(np.percentile(numeric, 75)),
        }

    return stats
