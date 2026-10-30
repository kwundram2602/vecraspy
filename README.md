# vecraspy

A toolbox of **vec**tor, **ras**ter and terrain utilities
for **Py**thon geospatial work.

vecraspy collects functions that tend to get re-written in every remote
sensing / DEM project — clipping and aligning GeoTIFFs, deriving slope, aspect
and hillshade, building trajectories from point layers, quick-look plotting —
behind one flat, consistently-shaped API. 

It is built on `rasterio`, `geopandas`/`shapely`, `numpy`, `matplotlib` and
`whitebox-workflows`. Python 3.13+.

## Installation

```bash
uv add git+<repo-url>
# or, for local development
uv sync --all-extras
```
## Layout

```
src/vecraspy/
├── raster.py    GeoTIFF I/O, clipping, alignment, reprojection, sampling
├── vector.py    Point layers, trajectories, AOI construction and filtering
├── convert.py   AOI ↔ WKT / GeoJSON / bounding box
├── terrain.py   Slope, aspect, hillshade (whitebox-workflows backed)
├── plot.py      Matplotlib quick-looks with sensible defaults per product
└── sr.py        DTM super-resolution (guided filter + thermal erosion)
scripts/         Packaging helpers exposed as console scripts
tests/           pytest suite, one module per source module
```

## What's supported

### Raster (`vecraspy.raster`)

| Function | Purpose |
| --- | --- |
| `merge_tifs` | Stitch many GeoTIFFs into one, optionally reprojecting/resampling |
| `clip_tif_by_aoi` | Clip to an AOI geometry, with optional cropping and nodata |
| `align_raster_grid` | Snap a raster onto a reference raster's exact grid |
| `reproject_raster` | Reproject to a target CRS, optionally at a new resolution |
| `scale_raster_to_gsd` | Resample to a target ground sample distance |
| `aggregate_raster` | Coarsen a `wbw.Raster` by an integer factor (mean/max/…) |
| `same_nodata_mask` | Transfer a reference raster's nodata mask onto a target |
| `tif_bounds_as_polygon` | Bounding box, or the exact valid-data footprint |
| `filter_tifs_by_aoi` | Keep only the tiles that intersect an AOI |
| `ndvi` | Normalised Difference Vegetation Index from red/NIR arrays |
| `extract_raster_values_at_points` | Sample rasters at point locations |
| `summarize_extracted_raster_values` | Per-band summary statistics of those samples |

### Vector (`vecraspy.vector`)

| Function | Purpose |
| --- | --- |
| `read_points` | Read a point layer from any OGR-supported format |
| `group_by_id` | Split a `GeoDataFrame` into groups by an id column |
| `build_trajectory` / `build_trajectories` | Turn ordered points into `Trajectory` objects |
| `write_trajectories` | Write each trajectory to its own GeoPackage |
| `compute_aoi` | Convex or concave hull covering a fraction of the points |
| `enlarge_aoi` | Scale an AOI rectangle around its centre |
| `get_bounds_as_gdf` | Bounding box of a `GeoDataFrame` as a single-row frame |
| `filter_by_aoi` | Spatial filter by `intersects` or `within` |
| `get_aoi_center` | AOI centre as `(lat, lon)` in EPSG:4326 |
| `get_stac_spatial_extent` | AOI as a STAC-ready bbox list |

`Trajectory` is a lightweight dataclass holding an ordered sequence of points
plus its id — the common shape for GPS tracks and flight lines.

### Conversion (`vecraspy.convert`)

AOIs can be a `GeoDataFrame`, `GeoSeries`, shapely geometry or a vector file path;
bounding boxes are `(minx, miny, maxx, maxy)`, given as four numbers or a string
such as `"7.1,51.9,7.3,52.0"`.

| Function | Purpose |
| --- | --- |
| `aoi_to_wkt` | Union of the AOI as WKT, optionally reprojected and rounded |
| `aoi_to_bbox` | AOI bounding box, optionally in another CRS |
| `aoi_to_geojson` | Union of the AOI as a GeoJSON geometry (EPSG:4326 by default) |
| `wkt_to_aoi` | WKT geometry → single-row `GeoDataFrame` |
| `geojson_to_aoi` | GeoJSON geometry, Feature or FeatureCollection → `GeoDataFrame` |
| `bbox_to_aoi` | Bounding box → rectangle `GeoDataFrame` |
| `wkt_to_bbox` / `bbox_to_wkt` | Direct string conversions, no CRS involved |

### Terrain (`vecraspy.terrain`)

`slope` (degrees, radians or percent), `aspect`, `hillshade_path` — all file-to-file via
whitebox-workflows — plus `hillshade`, which shades a bare numpy elevation array
with configurable azimuth, altitude, z-factor and pixel spacing.

### Plotting (`vecraspy.plot`)

`plot_slope`, `plot_aspect`, `plot_dsm` and `plot_hillshade` return
`(Figure, Axes)` so they drop into existing subplot grids via `ax=`. Each
carries defaults suited to its product: unit-aware fixed ranges for slope, a
circular colourmap for aspect, percentile contrast stretching for DSMs.

### DTM super-resolution (`vecraspy.sr`)

Upsample a DTM/DEM to the resolution of a co-registered optical image, using the
optical image as a structural guide (guided image filter), with an optional
thermal-erosion detail-polish pass.

```python
from vecraspy import super_resolve_dtm

super_resolve_dtm(
    "dtm_30m.tif",
    "optical_2m5.tif",
    "dtm_superres.tif",
    apply_erosion=True,
    erosion_kwargs={"iterations": 30},
)
```

## Example

```python
from vecraspy import clip_tif_by_aoi, compute_aoi, plot_slope, read_points, slope

points = read_points("survey.gpkg")
aoi = compute_aoi(points, coverage=0.95, method="concave")

clip_tif_by_aoi("dem.tif", "dem_aoi.tif", aoi)
slope("dem_aoi.tif", "slope.tif", units="degrees")
plot_slope("slope.tif", title="Slope over survey area")
```
## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format .
uv run ty check
```
