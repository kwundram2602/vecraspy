# merge_tifs Design

**Date:** 2026-04-30
**Status:** Approved

## Summary

Add a `merge_tifs` function to `src/vecraspy/raster.py` that stitches multiple GeoTIFFs into a single output GeoTIFF. Source TIFs with differing CRS or resolution are reprojected and resampled on-the-fly using `rasterio.vrt.WarpedVRT` before merging — no temporary files are written to disk.

---

## Interface

```python
def merge_tifs(
    source: list[Path | str] | Path | str,
    output_path: Path | str,
    *,
    target_crs: str | None = None,
    target_resolution: tuple[float, float] | None = None,
    resampling: str = "bilinear",
    nodata: float | None = None,
) -> Path:
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | `list[Path \| str] \| Path \| str` | — | Explicit list of TIF paths **or** a directory; all `*.tif`/`*.tiff` files inside the directory are used |
| `output_path` | `Path \| str` | — | Destination path for the merged GeoTIFF |
| `target_crs` | `str \| None` | `None` | EPSG string or WKT; defaults to first TIF's CRS |
| `target_resolution` | `tuple[float, float] \| None` | `None` | `(x_res, y_res)` in target CRS units; defaults to first TIF's resolution |
| `resampling` | `str` | `"bilinear"` | Any rasterio resampling name (`"nearest"`, `"cubic"`, etc.) |
| `nodata` | `float \| None` | `None` | Output nodata value; defaults to first TIF's nodata |

### Returns

The resolved `output_path` as a `Path`.

---

## Internal Pipeline

1. **Collect paths** — if `source` is a directory, glob `*.tif` + `*.tiff` sorted; if it's a list, convert each to `Path`. Raise `ValueError` if the resolved list is empty.
2. **Read reference metadata** — open the first TIF to read fallback CRS, resolution, and nodata. Apply caller overrides for any that were provided.
3. **Wrap in `WarpedVRT`** — open all source files with `rasterio.open` and wrap each in `rasterio.vrt.WarpedVRT` configured with the target CRS, resolution, resampling method, and nodata. Virtual — no data is read yet, no temp files written.
4. **Merge** — pass the list of `WarpedVRT` objects to `rasterio.merge.merge`, which returns a numpy array and an affine transform covering the full mosaic extent.
5. **Write output** — open `output_path` for writing as a GeoTIFF using the merged array shape, target CRS, mosaic transform, and resolved nodata.
6. **Cleanup** — all datasets closed via context managers.

---

## Error Handling

| Situation | Behaviour |
|---|---|
| `source` directory does not exist | `FileNotFoundError` |
| `source` directory contains no TIFs | `ValueError` with directory path |
| A file in the list does not exist | `FileNotFoundError` with offending path |
| `resampling` is not a valid rasterio name | `ValueError` with bad value and list of valid names |
| `output_path` parent does not exist | propagate rasterio's `FileNotFoundError` |
| First TIF has `nodata=None` and caller omits `nodata` | proceed without nodata — no error |

---

## File Location

- Implementation: `src/vecraspy/raster.py`
- Export from: `src/vecraspy/__init__.py`
- No new files required.
