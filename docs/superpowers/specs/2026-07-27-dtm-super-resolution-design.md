# DTM Super-Resolution Pipeline Design

**Date:** 2026-07-27
**Status:** Approved

## Summary

Add a new module `src/vecraspy/sr.py` that increases the resolution of a DTM/DEM
GeoTIFF using a co-registered high-resolution optical image as a structural guide
(e.g. 30 m DTM + 2.5 m optical). Primary use case is visualization/cartography
(hillshades, relief maps), not hydrologically exact analysis.

Three strategies were evaluated and rejected or scoped down:

- **Depth Anything V2** — rejected. It estimates depth from a single image rather than
  refining an existing real elevation measurement; conceptually the wrong tool.
- **Real-ESRGAN applied directly to the DEM** — rejected as the core method. A GAN
  trained on natural photos hallucinates plausible-looking but physically arbitrary
  texture on elevation fields. Possibly useful later for pure visual stylization, not
  as the primary mechanism.
- **Erosion simulation** — adopted, but as an optional post-processing refinement step
  rather than the main upsampling mechanism.

The chosen core method is **Guided Image Filtering** (He et al.) using the optical
image as an edge/structure reference, with an optional **thermal erosion** simulation
as a detail-polish step. Both are implemented in pure `numpy` — **no new dependencies**
are required (no torch, no opencv, no scipy).

---

## Architecture & Data Flow

The alignment stage requires no new code — it is assembled from existing
`vecraspy.raster` functions:

```
DEM (e.g. 30 m) + optical (e.g. 2.5 m)
  → tif_bounds_as_polygon(dem)                 # determine DEM footprint
  → clip_tif_by_aoi(optical, footprint)        # crop optical to DEM AOI
  → align_raster_grid(reference=optical_clip,  # warp DEM onto optical's grid
                       target=dem)                (bicubic baseline)
  → guided_upsample_dem(...)                   # NEW: guided filter, optical as guide
  → same_nodata_mask(...)                      # restore original nodata mask
  → [optional] simulate_thermal_erosion(...)   # NEW: procedural micro-relief
```

Only the guided filter and the erosion simulation are new algorithms; everything
before and after reuses existing `vecraspy.raster` functions
(`tif_bounds_as_polygon`, `clip_tif_by_aoi`, `align_raster_grid`, `same_nodata_mask`).

---

## Interface

### `guided_upsample_dem` — core function

```python
def guided_upsample_dem(
    dem_path: Path | str,
    optical_path: Path | str,
    output_path: Path | str,
    *,
    band: int | None = None,
    radius: int = 8,
    eps: float = 1e-2,
    resampling: str = "cubic",
) -> Path:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dem_path` | `Path \| str` | — | Path to the source DEM GeoTIFF |
| `optical_path` | `Path \| str` | — | Path to the co-registered high-resolution optical GeoTIFF |
| `output_path` | `Path \| str` | — | Destination path for the super-resolved DEM |
| `band` | `int \| None` | `None` | 1-indexed optical band to use as guide; `None` averages all bands into a luminance guide |
| `radius` | `int` | `8` | Guided filter window radius, in pixels of the optical grid |
| `eps` | `float` | `1e-2` | Regularization term; the optical luminance guide is normalized to `[0, 1]` internally so this default matches the guided-filter literature |
| `resampling` | `str` | `"cubic"` | Rasterio resampling name used for the bicubic baseline warp |

**Returns:** the resolved `output_path` as a `Path`.

**Internal Pipeline:**

1. **Validate** — `dem_path`/`optical_path` exist, `radius > 0`, `eps > 0`, `band`
   within the optical raster's band count (if given), `resampling` in
   `_VALID_RESAMPLING_NAMES`.
2. **Footprint** — `tif_bounds_as_polygon(dem_path)`.
3. **Clip optical** — `clip_tif_by_aoi(optical_path, footprint)` to avoid warping the
   full optical extent when it is much larger than the DEM.
4. **Bicubic baseline** — `align_raster_grid(reference=optical_clip, target=dem_path,
   resampling=resampling)`, producing the DEM warped onto the optical grid (`p` in
   the guided filter).
5. **Guide image** — derive `I` from the optical raster: the selected `band`, or the
   mean across all bands when `band is None`; normalize to `[0, 1]`.
6. **Nodata fill** — temporarily fill nodata pixels in the upsampled DEM (mean of
   valid pixels) so box filters near the raster edge/nodata boundary are not
   contaminated by nodata values. Known limitation: pixels very close to a nodata
   boundary may be slightly smoothed; documented in the function's docstring.
7. **Guided filter** — compute box filters via a numpy cumulative-sum integral image
   (no scipy needed):
   `mean_I, mean_p, corr_I, corr_Ip` (window `radius`) →
   `var_I = corr_I - mean_I**2`, `cov_Ip = corr_Ip - mean_I * mean_p` →
   `a = cov_Ip / (var_I + eps)`, `b = mean_p - a * mean_I` →
   `mean_a, mean_b` (box filter of `a`, `b`) → `q = mean_a * I + mean_b`.
8. **Restore nodata** — reapply the nodata mask captured in step 4.
9. **Write output** — GeoTIFF with the profile (CRS, transform, shape) from step 4.

### `simulate_thermal_erosion` — optional detail-polish step

```python
def simulate_thermal_erosion(
    dem_path: Path | str,
    output_path: Path | str,
    *,
    iterations: int = 50,
    talus_angle: float = 35.0,
    transfer_rate: float = 0.5,
) -> Path:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dem_path` | `Path \| str` | — | Path to the source DEM GeoTIFF |
| `output_path` | `Path \| str` | — | Destination path for the eroded DEM |
| `iterations` | `int` | `50` | Number of simulation steps |
| `talus_angle` | `float` | `35.0` | Talus (repose) angle in degrees; cells whose slope to their steepest of 8 neighbours exceeds this angle shed material |
| `transfer_rate` | `float` | `0.5` | Fraction (0, 1] of the excess above the talus-stable level moved per iteration |

**Returns:** the resolved `output_path` as a `Path`.

Fully vectorized over numpy array operations per iteration (no per-cell/per-particle
Python loop): compute the 8-neighbour slope, move material to the steepest neighbour
exceeding `talus_angle`, repeat `iterations` times. Operates purely on DEM values,
independent of optical imagery. Nodata cells are excluded from material transport and
the nodata mask is restored at the end.

### `super_resolve_dtm` — convenience function

```python
def super_resolve_dtm(
    dem_path: Path | str,
    optical_path: Path | str,
    output_path: Path | str,
    *,
    band: int | None = None,
    radius: int = 8,
    eps: float = 1e-2,
    apply_erosion: bool = False,
    erosion_kwargs: dict | None = None,
) -> Path:
```

Chains `guided_upsample_dem` and, when `apply_erosion=True`,
`simulate_thermal_erosion` (parameters via `erosion_kwargs`) — same convenience
pattern as `build_trajectories` in `vector.py`, which chains
`read_points → group_by_id → build_trajectory`.

---

## Error Handling

| Situation | Behaviour |
|---|---|
| `dem_path` / `optical_path` does not exist | `FileNotFoundError` |
| `radius <= 0` | `ValueError` |
| `eps <= 0` | `ValueError` |
| `band` outside the optical raster's band count | `ValueError` |
| `resampling` not in `_VALID_RESAMPLING_NAMES` | `ValueError` |
| `iterations <= 0` | `ValueError` |
| `talus_angle` outside `(0, 90)` | `ValueError` |
| `transfer_rate` outside `(0, 1]` | `ValueError` |

---

## Testing

`tests/test_sr.py`, following the local `_write_tif` fixture pattern used in
`tests/test_raster.py` (synthetic small GeoTIFFs via `rasterio` + `from_bounds`,
`tmp_path` fixture, `pytest.raises(..., match=...)`):

- Output grid (shape, transform, CRS) matches the optical grid.
- Filtered values stay close to the bicubic baseline (no value blow-up beyond a small
  tolerance).
- A synthetic edge in the guide image produces a measurable step in the output at the
  same location (core guided-filter behaviour).
- Nodata mask survives the full pipeline.
- `simulate_thermal_erosion` reduces local extremes at artificially steep cells over
  several iterations; flat terrain is unchanged.
- Parameter validation: every error case in the table above.
- `super_resolve_dtm` chains both steps correctly, with and without
  `apply_erosion=True`.

---

## File Location

- Implementation: `src/vecraspy/sr.py` (currently empty)
- Tests: `tests/test_sr.py` (new)
- Export from: `src/vecraspy/__init__.py` (`guided_upsample_dem`,
  `simulate_thermal_erosion`, `super_resolve_dtm`, alphabetized into `__all__` and
  the import block)
- Reused, unchanged: `src/vecraspy/raster.py` (`tif_bounds_as_polygon`,
  `clip_tif_by_aoi`, `align_raster_grid`, `same_nodata_mask`,
  `_VALID_RESAMPLING_NAMES`)
