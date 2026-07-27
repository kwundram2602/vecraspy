# DTM Super-Resolution Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `src/vecraspy/sr.py` with `guided_upsample_dem`, `simulate_thermal_erosion`, and `super_resolve_dtm` — a DTM super-resolution pipeline that uses a co-registered optical image as a structural guide, with an optional thermal-erosion detail-polish step.

**Architecture:** DEM alignment (clip optical to DEM footprint, warp DEM onto the optical grid) reuses existing `vecraspy.raster` functions unchanged. Two new numpy-only algorithms are added: a guided image filter (He et al.) using an integral-image box filter, and a vectorized talus-angle thermal erosion cellular automaton. `super_resolve_dtm` chains the two, mirroring the `build_trajectories` convenience-function pattern in `vector.py`.

**Tech Stack:** Python 3.13, numpy, rasterio, pytest — no new dependencies.

## Global Constraints

- No new dependencies (spec requirement: pure numpy/rasterio, matching `pyproject.toml`'s existing `dependencies` list).
- Follow `raster.py` conventions: `Path` args accept `Path | str`, `FileNotFoundError` for missing files, `ValueError` for invalid parameters with the exact message patterns used elsewhere (e.g. `f"file not found: {path}"`, `f"invalid resampling {resampling!r}; valid names: {sorted(_VALID_RESAMPLING_NAMES)}"`).
- Docstrings use the Args/Returns/Raises style seen throughout `raster.py` and `terrain.py`.
- Tests use the local `_write_tif`/`_write_multiband_tif` fixture pattern from `tests/test_raster.py` (no shared `conftest.py` in this repo), `tmp_path` fixture, `pytest.raises(..., match=...)`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/vecraspy/sr.py` | Modify (currently empty) | `_box_filter`, `_guided_filter`, `guided_upsample_dem`, `_shift`, `_NEIGHBOR_OFFSETS`, `simulate_thermal_erosion`, `super_resolve_dtm` |
| `tests/test_sr.py` | Create | All tests for the above |
| `src/vecraspy/__init__.py` | Modify | Export `guided_upsample_dem`, `simulate_thermal_erosion`, `super_resolve_dtm` |

---

### Task 1: Test file with fixture helpers

**Files:**
- Create: `tests/test_sr.py`

**Interfaces:**
- Produces: `_write_tif(path, data, *, crs="EPSG:32632", west=0.0, south=0.0, east=4.0, north=4.0, nodata=None) -> None` (single-band), `_write_multiband_tif(path, data, *, crs="EPSG:32632", west=0.0, south=0.0, east=4.0, north=4.0, nodata=None) -> None` (multi-band, `data.shape == (count, height, width)`)

- [ ] **Step 1: Create the test file with both fixture helpers**

```python
"""Tests for vecraspy.sr."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds


def _write_tif(
    path: Path,
    data: np.ndarray,
    *,
    crs: str = "EPSG:32632",
    west: float = 0.0,
    south: float = 0.0,
    east: float = 4.0,
    north: float = 4.0,
    nodata: float | None = None,
) -> None:
    height, width = data.shape
    transform = from_bounds(west, south, east, north, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=CRS.from_user_input(crs),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def _write_multiband_tif(
    path: Path,
    data: np.ndarray,
    *,
    crs: str = "EPSG:32632",
    west: float = 0.0,
    south: float = 0.0,
    east: float = 4.0,
    north: float = 4.0,
    nodata: float | None = None,
) -> None:
    count, height, width = data.shape
    transform = from_bounds(west, south, east, north, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=CRS.from_user_input(crs),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)
```

- [ ] **Step 2: Verify the file parses cleanly**

Run: `uv run python -c "import tests.test_sr"`
Expected: no output (no import errors)

- [ ] **Step 3: Commit**

```bash
git add tests/test_sr.py
git commit -m "test: add fixture helpers for vecraspy.sr tests"
```

---

### Task 2: `_box_filter` integral-image helper

**Files:**
- Modify: `tests/test_sr.py` — add box filter tests
- Modify: `src/vecraspy/sr.py` — create module with `_box_filter`

**Interfaces:**
- Produces: `_box_filter(arr: np.ndarray, radius: int) -> np.ndarray` — local mean over a `(2*radius+1)` square window, edge-padded, same shape as `arr`.

- [ ] **Step 1: Write the failing tests**

```python
from vecraspy.sr import _box_filter


def test_box_filter_constant_array_unchanged():
    arr = np.full((10, 10), 5.0)
    result = _box_filter(arr, radius=2)
    np.testing.assert_allclose(result, arr)


def test_box_filter_matches_naive_reference():
    rng = np.random.default_rng(42)
    arr = rng.random((10, 10))
    radius = 2
    result = _box_filter(arr, radius)

    padded = np.pad(arr, radius, mode="edge")
    window = 2 * radius + 1
    expected = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            expected[i, j] = padded[i : i + window, j : j + window].mean()

    np.testing.assert_allclose(result, expected, rtol=1e-10)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_sr.py -k "box_filter" -v`
Expected: `ImportError` — `vecraspy.sr` has no `_box_filter` (module is empty)

- [ ] **Step 3: Implement `_box_filter` in `sr.py`**

```python
"""DTM super-resolution: optical-guided filtering and thermal erosion."""

import math
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from vecraspy.raster import align_raster_grid, clip_tif_by_aoi, tif_bounds_as_polygon

_VALID_RESAMPLING_NAMES: frozenset[str] = frozenset(r.name for r in Resampling)


def _box_filter(arr: np.ndarray, radius: int) -> np.ndarray:
    """Local mean of arr over a (2*radius+1) square window (edge-padded)."""
    height, width = arr.shape
    window = 2 * radius + 1
    padded = np.pad(arr, radius, mode="edge")
    cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cumsum = np.pad(cumsum, ((1, 0), (1, 0)), mode="constant")
    total = (
        cumsum[window : window + height, window : window + width]
        - cumsum[0:height, window : window + width]
        - cumsum[window : window + height, 0:width]
        + cumsum[0:height, 0:width]
    )
    return total / (window * window)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_sr.py -k "box_filter" -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/vecraspy/sr.py tests/test_sr.py
git commit -m "feat: add _box_filter integral-image helper"
```

---

### Task 3: `guided_upsample_dem` — core implementation and happy-path tests

**Files:**
- Modify: `tests/test_sr.py` — add guided-upsample tests
- Modify: `src/vecraspy/sr.py` — add `_guided_filter` and `guided_upsample_dem`

**Interfaces:**
- Consumes: `_box_filter(arr, radius) -> np.ndarray` (Task 2); `tif_bounds_as_polygon(path) -> Polygon`, `clip_tif_by_aoi(input_path, output_path, aoi, *, aoi_crs=..., nodata=..., crop=...) -> Path`, `align_raster_grid(reference_path, target_path, output_path, *, resampling=...) -> Path` (all from `vecraspy.raster`, unchanged)
- Produces: `guided_upsample_dem(dem_path, optical_path, output_path, *, band=None, radius=8, eps=1e-2, resampling="cubic") -> Path`

- [ ] **Step 1: Write the failing happy-path tests**

```python
from vecraspy.sr import guided_upsample_dem
from vecraspy.raster import align_raster_grid


def test_guided_upsample_dem_output_matches_optical_grid(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"
    _write_tif(
        dem,
        np.arange(16, dtype=np.float32).reshape(4, 4) + 10.0,
        west=0,
        south=0,
        east=40,
        north=40,
    )
    guide = np.stack([np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)] * 3)
    _write_multiband_tif(optical, guide, west=0, south=0, east=40, north=40)

    out = tmp_path / "out.tif"
    result = guided_upsample_dem(dem, optical, out)

    assert result == out
    with rasterio.open(optical) as opt_src, rasterio.open(out) as out_src:
        assert out_src.width == opt_src.width
        assert out_src.height == opt_src.height
        assert out_src.crs == opt_src.crs


def test_guided_upsample_dem_constant_guide_returns_box_filtered_baseline(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"
    dem_data = np.arange(16, dtype=np.float32).reshape(4, 4) + 10.0
    _write_tif(dem, dem_data, west=0, south=0, east=40, north=40)
    _write_multiband_tif(
        optical, np.ones((1, 16, 16), dtype=np.float32), west=0, south=0, east=40, north=40
    )

    out = tmp_path / "out.tif"
    guided_upsample_dem(dem, optical, out, radius=2)

    with rasterio.open(out) as ds:
        result = ds.read(1)

    with tempfile.TemporaryDirectory() as tmp:
        baseline = Path(tmp) / "baseline.tif"
        align_raster_grid(optical, dem, baseline, resampling="cubic")
        with rasterio.open(baseline) as ds:
            from vecraspy.sr import _box_filter

            # A constant guide carries no structure: var_I and cov_Ip are both
            # exactly zero everywhere, so a=0 and b=mean_p. The guided filter's
            # own final averaging step then box-filters b again (mean_b), so
            # the closed form is a box filter applied TWICE, not once.
            once = _box_filter(ds.read(1).astype(np.float64), 2)
            expected = _box_filter(once, 2)

    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_guided_upsample_dem_sharpens_edge_relative_to_baseline(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"

    dem_data = np.zeros((4, 4), dtype=np.float32)
    dem_data[:, 2:] = 100.0
    _write_tif(dem, dem_data, west=0, south=0, east=40, north=40)

    guide_data = np.zeros((1, 16, 16), dtype=np.float32)
    guide_data[:, :, 8:] = 1.0
    _write_multiband_tif(optical, guide_data, west=0, south=0, east=40, north=40)

    baseline = tmp_path / "baseline.tif"
    align_raster_grid(optical, dem, baseline, resampling="cubic")

    out = tmp_path / "out.tif"
    guided_upsample_dem(dem, optical, out, radius=3, eps=1e-4)

    with rasterio.open(baseline) as ds:
        baseline_row = ds.read(1)[8]
    with rasterio.open(out) as ds:
        guided_row = ds.read(1)[8]

    # The guided filter tracks the guide's edge strongly (small eps), producing
    # one steep jump right at the guide boundary, plus shallow decay on both
    # sides from the window averaging — a wider *span* of non-extreme pixels
    # than the bicubic baseline, but a much steeper *single-step* jump at the
    # edge itself. Steepest single-step gradient is the right sharpness metric.
    def steepest_gradient(row):
        return float(np.max(np.abs(np.diff(row))))

    assert steepest_gradient(guided_row) > steepest_gradient(baseline_row)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_sr.py -k "guided_upsample" -v`
Expected: `ImportError` — `guided_upsample_dem` not defined yet

- [ ] **Step 3: Implement `_guided_filter` and `guided_upsample_dem` in `sr.py`**

Append to `src/vecraspy/sr.py`:

```python
def _guided_filter(
    guide: np.ndarray, source: np.ndarray, radius: int, eps: float
) -> np.ndarray:
    mean_I = _box_filter(guide, radius)
    mean_p = _box_filter(source, radius)
    corr_I = _box_filter(guide * guide, radius)
    corr_Ip = _box_filter(guide * source, radius)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)

    return mean_a * guide + mean_b


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
    """Upsample a DEM to the resolution of a co-registered optical image.

    Bicubic-warps the DEM onto the optical image's grid, then refines the
    result with a guided image filter (He et al.) using the optical image
    as an edge/structure reference. Pixels near a nodata boundary in the
    DEM may be slightly smoothed by the filter before the original nodata
    mask is reapplied.

    Args:
        dem_path: Path to the source DEM GeoTIFF.
        optical_path: Path to the co-registered high-resolution optical GeoTIFF.
        output_path: Path for the super-resolved output GeoTIFF.
        band: 1-indexed optical band to use as the guide. Defaults to the
            mean of all bands when None.
        radius: Guided filter window radius, in pixels of the optical grid.
        eps: Guided filter regularization term; the guide is normalized to
            [0, 1] internally, so the default matches the literature.
        resampling: Rasterio resampling name for the bicubic baseline warp.

    Returns:
        The resolved output_path as a Path.

    Raises:
        FileNotFoundError: If dem_path or optical_path does not exist.
        ValueError: If radius <= 0, eps <= 0, band is out of range, or
            resampling is not a valid rasterio resampling algorithm.
    """
    dem_path = Path(dem_path)
    optical_path = Path(optical_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"file not found: {dem_path}")
    if not optical_path.exists():
        raise FileNotFoundError(f"file not found: {optical_path}")
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if resampling not in _VALID_RESAMPLING_NAMES:
        raise ValueError(
            f"invalid resampling {resampling!r}; "
            f"valid names: {sorted(_VALID_RESAMPLING_NAMES)}"
        )

    with rasterio.open(optical_path) as optical_src:
        band_count = optical_src.count
    if band is not None and not (1 <= band <= band_count):
        raise ValueError(f"band must be in [1, {band_count}], got {band}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        footprint = tif_bounds_as_polygon(dem_path)
        with rasterio.open(dem_path) as dem_src:
            dem_crs_wkt = dem_src.crs.to_wkt()

        optical_clip = tmp_dir_path / "optical_clip.tif"
        clip_tif_by_aoi(optical_path, optical_clip, footprint, aoi_crs=dem_crs_wkt)

        dem_upsampled = tmp_dir_path / "dem_upsampled.tif"
        align_raster_grid(optical_clip, dem_path, dem_upsampled, resampling=resampling)

        with rasterio.open(dem_upsampled) as up_src:
            baseline = up_src.read(1).astype(np.float64)
            profile = up_src.profile.copy()
            nodata = up_src.nodata
            valid_mask = up_src.read_masks(1) != 0

        with rasterio.open(optical_clip) as guide_src:
            if band is not None:
                guide_raw = guide_src.read(band).astype(np.float64)
            else:
                guide_raw = guide_src.read().astype(np.float64).mean(axis=0)

        guide_min = guide_raw.min()
        guide_max = guide_raw.max()
        if guide_max > guide_min:
            guide = (guide_raw - guide_min) / (guide_max - guide_min)
        else:
            guide = np.zeros_like(guide_raw)

        baseline_filled = baseline.copy()
        if not valid_mask.all():
            fill_value = baseline[valid_mask].mean() if valid_mask.any() else 0.0
            baseline_filled[~valid_mask] = fill_value

        result = _guided_filter(guide, baseline_filled, radius, eps)

        if nodata is not None:
            result[~valid_mask] = nodata

        profile.update(dtype="float32")
        out_path = Path(output_path)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(result.astype(np.float32), 1)

    return out_path
```

- [ ] **Step 4: Run to verify happy-path tests pass**

Run: `uv run pytest tests/test_sr.py -k "guided_upsample" -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/vecraspy/sr.py tests/test_sr.py
git commit -m "feat: add guided_upsample_dem with guided image filter"
```

---

### Task 4: `guided_upsample_dem` — nodata preservation test

**Files:**
- Modify: `tests/test_sr.py` — add nodata test (implementation already handles this from Task 3)

- [ ] **Step 1: Write the test**

```python
def test_guided_upsample_dem_preserves_nodata(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"

    dem_data = np.full((4, 4), 10.0, dtype=np.float32)
    dem_data[0, :] = -9999.0
    _write_tif(dem, dem_data, west=0, south=0, east=40, north=40, nodata=-9999.0)
    _write_multiband_tif(
        optical, np.ones((1, 16, 16), dtype=np.float32), west=0, south=0, east=40, north=40
    )

    out = tmp_path / "out.tif"
    guided_upsample_dem(dem, optical, out)

    with rasterio.open(out) as ds:
        assert ds.nodata == -9999.0
        data = ds.read(1)
        assert data[0, 0] == -9999.0
        assert data[-1, 0] != -9999.0
```

- [ ] **Step 2: Run to verify it passes**

Run: `uv run pytest tests/test_sr.py -k "preserves_nodata" -v`
Expected: 1 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_sr.py
git commit -m "test: add guided_upsample_dem nodata preservation test"
```

---

### Task 5: `guided_upsample_dem` — parameter validation tests

**Files:**
- Modify: `tests/test_sr.py` — add error-path tests (implementation already handles this from Task 3)

- [ ] **Step 1: Write the tests**

```python
def test_guided_upsample_dem_missing_dem_raises(tmp_path):
    optical = tmp_path / "optical.tif"
    _write_multiband_tif(optical, np.ones((1, 4, 4), dtype=np.float32))
    with pytest.raises(FileNotFoundError, match="file not found"):
        guided_upsample_dem(tmp_path / "ghost.tif", optical, tmp_path / "out.tif")


def test_guided_upsample_dem_missing_optical_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    with pytest.raises(FileNotFoundError, match="file not found"):
        guided_upsample_dem(dem, tmp_path / "ghost.tif", tmp_path / "out.tif")


def test_guided_upsample_dem_invalid_radius_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    _write_multiband_tif(optical, np.ones((1, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="radius must be > 0"):
        guided_upsample_dem(dem, optical, tmp_path / "out.tif", radius=0)


def test_guided_upsample_dem_invalid_eps_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    _write_multiband_tif(optical, np.ones((1, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="eps must be > 0"):
        guided_upsample_dem(dem, optical, tmp_path / "out.tif", eps=0)


def test_guided_upsample_dem_invalid_band_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    _write_multiband_tif(optical, np.ones((1, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match=r"band must be in \[1, 1\]"):
        guided_upsample_dem(dem, optical, tmp_path / "out.tif", band=2)


def test_guided_upsample_dem_invalid_resampling_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    _write_multiband_tif(optical, np.ones((1, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="invalid resampling"):
        guided_upsample_dem(dem, optical, tmp_path / "out.tif", resampling="magic")
```

- [ ] **Step 2: Run to verify they pass**

Run: `uv run pytest tests/test_sr.py -k "guided_upsample" -v`
Expected: all guided_upsample_dem tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_sr.py
git commit -m "test: add guided_upsample_dem parameter validation tests"
```

---

### Task 6: `simulate_thermal_erosion` — core implementation and happy-path tests

**Files:**
- Modify: `tests/test_sr.py` — add erosion tests
- Modify: `src/vecraspy/sr.py` — add `_NEIGHBOR_OFFSETS`, `_shift`, `simulate_thermal_erosion`

**Interfaces:**
- Produces: `simulate_thermal_erosion(dem_path, output_path, *, iterations=50, talus_angle=35.0, transfer_rate=0.5) -> Path`

- [ ] **Step 1: Write the failing tests**

```python
from vecraspy.sr import simulate_thermal_erosion


def test_simulate_thermal_erosion_flat_terrain_unchanged(tmp_path):
    dem = tmp_path / "dem.tif"
    data = np.full((10, 10), 50.0, dtype=np.float32)
    _write_tif(dem, data, west=0, south=0, east=10, north=10)

    out = tmp_path / "out.tif"
    simulate_thermal_erosion(dem, out, iterations=10)

    with rasterio.open(out) as ds:
        result = ds.read(1)
    np.testing.assert_allclose(result, data, atol=1e-6)


def test_simulate_thermal_erosion_reduces_spike(tmp_path):
    dem = tmp_path / "dem.tif"
    data = np.full((9, 9), 10.0, dtype=np.float32)
    data[4, 4] = 100.0
    _write_tif(dem, data, west=0, south=0, east=9, north=9)

    out = tmp_path / "out.tif"
    simulate_thermal_erosion(dem, out, iterations=20, talus_angle=45.0, transfer_rate=0.5)

    with rasterio.open(out) as ds:
        result = ds.read(1)

    assert result[4, 4] < 100.0
    assert result[4, 4] > 10.0


def test_simulate_thermal_erosion_conserves_total_mass(tmp_path):
    dem = tmp_path / "dem.tif"
    data = np.full((9, 9), 10.0, dtype=np.float32)
    data[4, 4] = 100.0
    _write_tif(dem, data, west=0, south=0, east=9, north=9)

    out = tmp_path / "out.tif"
    simulate_thermal_erosion(dem, out, iterations=20, talus_angle=45.0, transfer_rate=0.5)

    with rasterio.open(out) as ds:
        result = ds.read(1)

    assert result.sum() == pytest.approx(data.sum(), rel=1e-4)


def test_simulate_thermal_erosion_ignores_nodata_neighbors(tmp_path):
    dem = tmp_path / "dem.tif"
    data = np.full((9, 9), 10.0, dtype=np.float32)
    data[4, 4] = 100.0
    data[4, 5] = -9999.0  # nodata cell right next to the spike
    _write_tif(dem, data, west=0, south=0, east=9, north=9, nodata=-9999.0)

    out = tmp_path / "out.tif"
    simulate_thermal_erosion(dem, out, iterations=20, talus_angle=45.0, transfer_rate=0.5)

    with rasterio.open(out) as ds:
        result = ds.read(1)
        valid = ds.read_masks(1) != 0

    # The nodata cell must stay untouched, and no real elevation may drain
    # into it — total mass over the *valid* region alone is conserved.
    assert result[4, 5] == -9999.0
    assert result[valid].sum() == pytest.approx(data[valid].sum(), rel=1e-4)


def test_simulate_thermal_erosion_returns_path(tmp_path):
    dem = tmp_path / "dem.tif"
    _write_tif(dem, np.full((4, 4), 5.0, dtype=np.float32))
    out = tmp_path / "out.tif"
    result = simulate_thermal_erosion(dem, out)
    assert isinstance(result, Path)
    assert result == out
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_sr.py -k "thermal_erosion" -v`
Expected: `ImportError` — `simulate_thermal_erosion` not defined yet

- [ ] **Step 3: Implement in `sr.py`**

Append to `src/vecraspy/sr.py`:

```python
_NEIGHBOR_OFFSETS: list[tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return arr shifted by (dy, dx), replicating edge values (no wrap)."""
    height, width = arr.shape
    padded = np.pad(arr, 1, mode="edge")
    return padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]


def simulate_thermal_erosion(
    dem_path: Path | str,
    output_path: Path | str,
    *,
    iterations: int = 50,
    talus_angle: float = 35.0,
    transfer_rate: float = 0.5,
) -> Path:
    """Refine a DEM with a vectorized talus-angle thermal erosion simulation.

    Each iteration, every cell sheds material toward its steepest of 8
    neighbours if the slope to that neighbour exceeds talus_angle, moving a
    transfer_rate fraction of the excess toward the talus-stable level.
    Purely procedural, independent of any optical guide. Nodata cells do not
    participate in material transport.

    Args:
        dem_path: Path to the source DEM GeoTIFF.
        output_path: Path for the eroded output GeoTIFF.
        iterations: Number of simulation steps.
        talus_angle: Talus (repose) angle in degrees, in (0, 90).
        transfer_rate: Fraction (0, 1] of the excess above the talus-stable
            level moved per iteration.

    Returns:
        The resolved output_path as a Path.

    Raises:
        FileNotFoundError: If dem_path does not exist.
        ValueError: If iterations <= 0, talus_angle is outside (0, 90), or
            transfer_rate is outside (0, 1].
    """
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"file not found: {dem_path}")
    if iterations <= 0:
        raise ValueError(f"iterations must be > 0, got {iterations}")
    if not 0 < talus_angle < 90:
        raise ValueError(f"talus_angle must be in (0, 90), got {talus_angle}")
    if not 0 < transfer_rate <= 1:
        raise ValueError(f"transfer_rate must be in (0, 1], got {transfer_rate}")

    with rasterio.open(dem_path) as src:
        height_map = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        nodata = src.nodata
        valid_mask = src.read_masks(1) != 0
        pixel_size = (abs(src.transform.a) + abs(src.transform.e)) / 2

    talus_slope = math.tan(math.radians(talus_angle))
    rows, cols = np.indices(height_map.shape)

    for _ in range(iterations):
        best_diff = np.zeros_like(height_map)
        best_dy = np.zeros(height_map.shape, dtype=np.int8)
        best_dx = np.zeros(height_map.shape, dtype=np.int8)

        for dy, dx in _NEIGHBOR_OFFSETS:
            neighbor = _shift(height_map, dy, dx)
            neighbor_valid = _shift(valid_mask, dy, dx)
            distance = pixel_size * math.hypot(dy, dx)
            diff = height_map - neighbor - talus_slope * distance
            # A nodata neighbour must never be chosen as a transport target —
            # its raw sentinel value (e.g. -9999) would otherwise look like an
            # enormous drop and drain real elevation from valid cells at the
            # border, since the nodata mask is only restored once at the end.
            diff = np.where(neighbor_valid, diff, -np.inf)
            is_better = diff > best_diff
            best_diff = np.where(is_better, diff, best_diff)
            best_dy = np.where(is_better, dy, best_dy)
            best_dx = np.where(is_better, dx, best_dx)

        transportable = valid_mask & (best_diff > 0)
        amount = np.where(transportable, transfer_rate * best_diff / 2, 0.0)

        target_rows = np.clip(rows + best_dy, 0, height_map.shape[0] - 1)
        target_cols = np.clip(cols + best_dx, 0, height_map.shape[1] - 1)

        inflow = np.zeros_like(height_map)
        np.add.at(
            inflow,
            (target_rows[transportable], target_cols[transportable]),
            amount[transportable],
        )

        height_map = height_map - amount + inflow

    if nodata is not None:
        height_map[~valid_mask] = nodata

    profile.update(dtype="float32")
    out_path = Path(output_path)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(height_map.astype(np.float32), 1)

    return out_path
```

- [ ] **Step 4: Run to verify happy-path tests pass**

Run: `uv run pytest tests/test_sr.py -k "thermal_erosion" -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/vecraspy/sr.py tests/test_sr.py
git commit -m "feat: add simulate_thermal_erosion cellular automaton"
```

---

### Task 7: `simulate_thermal_erosion` — parameter validation tests

**Files:**
- Modify: `tests/test_sr.py` — add error-path tests (implementation already handles this from Task 6)

- [ ] **Step 1: Write the tests**

```python
def test_simulate_thermal_erosion_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="file not found"):
        simulate_thermal_erosion(tmp_path / "ghost.tif", tmp_path / "out.tif")


def test_simulate_thermal_erosion_invalid_iterations_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="iterations must be > 0"):
        simulate_thermal_erosion(dem, tmp_path / "out.tif", iterations=0)


def test_simulate_thermal_erosion_invalid_talus_angle_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="talus_angle must be in"):
        simulate_thermal_erosion(dem, tmp_path / "out.tif", talus_angle=90.0)


def test_simulate_thermal_erosion_invalid_transfer_rate_raises(tmp_path):
    dem = tmp_path / "dem.tif"
    _write_tif(dem, np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="transfer_rate must be in"):
        simulate_thermal_erosion(dem, tmp_path / "out.tif", transfer_rate=0.0)
```

- [ ] **Step 2: Run to verify they pass**

Run: `uv run pytest tests/test_sr.py -k "thermal_erosion" -v`
Expected: all simulate_thermal_erosion tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_sr.py
git commit -m "test: add simulate_thermal_erosion parameter validation tests"
```

---

### Task 8: `super_resolve_dtm` convenience function

**Files:**
- Modify: `tests/test_sr.py` — add pipeline tests
- Modify: `src/vecraspy/sr.py` — add `super_resolve_dtm`

**Interfaces:**
- Consumes: `guided_upsample_dem(dem_path, optical_path, output_path, *, band=None, radius=8, eps=1e-2, resampling="cubic") -> Path` (Task 3), `simulate_thermal_erosion(dem_path, output_path, *, iterations=50, talus_angle=35.0, transfer_rate=0.5) -> Path` (Task 6)
- Produces: `super_resolve_dtm(dem_path, optical_path, output_path, *, band=None, radius=8, eps=1e-2, apply_erosion=False, erosion_kwargs=None) -> Path`

- [ ] **Step 1: Write the failing tests**

```python
from vecraspy.sr import super_resolve_dtm


def test_super_resolve_dtm_without_erosion_matches_guided_upsample(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"
    _write_tif(
        dem,
        np.arange(16, dtype=np.float32).reshape(4, 4) + 10.0,
        west=0,
        south=0,
        east=40,
        north=40,
    )
    _write_multiband_tif(
        optical, np.ones((1, 16, 16), dtype=np.float32), west=0, south=0, east=40, north=40
    )

    direct = tmp_path / "direct.tif"
    guided_upsample_dem(dem, optical, direct)

    via_pipeline = tmp_path / "pipeline.tif"
    result = super_resolve_dtm(dem, optical, via_pipeline)

    assert result == via_pipeline
    with rasterio.open(direct) as a, rasterio.open(via_pipeline) as b:
        np.testing.assert_allclose(a.read(1), b.read(1))


def test_super_resolve_dtm_with_erosion_applies_erosion(tmp_path):
    dem = tmp_path / "dem.tif"
    optical = tmp_path / "optical.tif"
    _write_tif(
        dem,
        np.arange(16, dtype=np.float32).reshape(4, 4) + 10.0,
        west=0,
        south=0,
        east=40,
        north=40,
    )
    _write_multiband_tif(
        optical, np.ones((1, 16, 16), dtype=np.float32), west=0, south=0, east=40, north=40
    )

    without_erosion = tmp_path / "without.tif"
    guided_upsample_dem(dem, optical, without_erosion, radius=2)

    with_erosion = tmp_path / "with.tif"
    super_resolve_dtm(
        dem,
        optical,
        with_erosion,
        radius=2,
        apply_erosion=True,
        erosion_kwargs={"iterations": 5, "talus_angle": 5.0, "transfer_rate": 1.0},
    )

    with rasterio.open(without_erosion) as a, rasterio.open(with_erosion) as b:
        data_a = a.read(1)
        data_b = b.read(1)
    assert not np.allclose(data_a, data_b)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_sr.py -k "super_resolve_dtm" -v`
Expected: `ImportError` — `super_resolve_dtm` not defined yet

- [ ] **Step 3: Implement in `sr.py`**

Append to `src/vecraspy/sr.py`:

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
    """Super-resolve a DEM: guided upsampling, plus optional thermal erosion.

    Chains guided_upsample_dem and, when apply_erosion is True,
    simulate_thermal_erosion.

    Args:
        dem_path: Path to the source DEM GeoTIFF.
        optical_path: Path to the co-registered high-resolution optical GeoTIFF.
        output_path: Path for the final output GeoTIFF.
        band: Forwarded to guided_upsample_dem.
        radius: Forwarded to guided_upsample_dem.
        eps: Forwarded to guided_upsample_dem.
        apply_erosion: If True, run simulate_thermal_erosion on the guided
            upsampling result before writing output_path.
        erosion_kwargs: Keyword arguments forwarded to simulate_thermal_erosion
            when apply_erosion is True (e.g. iterations, talus_angle,
            transfer_rate).

    Returns:
        The resolved output_path as a Path.
    """
    output_path = Path(output_path)

    if not apply_erosion:
        return guided_upsample_dem(
            dem_path, optical_path, output_path, band=band, radius=radius, eps=eps
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        upsampled = Path(tmp_dir) / "upsampled.tif"
        guided_upsample_dem(
            dem_path, optical_path, upsampled, band=band, radius=radius, eps=eps
        )
        simulate_thermal_erosion(upsampled, output_path, **(erosion_kwargs or {}))

    return output_path
```

- [ ] **Step 4: Run to verify tests pass**

Run: `uv run pytest tests/test_sr.py -k "super_resolve_dtm" -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/vecraspy/sr.py tests/test_sr.py
git commit -m "feat: add super_resolve_dtm convenience pipeline"
```

---

### Task 9: Export from `__init__.py`

**Files:**
- Modify: `src/vecraspy/__init__.py`

- [ ] **Step 1: Add the imports and `__all__` entries**

In `src/vecraspy/__init__.py`, add to the import block:

```python
from vecraspy.sr import (
    guided_upsample_dem,
    simulate_thermal_erosion,
    super_resolve_dtm,
)
```

And add to `__all__` (keep the list alphabetized):

```python
    "guided_upsample_dem",
    "simulate_thermal_erosion",
    "super_resolve_dtm",
```

- [ ] **Step 2: Verify the imports work**

Run: `uv run python -c "from vecraspy import guided_upsample_dem, simulate_thermal_erosion, super_resolve_dtm; print(guided_upsample_dem, simulate_thermal_erosion, super_resolve_dtm)"`
Expected: prints the three function objects, no errors

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass, including all `tests/test_sr.py` tests and no regressions elsewhere

- [ ] **Step 4: Lint**

Run: `uv run ruff check src/vecraspy/sr.py tests/test_sr.py src/vecraspy/__init__.py`
Expected: no issues

- [ ] **Step 5: Commit**

```bash
git add src/vecraspy/__init__.py
git commit -m "feat: export sr pipeline functions from vecraspy package"
```
