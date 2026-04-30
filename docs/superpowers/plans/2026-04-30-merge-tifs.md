# merge_tifs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `merge_tifs` to `raster.py` — stitches GeoTIFFs from a directory or list into a single output GeoTIFF, reprojecting on-the-fly via `WarpedVRT`.

**Architecture:** Each source TIF is wrapped in a `rasterio.vrt.WarpedVRT` (a zero-copy virtual view reprojected to the target CRS and resolution), then passed to `rasterio.merge.merge`. No temp files are written. The merged array is written to disk as a GeoTIFF.

**Tech Stack:** Python 3.13, rasterio, numpy, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/vecraspy/raster.py` | Modify | Add `_collect_tif_paths` helper and `merge_tifs` function |
| `src/vecraspy/__init__.py` | Modify | Export `merge_tifs` |
| `tests/test_raster.py` | Create | All tests for `merge_tifs` |

---

### Task 1: Test file with `_write_tif` fixture helper

**Files:**
- Create: `tests/test_raster.py`

- [ ] **Step 1: Create the test file with the fixture helper**

```python
"""Tests for vecraspy.raster."""

from pathlib import Path

import numpy as np
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
```

- [ ] **Step 2: Verify the file parses cleanly**

Run: `uv run python -c "import tests.test_raster"`
Expected: no output (no import errors)

---

### Task 2: Input collection — directory and list paths

**Files:**
- Modify: `tests/test_raster.py` — add collection tests
- Modify: `src/vecraspy/raster.py` — add `_collect_tif_paths`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_raster.py`:

```python
import pytest

from vecraspy.raster import _collect_tif_paths


def test_collect_from_list(tmp_path):
    t1 = tmp_path / "a.tif"
    t2 = tmp_path / "b.tif"
    _write_tif(t1, np.ones((4, 4), dtype=np.float32))
    _write_tif(t2, np.ones((4, 4), dtype=np.float32))
    result = _collect_tif_paths([t1, t2])
    assert result == [t1, t2]


def test_collect_from_directory(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _write_tif(tiles / "a.tif", np.ones((4, 4), dtype=np.float32))
    _write_tif(tiles / "b.tiff", np.ones((4, 4), dtype=np.float32))
    result = _collect_tif_paths(tiles)
    assert len(result) == 2
    assert all(p.suffix.lower() in {".tif", ".tiff"} for p in result)


def test_collect_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="directory not found"):
        _collect_tif_paths(tmp_path / "nonexistent")


def test_collect_empty_directory_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no TIF files found"):
        _collect_tif_paths(empty)


def test_collect_missing_file_in_list_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="file not found"):
        _collect_tif_paths([tmp_path / "ghost.tif"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_raster.py -k "collect" -v`
Expected: `ImportError` or `FAILED` — `_collect_tif_paths` does not exist yet

- [ ] **Step 3: Add `_collect_tif_paths` to `raster.py`**

Add after the existing imports in `src/vecraspy/raster.py`:

```python
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
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_raster.py -k "collect" -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_raster.py src/vecraspy/raster.py
git commit -m "feat: add _collect_tif_paths helper with tests"
```

---

### Task 3: Core merge pipeline — happy path

**Files:**
- Modify: `tests/test_raster.py` — add happy-path tests
- Modify: `src/vecraspy/raster.py` — add `merge_tifs`

- [ ] **Step 1: Write the failing happy-path tests**

Append to `tests/test_raster.py`:

```python
from vecraspy.raster import merge_tifs


def test_merge_tifs_from_list_stitches_horizontally(tmp_path):
    t1 = tmp_path / "tile1.tif"
    t2 = tmp_path / "tile2.tif"
    _write_tif(t1, np.ones((4, 4), dtype=np.float32), west=0, east=4, south=0, north=4)
    _write_tif(t2, np.full((4, 4), 2.0, dtype=np.float32), west=4, east=8, south=0, north=4)

    out = tmp_path / "merged.tif"
    result = merge_tifs([t1, t2], out)

    assert result == out
    assert out.exists()
    with rasterio.open(out) as ds:
        assert ds.width == 8
        assert ds.height == 4
        assert ds.count == 1


def test_merge_tifs_from_directory(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _write_tif(tiles / "a.tif", np.ones((4, 4), dtype=np.float32), west=0, east=4, south=0, north=4)
    _write_tif(tiles / "b.tif", np.ones((4, 4), dtype=np.float32), west=4, east=8, south=0, north=4)

    out = tmp_path / "merged.tif"
    result = merge_tifs(tiles, out)

    assert result == out
    with rasterio.open(out) as ds:
        assert ds.width == 8


def test_merge_tifs_returns_path(tmp_path):
    t1 = tmp_path / "t.tif"
    _write_tif(t1, np.ones((4, 4), dtype=np.float32))
    out = tmp_path / "out.tif"
    result = merge_tifs([t1], out)
    assert isinstance(result, Path)
    assert result == out


def test_merge_tifs_preserves_nodata_from_first_tif(tmp_path):
    t1 = tmp_path / "t1.tif"
    t2 = tmp_path / "t2.tif"
    data = np.ones((4, 4), dtype=np.float32)
    data[0, 0] = -9999.0
    _write_tif(t1, data, west=0, east=4, south=0, north=4, nodata=-9999.0)
    _write_tif(t2, np.ones((4, 4), dtype=np.float32), west=4, east=8, south=0, north=4, nodata=-9999.0)

    out = tmp_path / "merged.tif"
    merge_tifs([t1, t2], out)

    with rasterio.open(out) as ds:
        assert ds.nodata == -9999.0


def test_merge_tifs_caller_overrides_nodata(tmp_path):
    t1 = tmp_path / "t.tif"
    _write_tif(t1, np.ones((4, 4), dtype=np.float32), nodata=-1.0)
    out = tmp_path / "out.tif"
    merge_tifs([t1], out, nodata=0.0)
    with rasterio.open(out) as ds:
        assert ds.nodata == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_raster.py -k "merge_tifs" -v`
Expected: `ImportError` or `FAILED` — `merge_tifs` not defined yet

- [ ] **Step 3: Add imports needed by `merge_tifs` to `raster.py`**

Update the import block at the top of `src/vecraspy/raster.py` to:

```python
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
```

- [ ] **Step 4: Add `merge_tifs` to `raster.py`**

Append after `_collect_tif_paths`:

```python
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
```

- [ ] **Step 5: Run to verify happy-path tests pass**

Run: `uv run pytest tests/test_raster.py -k "merge_tifs" -v`
Expected: all merge_tifs tests pass

- [ ] **Step 6: Commit**

```bash
git add src/vecraspy/raster.py tests/test_raster.py
git commit -m "feat: add merge_tifs with WarpedVRT pipeline"
```

---

### Task 4: Error handling — invalid resampling

**Files:**
- Modify: `tests/test_raster.py` — add error tests

- [ ] **Step 1: Write the failing error tests**

Append to `tests/test_raster.py`:

```python
def test_merge_tifs_invalid_resampling_raises(tmp_path):
    t1 = tmp_path / "t.tif"
    _write_tif(t1, np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="invalid resampling"):
        merge_tifs([t1], tmp_path / "out.tif", resampling="notamethod")


def test_merge_tifs_missing_file_in_list_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="file not found"):
        merge_tifs([tmp_path / "ghost.tif"], tmp_path / "out.tif")


def test_merge_tifs_missing_source_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="directory not found"):
        merge_tifs(tmp_path / "nonexistent", tmp_path / "out.tif")


def test_merge_tifs_empty_directory_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no TIF files found"):
        merge_tifs(empty, tmp_path / "out.tif")
```

- [ ] **Step 2: Run to verify they pass (error handling already implemented in Task 3)**

Run: `uv run pytest tests/test_raster.py -v`
Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_raster.py
git commit -m "test: add merge_tifs error handling tests"
```

---

### Task 5: Export `merge_tifs` from `__init__.py`

**Files:**
- Modify: `src/vecraspy/__init__.py`

- [ ] **Step 1: Add the export**

Replace the contents of `src/vecraspy/__init__.py` with:

```python
"""vecraspy — raster and vector utility functions."""

from vecraspy.raster import merge_tifs, tif_bounds_as_polygon

__all__ = ["merge_tifs", "tif_bounds_as_polygon"]
```

- [ ] **Step 2: Verify the import works**

Run: `uv run python -c "from vecraspy import merge_tifs; print(merge_tifs)"`
Expected: `<function merge_tifs at 0x...>`

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add src/vecraspy/__init__.py
git commit -m "feat: export merge_tifs from vecraspy package"
```
