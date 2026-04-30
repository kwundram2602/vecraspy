# plot_slope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `plot_slope` to `src/vecraspy/plot.py` with unit-aware fixed display range and `"RdYlGn_r"` default colourmap.

**Architecture:** `plot_slope` follows the exact same pattern as the existing `plot_dsm` function — path in, rasterio read, matplotlib imshow, return `(Figure, Axes)`. The only structural addition is `_SLOPE_UNIT_STYLES`, a module-level dict that maps each unit name to `(vmin, vmax, colorbar_label)` so all three are looked up in one place.

**Tech Stack:** Python 3.13, rasterio, numpy, matplotlib, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/vecraspy/plot.py` | Modify | Add `DEFAULT_SLOPE_CMAP`, `_SLOPE_UNIT_STYLES`, `plot_slope` |
| `tests/test_plot.py` | Create | All tests for `plot_slope` |

---

### Task 1: Test file with `_write_slope_tif` helper

**Files:**
- Create: `tests/test_plot.py`

- [ ] **Step 1: Create the test file**

Write `tests/test_plot.py` with this content:

```python
"""Tests for vecraspy.plot."""

import math
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

matplotlib.use("Agg")  # non-interactive backend for tests

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _write_slope_tif(
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

- [ ] **Step 2: Verify it parses cleanly**

Run from `d:\py_projects\vecraspy`:
```
uv run python -c "import importlib.util; spec = importlib.util.spec_from_file_location('t', 'tests/test_plot.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"
```
Expected: no output (no import errors)

- [ ] **Step 3: Commit**

```bash
git add tests/test_plot.py
git commit -m "test: add test_plot.py with _write_slope_tif helper"
```

---

### Task 2: Error handling tests + validation implementation

**Files:**
- Modify: `tests/test_plot.py` — append error tests
- Modify: `src/vecraspy/plot.py` — add constants and validation stub

- [ ] **Step 1: Append error tests to `tests/test_plot.py`**

```python
import pytest

from vecraspy.plot import plot_slope


def test_plot_slope_invalid_units_raises(tmp_path):
    tif = tmp_path / "slope.tif"
    _write_slope_tif(tif, np.full((4, 4), 30.0, dtype=np.float32))
    with pytest.raises(ValueError, match="invalid units"):
        plot_slope(tif, units="gradians")


def test_plot_slope_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        plot_slope(tmp_path / "ghost.tif")


def test_plot_slope_all_nodata_raises(tmp_path):
    tif = tmp_path / "slope.tif"
    data = np.full((4, 4), -9999.0, dtype=np.float32)
    _write_slope_tif(tif, data, nodata=-9999.0)
    with pytest.raises(ValueError, match="no valid"):
        plot_slope(tif)
```

- [ ] **Step 2: Run to verify they fail**

Run:
```
uv run pytest tests/test_plot.py -k "raises" -v
```
Expected: `ImportError` or `FAILED` — `plot_slope` not defined yet

- [ ] **Step 3: Add constants and `plot_slope` stub to `plot.py`**

Add `import math` to the imports at the top of `src/vecraspy/plot.py`:

```python
"""Plotting utilities for raster and vector geodata."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.axes import Axes
from matplotlib.figure import Figure
```

Then append after the existing `DEFAULT_FIGURE_SIZE` constant (before `plot_dsm`):

```python
DEFAULT_SLOPE_CMAP = "RdYlGn_r"

_SLOPE_UNIT_STYLES: dict[str, tuple[float, float, str]] = {
    "degrees": (0.0, 90.0, "Slope (°)"),
    "radians": (0.0, math.pi / 2, "Slope (rad)"),
    "percent": (0.0, 100.0, "Slope (%)"),
}


def plot_slope(
    path: Path | str,
    *,
    units: str = "degrees",
    cmap: str = DEFAULT_SLOPE_CMAP,
    band: int = 1,
    ax: Axes | None = None,
    show_colorbar: bool = True,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a slope raster with a unit-aware fixed display range.

    The display range is fixed per unit type (degrees: 0–90, radians: 0–π/2,
    percent: 0–100) so comparisons across scenes are visually consistent.
    Nodata pixels are masked and shown as transparent.

    Args:
        path: Path to the slope GeoTIFF.
        units: Units of the slope values — "degrees", "radians", or "percent".
            Controls vmin, vmax, and the colorbar label. Defaults to "degrees".
        cmap: Matplotlib colourmap name. Defaults to "RdYlGn_r".
        band: Raster band index (1-based). Defaults to 1.
        ax: Existing Axes to draw into. Creates a new figure when None.
        show_colorbar: Attach a colourbar. Defaults to True.
        title: Axes title. Defaults to the file stem when None.

    Returns:
        Tuple of (Figure, Axes) for further customisation.

    Raises:
        ValueError: If units is not one of "degrees", "radians", "percent".
        FileNotFoundError: If the file does not exist.
        ValueError: If all pixels in the band are nodata.
    """
    if units not in _SLOPE_UNIT_STYLES:
        raise ValueError(
            f"invalid units {units!r}; valid values: {sorted(_SLOPE_UNIT_STYLES)}"
        )

    slope_path = Path(path)
    if not slope_path.exists():
        raise FileNotFoundError(f"file not found: {slope_path}")

    raise NotImplementedError
```

- [ ] **Step 4: Run to verify error tests pass**

Run:
```
uv run pytest tests/test_plot.py -k "raises" -v
```
Expected: 3 passed (invalid units, missing file, all nodata all raise before `NotImplementedError`)

Wait — `test_plot_slope_all_nodata_raises` needs the file to exist and be read before the nodata check. The stub raises `NotImplementedError` before reaching that check. Replace the stub body after the two guard clauses with the read + nodata check, keeping `NotImplementedError` at the end:

```python
    if units not in _SLOPE_UNIT_STYLES:
        raise ValueError(
            f"invalid units {units!r}; valid values: {sorted(_SLOPE_UNIT_STYLES)}"
        )

    slope_path = Path(path)
    if not slope_path.exists():
        raise FileNotFoundError(f"file not found: {slope_path}")

    with rasterio.open(slope_path) as src:
        raw = src.read(band)
        nodata = src.nodata
        bounds = src.bounds

    data = raw.astype(float)
    if nodata is not None:
        data = np.where(raw == nodata, np.nan, data)

    valid = data[np.isfinite(data)]
    if valid.size == 0:
        raise ValueError(
            f"no valid (non-nodata) pixels in band {band} of {slope_path}"
        )

    raise NotImplementedError
```

- [ ] **Step 5: Run error tests again**

Run:
```
uv run pytest tests/test_plot.py -k "raises" -v
```
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add tests/test_plot.py src/vecraspy/plot.py
git commit -m "feat: add plot_slope validation and error handling"
```

---

### Task 3: Happy-path tests + complete implementation

**Files:**
- Modify: `tests/test_plot.py` — append happy-path tests
- Modify: `src/vecraspy/plot.py` — replace `NotImplementedError` with full implementation

- [ ] **Step 1: Append happy-path tests to `tests/test_plot.py`**

```python
def test_plot_slope_returns_figure_and_axes(tmp_path):
    tif = tmp_path / "slope.tif"
    _write_slope_tif(tif, np.full((4, 4), 30.0, dtype=np.float32))
    fig, ax = plot_slope(tif)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_slope_degrees_fixed_range(tmp_path):
    tif = tmp_path / "slope.tif"
    _write_slope_tif(tif, np.full((4, 4), 30.0, dtype=np.float32))
    fig, ax = plot_slope(tif, units="degrees")
    vmin, vmax = ax.images[0].get_clim()
    assert vmin == 0.0
    assert vmax == 90.0
    plt.close(fig)


def test_plot_slope_radians_fixed_range(tmp_path):
    tif = tmp_path / "slope.tif"
    _write_slope_tif(tif, np.full((4, 4), 0.5, dtype=np.float32))
    fig, ax = plot_slope(tif, units="radians")
    vmin, vmax = ax.images[0].get_clim()
    assert vmin == 0.0
    assert abs(vmax - math.pi / 2) < 1e-6
    plt.close(fig)


def test_plot_slope_percent_fixed_range(tmp_path):
    tif = tmp_path / "slope.tif"
    _write_slope_tif(tif, np.full((4, 4), 50.0, dtype=np.float32))
    fig, ax = plot_slope(tif, units="percent")
    vmin, vmax = ax.images[0].get_clim()
    assert vmin == 0.0
    assert vmax == 100.0
    plt.close(fig)


def test_plot_slope_colorbar_label_degrees(tmp_path):
    tif = tmp_path / "slope.tif"
    _write_slope_tif(tif, np.full((4, 4), 30.0, dtype=np.float32))
    fig, ax = plot_slope(tif, units="degrees", show_colorbar=True)
    colorbar_ax = fig.axes[-1]
    assert colorbar_ax.get_ylabel() == "Slope (°)"
    plt.close(fig)


def test_plot_slope_default_title_is_file_stem(tmp_path):
    tif = tmp_path / "my_slope.tif"
    _write_slope_tif(tif, np.full((4, 4), 30.0, dtype=np.float32))
    fig, ax = plot_slope(tif)
    assert ax.get_title() == "my_slope"
    plt.close(fig)


def test_plot_slope_custom_title(tmp_path):
    tif = tmp_path / "slope.tif"
    _write_slope_tif(tif, np.full((4, 4), 30.0, dtype=np.float32))
    fig, ax = plot_slope(tif, title="My slope map")
    assert ax.get_title() == "My slope map"
    plt.close(fig)


def test_plot_slope_accepts_existing_axes(tmp_path):
    tif = tmp_path / "slope.tif"
    _write_slope_tif(tif, np.full((4, 4), 30.0, dtype=np.float32))
    fig_existing, ax_existing = plt.subplots()
    fig, ax = plot_slope(tif, ax=ax_existing)
    assert ax is ax_existing
    assert fig is fig_existing
    plt.close(fig)
```

- [ ] **Step 2: Run to verify they fail**

Run:
```
uv run pytest tests/test_plot.py -v
```
Expected: happy-path tests `FAILED` with `NotImplementedError`, error tests still pass

- [ ] **Step 3: Replace `raise NotImplementedError` with full implementation**

Replace the final `raise NotImplementedError` line in `plot_slope` with:

```python
    vmin, vmax, colorbar_label = _SLOPE_UNIT_STYLES[units]
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    else:
        fig = ax.get_figure()

    img = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="upper",
    )

    ax.set_title(title if title is not None else slope_path.stem)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")

    if show_colorbar:
        fig.colorbar(img, ax=ax, label=colorbar_label)

    return fig, ax
```

- [ ] **Step 4: Run full test suite**

Run:
```
uv run pytest tests/ -v
```
Expected: all tests pass (existing raster tests + all new plot tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_plot.py src/vecraspy/plot.py
git commit -m "feat: add plot_slope with unit-aware fixed display range"
```
