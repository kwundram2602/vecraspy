# Trajectory Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trajectory pipeline: read points from GeoPackage → group by ID → build `Trajectory` objects in `vecraspy`, then plot or animate them over a GeoTIFF raster in `mapgod`.

**Architecture:** `vecraspy/vector.py` provides a `Trajectory` dataclass and four functions (`read_points`, `group_by_id`, `build_trajectory`, `build_trajectories`). `mapgod/trajectory.py` provides `plot_trajectory` and `animate_trajectory` which accept `Trajectory | list[Trajectory]` plus a GeoTIFF path. The packages are coupled only through the `Trajectory` type — mapgod imports it from vecraspy.

**Tech Stack:** `geopandas`, `shapely`, `rasterio`, `matplotlib`, `matplotlib.animation.FuncAnimation`, `pillow` (GIF), `pytest`

---

## File Map

| Action | Path |
|--------|------|
| Create | `vecraspy/src/vecraspy/vector.py` |
| Modify | `vecraspy/src/vecraspy/__init__.py` |
| Create | `vecraspy/tests/test_vector.py` |
| Create | `mapgod/src/mapgod/trajectory.py` |
| Create | `mapgod/src/mapgod/__init__.py` |
| Create | `mapgod/tests/__init__.py` (empty) |
| Create | `mapgod/tests/test_trajectory.py` |
| Modify | `mapgod/pyproject.toml` (add vecraspy + pillow deps) |

---

## Task 1: `Trajectory` dataclass + `read_points`

**Files:**
- Create: `vecraspy/src/vecraspy/vector.py`
- Create: `vecraspy/tests/test_vector.py`

- [ ] **Step 1: Write the failing test**

`vecraspy/tests/test_vector.py`:
```python
"""Tests for vecraspy.vector."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from vecraspy.vector import Trajectory, read_points


def _write_gpkg(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a minimal point GeoPackage from a list of row dicts (must include 'geometry')."""
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    path = tmp_path / "test.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def test_read_points_returns_geodataframe(tmp_path):
    path = _write_gpkg(tmp_path, [
        {"geometry": Point(1.0, 2.0)},
        {"geometry": Point(3.0, 4.0)},
    ])
    gdf = read_points(path)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 2


def test_read_points_preserves_crs(tmp_path):
    path = _write_gpkg(tmp_path, [{"geometry": Point(1.0, 2.0)}])
    gdf = read_points(path)
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326


def test_trajectory_dataclass():
    gdf = gpd.GeoDataFrame({"geometry": [Point(0.0, 0.0)]}, crs="EPSG:4326")
    traj = Trajectory(id="track_1", points=gdf)
    assert traj.id == "track_1"
    assert len(traj.points) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```
cd d:/py_projects/vecraspy && uv run pytest tests/test_vector.py -v
```

Expected: `FAILED` / `ModuleNotFoundError: No module named 'vecraspy.vector'`

- [ ] **Step 3: Implement `Trajectory` + `read_points`**

Create `vecraspy/src/vecraspy/vector.py`:
```python
"""Point handling and trajectory building."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd


@dataclass
class Trajectory:
    """An ordered sequence of points representing a single track."""

    id: Any
    points: gpd.GeoDataFrame


def read_points(
    path: Path | str,
    layer: str | None = None,
) -> gpd.GeoDataFrame:
    """Read a point layer from a GeoPackage."""
    kwargs: dict[str, Any] = {}
    if layer is not None:
        kwargs["layer"] = layer
    return gpd.read_file(path, **kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

```
cd d:/py_projects/vecraspy && uv run pytest tests/test_vector.py::test_read_points_returns_geodataframe tests/test_vector.py::test_read_points_preserves_crs tests/test_vector.py::test_trajectory_dataclass -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```
cd d:/py_projects/vecraspy && git add src/vecraspy/vector.py tests/test_vector.py && git commit -m "feat(vector): add Trajectory dataclass and read_points"
```

---

## Task 2: `group_by_id`

**Files:**
- Modify: `vecraspy/src/vecraspy/vector.py`
- Modify: `vecraspy/tests/test_vector.py`

- [ ] **Step 1: Write the failing test**

Append to `vecraspy/tests/test_vector.py`:
```python
from vecraspy.vector import group_by_id


def test_group_by_id_splits_into_correct_groups(tmp_path):
    path = _write_gpkg(tmp_path, [
        {"geometry": Point(1.0, 1.0), "track_id": "a"},
        {"geometry": Point(2.0, 2.0), "track_id": "b"},
        {"geometry": Point(3.0, 3.0), "track_id": "a"},
    ])
    gdf = read_points(path)
    groups = group_by_id(gdf, "track_id")
    assert set(groups.keys()) == {"a", "b"}
    assert len(groups["a"]) == 2
    assert len(groups["b"]) == 1


def test_group_by_id_returns_geodataframes(tmp_path):
    path = _write_gpkg(tmp_path, [
        {"geometry": Point(0.0, 0.0), "id": 1},
        {"geometry": Point(1.0, 1.0), "id": 2},
    ])
    gdf = read_points(path)
    groups = group_by_id(gdf, "id")
    for val in groups.values():
        assert isinstance(val, gpd.GeoDataFrame)
```

- [ ] **Step 2: Run test to verify it fails**

```
cd d:/py_projects/vecraspy && uv run pytest tests/test_vector.py::test_group_by_id_splits_into_correct_groups -v
```

Expected: `FAILED` / `ImportError`

- [ ] **Step 3: Implement `group_by_id`**

Add to `vecraspy/src/vecraspy/vector.py` (after `read_points`):
```python
def group_by_id(
    gdf: gpd.GeoDataFrame,
    id_col: str,
) -> dict[Any, gpd.GeoDataFrame]:
    """Split a GeoDataFrame into groups by the values in id_col."""
    return {val: group.copy() for val, group in gdf.groupby(id_col)}
```

- [ ] **Step 4: Run tests to verify they pass**

```
cd d:/py_projects/vecraspy && uv run pytest tests/test_vector.py -v
```

Expected: all PASSED (5 tests)

- [ ] **Step 5: Commit**

```
cd d:/py_projects/vecraspy && git add src/vecraspy/vector.py tests/test_vector.py && git commit -m "feat(vector): add group_by_id"
```

---

## Task 3: `build_trajectory` + `build_trajectories`

**Files:**
- Modify: `vecraspy/src/vecraspy/vector.py`
- Modify: `vecraspy/tests/test_vector.py`

- [ ] **Step 1: Write the failing tests**

Append to `vecraspy/tests/test_vector.py`:
```python
from vecraspy.vector import build_trajectory, build_trajectories


def test_build_trajectory_preserves_points():
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(1.0, 1.0), Point(2.0, 2.0)]},
        crs="EPSG:4326",
    )
    traj = build_trajectory(gdf, id="x")
    assert isinstance(traj, Trajectory)
    assert len(traj.points) == 2
    assert traj.id == "x"


def test_build_trajectory_sorts_by_sort_col():
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [Point(3.0, 3.0), Point(1.0, 1.0), Point(2.0, 2.0)],
            "t": [3, 1, 2],
        },
        crs="EPSG:4326",
    )
    traj = build_trajectory(gdf, sort_col="t")
    xs = [geom.x for geom in traj.points.geometry]
    assert xs == [1.0, 2.0, 3.0]


def test_build_trajectories_no_id_col(tmp_path):
    path = _write_gpkg(tmp_path, [
        {"geometry": Point(0.0, 0.0)},
        {"geometry": Point(1.0, 1.0)},
    ])
    result = build_trajectories(path)
    assert len(result) == 1
    assert result[0].id is None
    assert len(result[0].points) == 2


def test_build_trajectories_with_id_col(tmp_path):
    path = _write_gpkg(tmp_path, [
        {"geometry": Point(0.0, 0.0), "tid": "a"},
        {"geometry": Point(1.0, 0.0), "tid": "b"},
        {"geometry": Point(2.0, 0.0), "tid": "a"},
    ])
    result = build_trajectories(path, id_col="tid")
    assert len(result) == 2
    ids = {t.id for t in result}
    assert ids == {"a", "b"}


def test_build_trajectories_sort_col(tmp_path):
    path = _write_gpkg(tmp_path, [
        {"geometry": Point(3.0, 0.0), "tid": "a", "t": 3},
        {"geometry": Point(1.0, 0.0), "tid": "a", "t": 1},
        {"geometry": Point(2.0, 0.0), "tid": "a", "t": 2},
    ])
    result = build_trajectories(path, id_col="tid", sort_col="t")
    xs = [geom.x for geom in result[0].points.geometry]
    assert xs == [1.0, 2.0, 3.0]
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd d:/py_projects/vecraspy && uv run pytest tests/test_vector.py::test_build_trajectory_preserves_points -v
```

Expected: `FAILED` / `ImportError`

- [ ] **Step 3: Implement `build_trajectory` + `build_trajectories`**

Add to `vecraspy/src/vecraspy/vector.py`:
```python
def build_trajectory(
    gdf: gpd.GeoDataFrame,
    sort_col: str | None = None,
    id: Any = None,
) -> Trajectory:
    """Build a Trajectory from an ordered (or sortable) GeoDataFrame of points."""
    pts = gdf.sort_values(sort_col) if sort_col is not None else gdf
    return Trajectory(id=id, points=pts.copy())


def build_trajectories(
    path: Path | str,
    id_col: str | None = None,
    sort_col: str | None = None,
    layer: str | None = None,
) -> list[Trajectory]:
    """Read a GeoPackage and return one Trajectory per unique id_col value.

    If id_col is None, the entire dataset is returned as a single Trajectory.
    """
    gdf = read_points(path, layer=layer)
    if id_col is None:
        return [build_trajectory(gdf, sort_col=sort_col)]
    groups = group_by_id(gdf, id_col)
    return [
        build_trajectory(pts, sort_col=sort_col, id=id_val)
        for id_val, pts in groups.items()
    ]
```

- [ ] **Step 4: Run all tests to verify they pass**

```
cd d:/py_projects/vecraspy && uv run pytest tests/test_vector.py -v
```

Expected: all 11 tests PASSED

- [ ] **Step 5: Commit**

```
cd d:/py_projects/vecraspy && git add src/vecraspy/vector.py tests/test_vector.py && git commit -m "feat(vector): add build_trajectory and build_trajectories"
```

---

## Task 4: Export `Trajectory` + `build_trajectories` from vecraspy

**Files:**
- Modify: `vecraspy/src/vecraspy/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `vecraspy/tests/test_vector.py`:
```python
def test_public_exports():
    import vecraspy
    assert hasattr(vecraspy, "Trajectory")
    assert hasattr(vecraspy, "build_trajectories")
```

- [ ] **Step 2: Run test to verify it fails**

```
cd d:/py_projects/vecraspy && uv run pytest tests/test_vector.py::test_public_exports -v
```

Expected: `FAILED` / `AssertionError`

- [ ] **Step 3: Update `__init__.py`**

Replace the contents of `vecraspy/src/vecraspy/__init__.py`:
```python
"""vecraspy — raster and vector utility functions."""

from vecraspy.raster import filter_tifs_by_aoi, merge_tifs, ndvi, tif_bounds_as_polygon
from vecraspy.terrain import hillshade
from vecraspy.vector import Trajectory, build_trajectories

__all__ = [
    "build_trajectories",
    "filter_tifs_by_aoi",
    "hillshade",
    "merge_tifs",
    "ndvi",
    "tif_bounds_as_polygon",
    "Trajectory",
]
```

- [ ] **Step 4: Run all vecraspy tests**

```
cd d:/py_projects/vecraspy && uv run pytest tests/ -v
```

Expected: all tests PASSED

- [ ] **Step 5: Commit**

```
cd d:/py_projects/vecraspy && git add src/vecraspy/__init__.py tests/test_vector.py && git commit -m "feat(vector): export Trajectory and build_trajectories"
```

---

## Task 5: Add vecraspy dependency to mapgod + `plot_trajectory`

**Files:**
- Modify: `mapgod/pyproject.toml`
- Create: `mapgod/src/mapgod/trajectory.py`
- Create: `mapgod/src/mapgod/__init__.py`
- Create: `mapgod/tests/__init__.py`
- Create: `mapgod/tests/test_trajectory.py`

- [ ] **Step 1: Install vecraspy into mapgod and add pillow**

```
cd d:/py_projects/mapgod && uv add --editable ../vecraspy && uv add pillow
```

Verify `mapgod/pyproject.toml` now contains vecraspy and pillow in `dependencies`.

- [ ] **Step 2: Write the failing test**

Create `mapgod/tests/__init__.py` (empty file).

Create `mapgod/tests/test_trajectory.py`:
```python
"""Tests for mapgod.trajectory."""

from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
import rasterio
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Point

matplotlib.use("Agg")

from vecraspy.vector import Trajectory


def _write_raster(tmp_path: Path) -> Path:
    """Write a 4x4 float32 GeoTIFF in EPSG:4326 covering [0,4] x [0,4]."""
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    transform = from_bounds(0.0, 0.0, 4.0, 4.0, 4, 4)
    path = tmp_path / "test.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


def _make_trajectory(id=None) -> Trajectory:
    pts = [Point(1.0, 1.0), Point(2.0, 2.0), Point(3.0, 3.0)]
    gdf = gpd.GeoDataFrame({"geometry": pts}, crs="EPSG:4326")
    return Trajectory(id=id, points=gdf)


def test_plot_trajectory_returns_figure_and_axes(tmp_path):
    from mapgod.trajectory import plot_trajectory

    raster = _write_raster(tmp_path)
    traj = _make_trajectory(id="t1")
    fig, ax = plot_trajectory(raster, traj)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_trajectory_list_of_trajectories(tmp_path):
    from mapgod.trajectory import plot_trajectory

    raster = _write_raster(tmp_path)
    trajs = [_make_trajectory(id="a"), _make_trajectory(id="b")]
    fig, ax = plot_trajectory(raster, trajs)
    assert isinstance(fig, Figure)


def test_plot_trajectory_accepts_existing_ax(tmp_path):
    import matplotlib.pyplot as plt
    from mapgod.trajectory import plot_trajectory

    raster = _write_raster(tmp_path)
    fig_pre, ax_pre = plt.subplots()
    fig, ax = plot_trajectory(raster, _make_trajectory(), ax=ax_pre)
    assert ax is ax_pre
```

- [ ] **Step 3: Run tests to verify they fail**

```
cd d:/py_projects/mapgod && uv run pytest tests/test_trajectory.py::test_plot_trajectory_returns_figure_and_axes -v
```

Expected: `FAILED` / `ModuleNotFoundError: No module named 'mapgod.trajectory'`

- [ ] **Step 4: Create `mapgod/src/mapgod/__init__.py`** (empty for now)

```python
"""mapgod — geospatial plotting utilities."""
```

- [ ] **Step 5: Implement `plot_trajectory`**

Create `mapgod/src/mapgod/trajectory.py`:
```python
"""Trajectory plotting and animation over GeoTIFF rasters."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from vecraspy.vector import Trajectory


def _load_raster(
    path: Path | str, band: int
) -> tuple[np.ndarray, list[float], Any]:
    """Return (data_2d, extent, raster_crs) from a GeoTIFF."""
    with rasterio.open(path) as src:
        raw = src.read(band).astype(float)
        nodata = src.nodata
        bounds = src.bounds
        crs = src.crs
    if nodata is not None:
        raw = np.where(raw == nodata, np.nan, raw)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    return raw, extent, crs


def _reproject_points(
    traj: Trajectory, raster_crs: Any
) -> tuple[list[float], list[float]]:
    """Return (xs, ys) of trajectory points reprojected to raster CRS."""
    epsg = raster_crs.to_epsg()
    target = f"EPSG:{epsg}" if epsg else raster_crs.to_wkt()
    pts = traj.points.to_crs(target) if traj.points.crs else traj.points
    return [g.x for g in pts.geometry], [g.y for g in pts.geometry]


def _color_cycle() -> list[str]:
    return [p["color"] for p in plt.rcParams["axes.prop_cycle"]]


def plot_trajectory(
    raster: Path | str,
    trajectories: Trajectory | list[Trajectory],
    *,
    band: int = 1,
    cmap: str = "gray",
    point_style: dict | None = None,
    line_style: dict | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """Plot one or more trajectories over a GeoTIFF raster (static)."""
    from vecraspy.vector import Trajectory as _Trajectory

    if isinstance(trajectories, _Trajectory):
        trajectories = [trajectories]

    data, extent, raster_crs = _load_raster(raster, band)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    valid = data[np.isfinite(data)]
    vmin = float(np.nanpercentile(valid, 2)) if valid.size > 0 else None
    vmax = float(np.nanpercentile(valid, 98)) if valid.size > 0 else None
    ax.imshow(data, cmap=cmap, extent=extent, origin="upper", vmin=vmin, vmax=vmax)

    ls: dict = {"linewidth": 1.5, "alpha": 0.8}
    ps: dict = {"markersize": 6, "zorder": 5}
    if line_style:
        ls.update(line_style)
    if point_style:
        ps.update(point_style)

    colors = _color_cycle()
    for i, traj in enumerate(trajectories):
        color = colors[i % len(colors)]
        xs, ys = _reproject_points(traj, raster_crs)
        if len(xs) > 1:
            ax.plot(xs, ys, color=color, **ls)
        ax.plot(xs[-1], ys[-1], "o", color=color, **ps)

    if title:
        ax.set_title(title)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    return fig, ax
```

- [ ] **Step 6: Run tests to verify they pass**

```
cd d:/py_projects/mapgod && uv run pytest tests/test_trajectory.py::test_plot_trajectory_returns_figure_and_axes tests/test_trajectory.py::test_plot_trajectory_list_of_trajectories tests/test_trajectory.py::test_plot_trajectory_accepts_existing_ax -v
```

Expected: 3 PASSED

- [ ] **Step 7: Commit**

```
cd d:/py_projects/mapgod && git add src/mapgod/__init__.py src/mapgod/trajectory.py tests/__init__.py tests/test_trajectory.py pyproject.toml uv.lock && git commit -m "feat(trajectory): add plot_trajectory"
```

---

## Task 6: `animate_trajectory`

**Files:**
- Modify: `mapgod/src/mapgod/trajectory.py`
- Modify: `mapgod/tests/test_trajectory.py`

- [ ] **Step 1: Write the failing tests**

Append to `mapgod/tests/test_trajectory.py`:
```python
def test_animate_trajectory_returns_funcanimation(tmp_path):
    from matplotlib.animation import FuncAnimation
    from mapgod.trajectory import animate_trajectory

    raster = _write_raster(tmp_path)
    traj = _make_trajectory(id="t1")
    anim = animate_trajectory(raster, traj)
    assert isinstance(anim, FuncAnimation)


def test_animate_trajectory_list_of_trajectories(tmp_path):
    from matplotlib.animation import FuncAnimation
    from mapgod.trajectory import animate_trajectory

    raster = _write_raster(tmp_path)
    trajs = [_make_trajectory(id="a"), _make_trajectory(id="b")]
    anim = animate_trajectory(raster, trajs)
    assert isinstance(anim, FuncAnimation)


def test_animate_trajectory_saves_gif(tmp_path):
    from mapgod.trajectory import animate_trajectory

    raster = _write_raster(tmp_path)
    traj = _make_trajectory()
    output = tmp_path / "out.gif"
    animate_trajectory(raster, traj, output=output, fps=5)
    assert output.exists()
    assert output.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd d:/py_projects/mapgod && uv run pytest tests/test_trajectory.py::test_animate_trajectory_returns_funcanimation -v
```

Expected: `FAILED` / `ImportError`

- [ ] **Step 3: Implement `animate_trajectory`**

Add to `mapgod/src/mapgod/trajectory.py` (after `plot_trajectory`):
```python
def animate_trajectory(
    raster: Path | str,
    trajectories: Trajectory | list[Trajectory],
    *,
    band: int = 1,
    cmap: str = "gray",
    point_style: dict | None = None,
    line_style: dict | None = None,
    interval: int = 100,
    output: Path | str | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
    fps: int = 10,
):
    """Animate one or more trajectories building up over a GeoTIFF raster.

    Each frame reveals one more point. Multiple trajectories run in parallel;
    shorter ones freeze at their last point. Returns a FuncAnimation object.
    Saves to .gif (pillow) or .mp4 (ffmpeg) when output is provided.
    """
    from matplotlib.animation import FuncAnimation
    from vecraspy.vector import Trajectory as _Trajectory

    if isinstance(trajectories, _Trajectory):
        trajectories = [trajectories]

    data, extent, raster_crs = _load_raster(raster, band)

    fig, ax = plt.subplots(figsize=figsize)

    valid = data[np.isfinite(data)]
    vmin = float(np.nanpercentile(valid, 2)) if valid.size > 0 else None
    vmax = float(np.nanpercentile(valid, 98)) if valid.size > 0 else None
    ax.imshow(data, cmap=cmap, extent=extent, origin="upper", vmin=vmin, vmax=vmax)

    if title:
        ax.set_title(title)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")

    ls: dict = {"linewidth": 1.5, "alpha": 0.8}
    ps: dict = {"markersize": 6, "zorder": 5}
    if line_style:
        ls.update(line_style)
    if point_style:
        ps.update(point_style)

    colors = _color_cycle()
    all_coords: list[tuple[list[float], list[float]]] = []
    for traj in trajectories:
        xs, ys = _reproject_points(traj, raster_crs)
        all_coords.append((xs, ys))

    n_frames = max(len(c[0]) for c in all_coords)

    # Pre-create one (line, point_marker) artist pair per trajectory.
    artist_groups = []
    for i, (xs, ys) in enumerate(all_coords):
        color = colors[i % len(colors)]
        (line,) = ax.plot([], [], color=color, **ls)
        (marker,) = ax.plot([], [], "o", color=color, **ps)
        artist_groups.append((line, marker, xs, ys))

    def _update(frame: int):
        updated = []
        for line, marker, xs, ys in artist_groups:
            n = min(frame + 1, len(xs))
            if n >= 2:
                line.set_data(xs[:n], ys[:n])
            else:
                line.set_data([], [])
            marker.set_data([xs[n - 1]], [ys[n - 1]])
            updated.extend([line, marker])
        return updated

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=interval, blit=True)

    if output is not None:
        output = Path(output)
        if output.suffix == ".gif":
            anim.save(output, writer="pillow", fps=fps)
        else:
            anim.save(output, fps=fps)

    return anim
```

- [ ] **Step 4: Run all mapgod tests**

```
cd d:/py_projects/mapgod && uv run pytest tests/test_trajectory.py -v
```

Expected: all 6 tests PASSED

- [ ] **Step 5: Commit**

```
cd d:/py_projects/mapgod && git add src/mapgod/trajectory.py tests/test_trajectory.py && git commit -m "feat(trajectory): add animate_trajectory"
```

---

## Task 7: Export public API from mapgod

**Files:**
- Modify: `mapgod/src/mapgod/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `mapgod/tests/test_trajectory.py`:
```python
def test_public_exports():
    import mapgod
    assert hasattr(mapgod, "plot_trajectory")
    assert hasattr(mapgod, "animate_trajectory")
```

- [ ] **Step 2: Run test to verify it fails**

```
cd d:/py_projects/mapgod && uv run pytest tests/test_trajectory.py::test_public_exports -v
```

Expected: `FAILED` / `AssertionError`

- [ ] **Step 3: Update `__init__.py`**

Replace `mapgod/src/mapgod/__init__.py`:
```python
"""mapgod — geospatial plotting utilities."""

from mapgod.trajectory import animate_trajectory, plot_trajectory

__all__ = [
    "animate_trajectory",
    "plot_trajectory",
]
```

- [ ] **Step 4: Run full test suite for both packages**

```
cd d:/py_projects/vecraspy && uv run pytest tests/ -v
cd d:/py_projects/mapgod && uv run pytest tests/ -v
```

Expected: all tests PASSED in both packages

- [ ] **Step 5: Commit**

```
cd d:/py_projects/mapgod && git add src/mapgod/__init__.py tests/test_trajectory.py && git commit -m "feat: export plot_trajectory and animate_trajectory from mapgod"
```

---

## End-to-end usage example

After all tasks complete, the full workflow looks like:

```python
from vecraspy import build_trajectories
from mapgod import plot_trajectory, animate_trajectory

# Functional pipeline in vecraspy
trajectories = build_trajectories(
    "tracks.gpkg",
    id_col="track_id",
    sort_col="timestamp",
)

# Static plot in mapgod
fig, ax = plot_trajectory("dem.tif", trajectories, title="Tracks")
fig.savefig("tracks.png", dpi=150)

# Animated GIF
animate_trajectory(
    "dem.tif",
    trajectories,
    output="tracks.gif",
    interval=80,
    fps=12,
)
```
