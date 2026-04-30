# plot_slope Design

**Date:** 2026-04-30
**Status:** Approved

## Summary

Add `plot_slope` to `src/vecraspy/plot.py` alongside `plot_dsm`. The function visualises a slope raster with a fixed display range that is driven by the `units` parameter, using `"RdYlGn_r"` as the default colourmap. Unit-aware colorbar labels are looked up from a module-level dict rather than scattered through the function body.

---

## Interface

```python
DEFAULT_SLOPE_CMAP = "RdYlGn_r"

_SLOPE_UNIT_STYLES: dict[str, tuple[float, float, str]] = {
    "degrees": (0.0, 90.0, "Slope (°)"),
    "radians": (0.0, 1.5707963, "Slope (rad)"),
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
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `Path \| str` | — | Path to the slope GeoTIFF |
| `units` | `str` | `"degrees"` | Units of the slope values — `"degrees"`, `"radians"`, or `"percent"`. Drives vmin, vmax, and colorbar label |
| `cmap` | `str` | `"RdYlGn_r"` | Matplotlib colourmap name |
| `band` | `int` | `1` | Raster band index (1-based) |
| `ax` | `Axes \| None` | `None` | Existing Axes to draw into; creates a new figure when `None` |
| `show_colorbar` | `bool` | `True` | Attach a colourbar |
| `title` | `str \| None` | `None` | Axes title; defaults to the file stem |

### Returns

`tuple[Figure, Axes]` — for further customisation after the call.

---

## Unit Styles

`_SLOPE_UNIT_STYLES` maps each supported unit to `(vmin, vmax, colorbar_label)`:

| units | vmin | vmax | colorbar label |
|---|---|---|---|
| `"degrees"` | 0.0 | 90.0 | `"Slope (°)"` |
| `"radians"` | 0.0 | π/2 ≈ 1.5707963 | `"Slope (rad)"` |
| `"percent"` | 0.0 | 100.0 | `"Slope (%)"` |

---

## Internal Pipeline

1. Validate `units` against `_SLOPE_UNIT_STYLES` — raise `ValueError` with the bad value and valid options if unknown
2. Check path exists — raise `FileNotFoundError` if not
3. Open with rasterio, read band, mask nodata pixels → `np.nan`
4. Raise `ValueError` if no valid (non-nodata) pixels
5. Look up `(vmin, vmax, colorbar_label)` from `_SLOPE_UNIT_STYLES[units]`
6. Create `(fig, ax)` via `plt.subplots` if `ax` is `None`; otherwise use existing axes and get figure via `ax.get_figure()`
7. `ax.imshow` with fixed `vmin`/`vmax`, `cmap`, spatial `extent`, `origin="upper"`
8. Set title (default: `path.stem`), x-label `"Easting"`, y-label `"Northing"`
9. Attach colourbar with `colorbar_label` when `show_colorbar=True`
10. Return `(fig, ax)`

---

## Error Handling

| Situation | Behaviour |
|---|---|
| `units` not in `_SLOPE_UNIT_STYLES` | `ValueError` with bad value and valid options |
| `path` does not exist | `FileNotFoundError` |
| All pixels are nodata | `ValueError` with band and path |

---

## File Location

- Implementation: `src/vecraspy/plot.py` (append after `plot_dsm`)
- No new files required
- No export needed (plot functions are not in `__init__.py`)
