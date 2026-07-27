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
