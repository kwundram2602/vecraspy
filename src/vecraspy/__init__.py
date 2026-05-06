"""vecraspy — raster and vector utility functions."""

from vecraspy.raster import (
    aggregate_raster,
    extract_raster_values_at_points,
    filter_tifs_by_aoi,
    merge_tifs,
    ndvi,
    scale_raster_to_gsd,
    summarize_extracted_raster_values,
    tif_bounds_as_polygon,
)
from vecraspy.terrain import hillshade
from vecraspy.vector import Trajectory, build_trajectories, write_trajectories

__all__ = [
    "build_trajectories",
    "aggregate_raster",
    "extract_raster_values_at_points",
    "filter_tifs_by_aoi",
    "hillshade",
    "merge_tifs",
    "ndvi",
    "scale_raster_to_gsd",
    "summarize_extracted_raster_values",
    "tif_bounds_as_polygon",
    "Trajectory",
    "write_trajectories",
]
