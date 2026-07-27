# vecraspy

Raster and vector utility functions.

## DTM Super-Resolution

Upsample a DTM/DEM to the resolution of a co-registered optical image, using
the optical image as a structural guide (guided image filter), with an
optional thermal-erosion detail-polish pass.

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
