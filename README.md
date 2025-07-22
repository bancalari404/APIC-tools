# pyAPIC

## Terminology

Recent versions adopt clearer naming for acquisition parameters:

| New name   | Old name      | Description                            |
|------------|---------------|----------------------------------------|
| `illum_px` | `freqXY_calib`| LED positions in pixel coordinates     |
| `illum_na` | `na_calib`    | LED positions expressed in NA units    |
| `system_na`| `na_cal`      | System numerical aperture              |
| `system_na_px` | `na_rp_cal` | System NA in pixel units               |
| `pixel_size` | `dpix_c`    | Camera pixel size                      |

The old attribute names remain available as aliases for backward
compatibility but new code should prefer the terms above.

## Development setup

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Then install the package with its development extras:

```bash
pip install -e .[dev]
```

This provides tooling for testing and linting the project.

## Public API

The following convenience imports are available from the top-level package:

```python
from pyAPIC import (
    Case,
    ImagingData,
    ReconParams,
    load_mat,
    to_pixel_coords,
)

```

They mirror the underlying modules located in ``pyAPIC.core`` and
``pyAPIC.io`` and may be used in examples and user code.
