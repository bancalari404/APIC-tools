"""pyAPIC public API."""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - version is resolved only when installed
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover - local source checkout
    __version__ = "0.0.0"

from .core.case import Case
from .core.parameters import ReconParams
from .imaging_utils import to_pixel_coords
from .io.mat_reader import ImagingData, load_mat

__all__ = [
    "__version__",
    "to_pixel_coords",
    "Case",
    "ReconParams",
    "ImagingData",
    "load_mat",
]
