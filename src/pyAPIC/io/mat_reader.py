from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np


@dataclass
class ImagingData:
    """Container for imaging data and acquisition parameters."""

    I_low: np.ndarray  # Intensity images stack, shape (N, H, W)
    illum_px: np.ndarray  # Illumination angles in pixel coords, shape (2, N)
    system_na_px: float  # System numerical aperture in pixel units
    pixel_size: Optional[float] = None  # Physical pixel size (e.g. um/px)
    magnification: Optional[float] = None  # Objective magnification
    illum_na: Optional[np.ndarray] = (
        None  # Illumination angles in NA units, shape (2, N)
    )
    system_na: Optional[float] = None  # System NA in unity
    wavelength: Optional[float] = None  # Illumination wavelength (same units as pixel_size)

    # ------------------------------------------------------------------
    # Backwards compatibility aliases
    # ------------------------------------------------------------------
    # The following property pairs expose the old attribute names while
    # internally storing the values under their new descriptive names.

    @property
    def freqXY_calib(self) -> np.ndarray:
        return self.illum_px

    @freqXY_calib.setter
    def freqXY_calib(self, value: np.ndarray) -> None:
        self.illum_px = value

    @property
    def dpix_c(self) -> Optional[float]:
        return self.pixel_size

    @dpix_c.setter
    def dpix_c(self, value: Optional[float]) -> None:
        self.pixel_size = value

    @property
    def na_calib(self) -> Optional[np.ndarray]:
        return self.illum_na

    @na_calib.setter
    def na_calib(self, value: Optional[np.ndarray]) -> None:
        self.illum_na = value

    @property
    def na_cal(self) -> Optional[float]:
        return self.system_na

    @na_cal.setter
    def na_cal(self, value: Optional[float]) -> None:
        self.system_na = value

    @property
    def na_rp_cal(self) -> float:
        return self.system_na_px

    @na_rp_cal.setter
    def na_rp_cal(self, value: float) -> None:
        self.system_na_px = value

    @property
    def mag(self) -> Optional[float]:
        return self.magnification

    @mag.setter
    def mag(self, value: Optional[float]) -> None:
        self.magnification = value


from ..imaging_utils import to_pixel_coords


def load_mat(path: str, downsample: int = 1) -> ImagingData:
    """
    Load imaging data and calibration parameters from a MATLAB .mat file (v7.3/HDF5).

    Parameters:
        path (str): Path to the .mat file.
        downsample (int): Factor by which to subsample the image stack along the illumination axis.

    Returns:
        ImagingData: Populated dataclass.
    """
    with h5py.File(path, "r") as f:
        I_low = f["I_low"][:]  # shape (N, H, W)
        freqXY_calib = f["freqXY_calib"][:] if "freqXY_calib" in f else None
        system_na_px = float(f["na_rp_cal"][()]) if "na_rp_cal" in f else None
        dpix_c = float(f["dpix_c"][()]) if "dpix_c" in f else None
        na_calib = f["na_calib"][:] if "na_calib" in f else None
        system_na = float(f["na_cal"][()]) if "na_cal" in f else None
        wavelength = float(f["lambda"][()]) if "lambda" in f else None
        magnification = float(f["mag"][()]) if "mag" in f else None

        if system_na_px is None and all(
            val is not None for val in (dpix_c, magnification, system_na, wavelength)
        ):
            im_size = min(I_low.shape[1:])
            system_na_px = im_size * dpix_c / magnification * system_na / wavelength

        if system_na_px is None:
            raise KeyError(
                "na_rp_cal not found and could not be computed from provided data"
            )

    # Subsample if requested
    if downsample > 1:
        I_low = I_low[::downsample]
        if freqXY_calib is not None:
            freqXY_calib = freqXY_calib[:, ::downsample]
        if na_calib is not None:
            na_calib = na_calib[:, ::downsample]

    if freqXY_calib is None and na_calib is not None and system_na is not None:
        center = np.array(I_low.shape[1:]) // 2
        freqXY_calib = to_pixel_coords(
            na_calib,
            system_na_px=system_na_px,
            system_na=system_na,
            wavelength=wavelength,
            center=center,
            units="na",
        )

    if freqXY_calib is None:
        raise KeyError(
            "freqXY_calib not found and could not be computed from provided data"
        )

    return ImagingData(
        I_low=I_low,
        illum_px=freqXY_calib,
        system_na_px=system_na_px,
        pixel_size=dpix_c,
        magnification=magnification,
        illum_na=na_calib,
        system_na=system_na,
        wavelength=wavelength,
    )
