from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np


@dataclass
class ImagingData:
    """
    Container for imaging data and acquisition parameters.
    """

    I_low: np.ndarray  # Intensity images stack, shape (N, H, W)
    freqXY_calib: np.ndarray  # Illumination angles in pixel coords, shape (2, N)
    na_rp_cal: float  # Numerical aperture in pixel units
    dpix_c: Optional[float] = None  # Physical pixel size (e.g. um/px)
    na_calib: Optional[np.ndarray] = (
        None  # Illumination angles in NA units, shape (2, N)
    )
    na_cal: Optional[float] = None  # System NA in unity
    wavelength: Optional[float] = None  # Illumination wavelength (same units as dpix_c)


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
        na_rp_cal = float(f["na_rp_cal"][()])
        dpix_c = float(f["dpix_c"][()]) if "dpix_c" in f else None
        na_calib = f["na_calib"][:] if "na_calib" in f else None
        na_cal = float(f["na_cal"][()]) if "na_cal" in f else None
        wavelength = float(f["lambda"][()]) if "lambda" in f else None

    # Subsample if requested
    if downsample > 1:
        I_low = I_low[::downsample]
        if freqXY_calib is not None:
            freqXY_calib = freqXY_calib[:, ::downsample]
        if na_calib is not None:
            na_calib = na_calib[:, ::downsample]

    if freqXY_calib is None and na_calib is not None and na_cal is not None:
        center = np.array(I_low.shape[1:]) // 2
        freqXY_calib = to_pixel_coords(
            na_calib,
            na_rp_cal=na_rp_cal,
            na_cal=na_cal,
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
        freqXY_calib=freqXY_calib,
        na_rp_cal=na_rp_cal,
        dpix_c=dpix_c,
        na_calib=na_calib,
        na_cal=na_cal,
        wavelength=wavelength,
    )
