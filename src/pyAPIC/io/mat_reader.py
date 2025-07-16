from dataclasses import dataclass
import numpy as np
import h5py


@dataclass
class ImagingData:
    """
    Container for imaging data and acquisition parameters.
    """
    I_low: np.ndarray          # Intensity images stack, shape (N, H, W)
    freqXY_calib: np.ndarray   # Illumination angles in pixel coords, shape (2, N)
    na_rp_cal: float           # Numerical aperture in pixel units
    dpix_c: float              # Physical pixel size (e.g. um/px)
    na_calib: np.ndarray       # Illumination angles in NA units, shape (2, N)
    na_cal: float              # System NA in unity
    wavelength: float          # Illumination wavelength (same units as dpix_c)


def load_mat(path: str, downsample: int = 1) -> ImagingData:
    """
    Load imaging data and calibration parameters from a MATLAB .mat file (v7.3/HDF5).

    Parameters:
        path (str): Path to the .mat file.
        downsample (int): Factor by which to subsample the image stack along the illumination axis.

    Returns:
        ImagingData: Populated dataclass.
    """
    with h5py.File(path, 'r') as f:
        # Read raw datasets
        I_low = f['I_low'][:]  # shape (N, H, W)
        freqXY_calib = f['freqXY_calib'][:]  # shape (2, N)
        na_rp_cal = float(f['na_rp_cal'][()])  # scalar or 1-element array
        dpix_c = float(f['dpix_c'][()])
        na_calib = f['na_calib'][:]  # shape (2, N)
        na_cal = float(f['na_cal'][()])
        # wavelength is often stored as a scalar array
        wavelength = float(f['lambda'][()])

    # Subsample if requested
    if downsample > 1:
        I_low = I_low[::downsample]
        freqXY_calib = freqXY_calib[:, ::downsample]
        na_calib = na_calib[:, ::downsample]

    return ImagingData(
        I_low=I_low,
        freqXY_calib=freqXY_calib,
        na_rp_cal=na_rp_cal,
        dpix_c=dpix_c,
        na_calib=na_calib,
        na_cal=na_cal,
        wavelength=wavelength
    )
