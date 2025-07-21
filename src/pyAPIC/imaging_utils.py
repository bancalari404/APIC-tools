from __future__ import annotations

import numpy as np


def to_pixel_coords(
    angles: np.ndarray,
    *,
    na_rp_cal: float,
    na_cal: float,
    center: np.ndarray,
    wavelength: float | None = None,
    units: str = "na",
) -> np.ndarray:
    """Convert illumination angles to the ``freqXY_calib`` pixel coordinates.

    Parameters
    ----------
    angles:
        Array of illumination angles of shape ``(2, N)``.
    na_rp_cal:
        Numerical aperture expressed in pixel units.
    na_cal:
        System numerical aperture (unitless).
    center:
        Image center ``(y, x)``.
    wavelength:
        Illumination wavelength. Required when ``units='na'``.
    units:
        Either ``'na'`` for ``n sin(\theta)`` or ``'freq'`` for spatial
        frequency units.
    """
    angles = np.asarray(angles)

    if units not in {"na", "freq"}:
        raise ValueError("units must be 'na' or 'freq'")

    if units == "na":
        if wavelength is None:
            raise ValueError("wavelength is required when units='na'")
        freq_uv = angles / wavelength
    else:
        freq_uv = angles

    con = na_rp_cal / na_cal
    center = np.asarray(center).reshape(2, 1)
    return freq_uv * con + center
