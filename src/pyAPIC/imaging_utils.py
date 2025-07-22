from __future__ import annotations

import numpy as np


def to_pixel_coords(
    angles: np.ndarray,
    *,
    system_na_px: float,
    system_na: float | None = None,
    center: np.ndarray,
    wavelength: float | None = None,
    units: str = "na",
    na_cal: float | None = None,
) -> np.ndarray:
    """Convert illumination angles to the ``illum_px`` pixel coordinates.

    Parameters
    ----------
    angles:
        Array of illumination angles of shape ``(2, N)``.
    system_na_px:
        System numerical aperture expressed in pixel units.
    system_na:
        System numerical aperture (unitless). ``na_cal`` may be used as an
        alias for backward compatibility.
    center:
        Image center ``(y, x)``.
    wavelength:
        Illumination wavelength. Required when ``units='na'``.
    units:
        Either ``'na'`` for ``n sin(\theta)`` or ``'freq'`` for spatial
        frequency units.
    """
    angles = np.asarray(angles)

    if system_na is None:
        system_na = na_cal
    if system_na is None:
        raise ValueError("system_na is required")

    if units not in {"na", "freq"}:
        raise ValueError("units must be 'na' or 'freq'")

    if units == "na":
        if wavelength is None:
            raise ValueError("wavelength is required when units='na'")
        freq_uv = angles / wavelength
    else:
        freq_uv = angles

    con = system_na_px / system_na
    center = np.asarray(center).reshape(2, 1)
    return freq_uv * con + center
