import numpy as np
import pytest

from pyAPIC.imaging_utils import to_pixel_coords


def test_to_pixel_coords_na_units():
    angles = np.array([[0.0, 1.0], [0.0, 0.0]])
    res = to_pixel_coords(
        angles,
        system_na_px=1.0,
        system_na=1.0,
        center=np.array([2, 2]),
        wavelength=1.0,
    )
    expected = angles + np.array([[2], [2]])
    assert np.allclose(res, expected)


def test_to_pixel_coords_freq_units():
    angles = np.array([[0.0, 1.0], [0.0, 0.0]])
    res = to_pixel_coords(
        angles,
        system_na_px=2.0,
        system_na=1.0,
        center=np.array([0, 0]),
        units="freq",
    )
    expected = angles * 2.0
    assert np.allclose(res, expected)


def test_to_pixel_coords_requires_wavelength():
    with pytest.raises(ValueError):
        to_pixel_coords(
            np.zeros((2, 1)),
            system_na_px=1.0,
            system_na=1.0,
            center=np.zeros(2),
        )


def test_to_pixel_coords_invalid_units():
    with pytest.raises(ValueError):
        to_pixel_coords(
            np.zeros((2, 1)),
            system_na_px=1.0,
            system_na=1.0,
            center=np.zeros(2),
            units="deg",
            wavelength=1.0,
        )


def test_to_pixel_coords_requires_na():
    with pytest.raises(ValueError):
        to_pixel_coords(
            np.zeros((2, 1)),
            system_na_px=1.0,
            center=np.zeros(2),
            wavelength=1.0,
        )
