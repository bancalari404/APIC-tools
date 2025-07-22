import numpy as np
import h5py
from pyAPIC.io.mat_reader import load_mat


def create_mat_file(
    path,
    include_freq=True,
    *,
    include_na_rp=True,
    include_mag=False,
    dpix_c=1.0,
    mag=2.0,
    na_cal=1.0,
    wavelength=1.0,
):
    with h5py.File(path, "w") as f:
        f.create_dataset("I_low", data=np.ones((2, 4, 4)))
        if include_freq:
            f.create_dataset("freqXY_calib", data=np.zeros((2, 2)))
        if include_na_rp:
            f.create_dataset("na_rp_cal", data=1.0)
        f.create_dataset("dpix_c", data=dpix_c)
        f.create_dataset("na_calib", data=np.zeros((2, 2)))
        f.create_dataset("na_cal", data=na_cal)
        f.create_dataset("lambda", data=wavelength)
        if include_mag:
            f.create_dataset("mag", data=mag)


def test_load_mat(tmp_path):
    path = tmp_path / "test.mat"
    create_mat_file(path)
    data = load_mat(str(path))

    assert data.I_low.shape == (2, 4, 4)
    assert data.illum_px.shape == (2, 2)
    assert data.system_na_px == 1.0
    assert data.pixel_size == 1.0
    assert data.illum_na.shape == (2, 2)
    assert data.system_na == 1.0
    assert data.wavelength == 1.0


def test_load_mat_downsample(tmp_path):
    path = tmp_path / "test.mat"
    create_mat_file(path)
    data = load_mat(str(path), downsample=2)

    assert data.I_low.shape == (1, 4, 4)
    assert data.illum_px.shape == (2, 1)
    assert data.illum_na.shape == (2, 1)


def test_load_mat_compute_freq(tmp_path):
    path = tmp_path / "test2.mat"
    create_mat_file(path, include_freq=False)
    data = load_mat(str(path))

    expected = np.full((2, 2), 2)
    assert np.array_equal(data.illum_px, expected)


def test_load_mat_magnification(tmp_path):
    path = tmp_path / "test_mag.mat"
    create_mat_file(path, include_mag=True, mag=1.5)
    data = load_mat(str(path))

    assert data.magnification == 1.5


def test_compute_na_rp_from_params(tmp_path):
    path = tmp_path / "test_compute.mat"
    create_mat_file(
        path,
        include_na_rp=False,
        include_mag=True,
        mag=2.0,
        dpix_c=1.0,
        na_cal=1.0,
        wavelength=0.5,
    )
    data = load_mat(str(path))

    assert np.isclose(data.system_na_px, 4.0)
