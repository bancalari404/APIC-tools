import numpy as np
import h5py
from pyAPIC.io.mat_reader import load_mat


def create_mat_file(path, include_freq=True):
    with h5py.File(path, "w") as f:
        f.create_dataset("I_low", data=np.ones((2, 4, 4)))
        if include_freq:
            f.create_dataset("freqXY_calib", data=np.zeros((2, 2)))
        f.create_dataset("na_rp_cal", data=1.0)
        f.create_dataset("dpix_c", data=1.0)
        f.create_dataset("na_calib", data=np.zeros((2, 2)))
        f.create_dataset("na_cal", data=1.0)
        f.create_dataset("lambda", data=1.0)


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
