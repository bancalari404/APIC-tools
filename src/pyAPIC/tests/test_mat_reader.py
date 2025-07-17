import numpy as np
import h5py
from pyAPIC.io.mat_reader import load_mat


def create_mat_file(path):
    with h5py.File(path, 'w') as f:
        f.create_dataset('I_low', data=np.ones((2, 4, 4)))
        f.create_dataset('freqXY_calib', data=np.zeros((2, 2)))
        f.create_dataset('na_rp_cal', data=1.0)
        f.create_dataset('dpix_c', data=1.0)
        f.create_dataset('na_calib', data=np.zeros((2, 2)))
        f.create_dataset('na_cal', data=1.0)
        f.create_dataset('lambda', data=1.0)


def test_load_mat(tmp_path):
    path = tmp_path / 'test.mat'
    create_mat_file(path)
    data = load_mat(str(path))

    assert data.I_low.shape == (2, 4, 4)
    assert data.freqXY_calib.shape == (2, 2)
    assert data.na_rp_cal == 1.0
    assert data.dpix_c == 1.0
    assert data.na_calib.shape == (2, 2)
    assert data.na_cal == 1.0
    assert data.wavelength == 1.0


def test_load_mat_downsample(tmp_path):
    path = tmp_path / 'test.mat'
    create_mat_file(path)
    data = load_mat(str(path), downsample=2)

    assert data.I_low.shape == (1, 4, 4)
    assert data.freqXY_calib.shape == (2, 1)
    assert data.na_calib.shape == (2, 1)

