import numpy as np
from pyAPIC.core.reconstructor import (
    directed_hilbert_transform_stack,
    pupil_mask_stack,
    reconstruct,
)
from pyAPIC.core.parameters import ReconParams
from pyAPIC.io.mat_reader import ImagingData


def test_directed_hilbert_transform_constant():
    re = np.ones((1, 4, 4))
    freq = np.zeros((2, 1))
    im = directed_hilbert_transform_stack(re, freq)
    assert im.shape == re.shape
    assert np.allclose(im, 0)


def test_pupil_mask_stack():
    mask = pupil_mask_stack((1, 5, 5), np.array([[2], [2]]), NA_pix=1)
    expected = np.zeros((1, 5, 5))
    coords = [(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)]
    for y, x in coords:
        expected[0, y, x] = 1.0
    assert mask.shape == expected.shape
    assert np.array_equal(mask, expected)


def make_imaging_data():
    return ImagingData(
        I_low=np.ones((2, 4, 4)),
        freqXY_calib=np.zeros((2, 2)),
        na_rp_cal=1.0,
    )


def test_reconstruct_no_aberration():
    data = make_imaging_data()
    params = ReconParams(reconstruct_aberration=False)
    result = reconstruct(data, params)

    assert "E_stack" in result
    assert result["E_stack"].shape == data.I_low.shape
    assert "E_stitched" in result
    assert result["E_stitched"].shape == data.I_low.shape[1:]
    assert np.all(np.isfinite(result["E_stitched"]))
    assert "aberration" not in result
