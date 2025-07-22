import pytest
import numpy as np
from pyAPIC.core.case import Case
from pyAPIC.core.parameters import ReconParams
from pyAPIC.io.mat_reader import ImagingData


def make_imaging_data():
    return ImagingData(
        I_low=np.ones((2, 4, 4)),
        illum_px=np.zeros((2, 2)),
        system_na_px=1.0,
    )


def test_case_result_before_run():
    case = Case(data=make_imaging_data(), params=ReconParams())
    with pytest.raises(RuntimeError):
        _ = case.result


def test_case_run_and_result():
    case = Case(data=make_imaging_data(), params=ReconParams())
    case.run()
    result = case.result
    assert "E_stitched" in result
    assert result["E_stitched"].shape == (4, 4)
