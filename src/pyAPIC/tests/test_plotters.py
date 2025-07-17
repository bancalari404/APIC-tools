import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pyAPIC.visual import plotters
from pyAPIC.io.mat_reader import ImagingData


def make_data():
    return ImagingData(
        I_low=np.ones((2, 4, 4)),
        freqXY_calib=np.zeros((2, 2)),
        na_rp_cal=1.0,
        dpix_c=1.0,
        na_calib=np.zeros((2, 2)),
        na_cal=1.0,
        wavelength=1.0,
    )


def test_plot_initial(monkeypatch):
    data = make_data()
    monkeypatch.setattr(plt, 'show', lambda: None)
    plotters.plot_initial(data, ncols=2)


def test_plot_results(monkeypatch):
    res = {'E_stack': np.ones((2, 4, 4), dtype=complex)}
    monkeypatch.setattr(plt, 'show', lambda: None)
    plotters.plot_results(res)
