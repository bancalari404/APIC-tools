import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from pyAPIC.visual import plotters
from pyAPIC.io.mat_reader import ImagingData


def make_data():
    return ImagingData(
        I_low=np.ones((2, 4, 4)),
        illum_px=np.zeros((2, 2)),
        system_na_px=1.0,
    )


def test_plot_initial(monkeypatch):
    data = make_data()
    monkeypatch.setattr(plt, "show", lambda: None)
    plotters.plot_input(data, ncols=2)


def test_plot_results(monkeypatch):
    res = {"E_stitched": np.ones((4, 4), dtype=complex)}
    monkeypatch.setattr(plt, "show", lambda: None)
    plotters.plot_results(res)
