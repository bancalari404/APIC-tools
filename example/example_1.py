from pyAPIC.core.case import Case
from pyAPIC.core.parameters import ReconParams

params = ReconParams(reconstruct_aberration=False)
example_case = Case.from_mat("example/Setup_18_TUG_3.mat", params)
example_case.run()
print("System NA (px):", example_case.data.system_na_px)
res = example_case.result
example_case.plot()
example_case.plot_input()
example_case.plot_E_stack()
