from pyAPIC.core.case import Case
from pyAPIC.core.parameters import ReconParams

params = ReconParams(reconstruct_aberration=True)
case = Case.from_mat("example/Setup_18_TUG_3.mat", params)
case.run()
res = case.result
# res['E_stack'], res['aberration']
case.plot()
