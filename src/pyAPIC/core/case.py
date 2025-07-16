from dataclasses import dataclass
from typing import Optional

from pyAPIC.io.mat_reader import ImagingData, load_mat
from pyAPIC.core.parameters import ReconParams

@dataclass
class Case:
    """
    Encapsulates a reconstruction case: data, parameters, and results.
    """
    data: ImagingData
    params: ReconParams
    _result: Optional[dict] = None

    @classmethod
    def from_mat(cls, mat_path: str, params: ReconParams, downsample: int = 1) -> "Case":
        """
        Create a Case by loading imaging data from a .mat file and assigning parameters.

        Args:
            mat_path (str): Path to the .mat file (HDF5 based).
            params (ReconParams): Reconstruction parameters.
            downsample (int): Factor to subsample the LED stack.

        Returns:
            Case: Initialized Case object (results empty until run()).
        """
        data = load_mat(mat_path, downsample)
        return cls(data=data, params=params)

    def run(self) -> None:
        """
        Execute the reconstruction algorithm. Populates self._result.
        """
        # Import here to avoid circular
        from pyAPIC.core.reconstructor import reconstruct

        self._result = reconstruct(self.data, self.params)

    @property
    def result(self) -> dict:
        """
        Retrieve reconstruction outputs. Raises if run() has not been called.
        """
        if self._result is None:
            raise RuntimeError("Reconstruction has not been executed. Call .run() first.")
        return self._result

    def save(self, path: str) -> None:
        """
        Save the result dict to disk (e.g., pickle or HDF5).

        Args:
            path (str): File path to save results.
        """
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.result, f)

    def plot(self):
        """
        Generate standard plots for this case (initial data, results).
        """
        from pyAPIC.visual.plotters import plot_initial, plot_results

        plot_initial(self.data)
        plot_results(self.result)
