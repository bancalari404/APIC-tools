from dataclasses import dataclass

@dataclass
class ReconParams:
    """
    Parameters for phase reconstruction.

    Attributes:
        reconstruct_aberration: Whether to perform aberration reconstruction.
    """
    reconstruct_aberration: bool = False
