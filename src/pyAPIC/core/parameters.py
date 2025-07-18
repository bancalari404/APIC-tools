from dataclasses import dataclass

@dataclass
class ReconParams:
    """
    Parameters for phase reconstruction.

    Attributes:
        reconstruct_aberration: Whether to perform aberration reconstruction.
        stitch_method: Method used to stitch the Fourier patches ("average" or
            "nearest").
    """
    reconstruct_aberration: bool = False
    stitch_method: str = "average"
