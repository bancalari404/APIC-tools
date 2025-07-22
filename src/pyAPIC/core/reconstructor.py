import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

from pyAPIC.io.mat_reader import ImagingData
from pyAPIC.core.parameters import ReconParams
from pyAPIC.core.solve_ctf_operators import get_ctf


# Fourier helpers
def fft2c(x: np.ndarray) -> np.ndarray:
    """Compute a centered 2-D Fourier transform.

    ``fft2`` is applied and the result is shifted so that the zero-frequency
    component appears at the center of the last two axes.
    """
    return fftshift(fft2(x), axes=(-2, -1))


def ifft2c(x: np.ndarray) -> np.ndarray:
    """Inverse transform corresponding to :func:`fft2c`.

    The input is shifted back with ``ifftshift`` before applying ``ifft2`` so
    that arrays transformed with :func:`fft2c` are perfectly inverted.
    """
    return ifft2(ifftshift(x, axes=(-2, -1)))


def directed_hilbert_transform_stack(
    Re_stack: np.ndarray, freqXY_stack: np.ndarray
) -> np.ndarray:
    """
    Perform the directed Hilbert transform on a stack of real-valued images.
    Returns the imaginary part stack.
    """
    # FFT of each frame
    F_Re = fft2c(Re_stack)
    N, H, W = F_Re.shape

    # frequency grid centered
    center = np.array([H, W]) // 2
    u = np.arange(H) - center[0]
    v = np.arange(W) - center[1]
    U, V = np.meshgrid(u, v, indexing="xy")
    U = np.broadcast_to(U, (N, H, W))
    V = np.broadcast_to(V, (N, H, W))

    # LED positions in pixel coords relative to center
    kx = freqXY_stack[0].reshape(N, 1, 1) - center[0]
    ky = freqXY_stack[1].reshape(N, 1, 1) - center[1]

    dot = U * kx + V * ky
    # Hilbert mask: -1j * sign(dot)
    hilbert_mask = -1j * np.sign(dot)

    F_hilbert = F_Re * hilbert_mask
    Im_stack = ifft2c(F_hilbert)
    return Im_stack


def pupil_mask_stack(
    shape: tuple, freqXY_stack: np.ndarray, NA_pix: float
) -> np.ndarray:
    """
    Generate pupil masks for each LED frame.
    shape: (N, H, W)
    freqXY_stack: (2, N)
    """
    N, H, W = shape
    u = np.arange(H)
    v = np.arange(W)
    U, V = np.meshgrid(u, v, indexing="xy")
    U = np.broadcast_to(U, (N, H, W))
    V = np.broadcast_to(V, (N, H, W))

    kx = freqXY_stack[0].reshape(N, 1, 1)
    ky = freqXY_stack[1].reshape(N, 1, 1)

    R = np.sqrt((U - kx) ** 2 + (V - ky) ** 2)
    mask = np.zeros((N, H, W), dtype=float)
    mask[R <= NA_pix] = 1.0
    return mask


def stitch(
    data: ImagingData,
    E_reconstructed: np.ndarray,
    CTF: np.ndarray | None = None,
    method: str = "nearest",
):
    """Combine the reconstructed field stack into a single complex field.

    Parameters
    ----------
    data : ImagingData
        Acquisition parameters used for pupil positioning.
    E_reconstructed : np.ndarray
        Complex field stack of shape (N, H, W).
    CTF : np.ndarray | None, optional
        Aberration transfer function. If provided it will be shifted to each
        LED position and multiplied with the pupil functions.
    method : str, optional
        How to combine the Fourier patches. Options are ``"average"`` and
        ``"nearest"``. ``"nearest"`` selects the patch whose illumination
        center is closest to each Fourier pixel while ``"average"`` performs an
        overlap average.  ``"nearest"`` is the default.

    Returns
    -------
    tuple
        ``(E_reconstructed, pupil_masks, effective_pupil, E_stitched)``
    """

    pupil_masks = pupil_mask_stack(
        E_reconstructed.shape, data.illum_px, data.system_na_px
    )

    if CTF is not None:
        CTFs = np.stack([CTF] * E_reconstructed.shape[0], axis=0)
        for i in range(CTFs.shape[0]):
            center = np.array(CTFs.shape[1:]) // 2
            shift = (np.round(data.illum_px[:, i] - center)).astype(int)[::-1]
            CTFs[i] = np.roll(CTF, shift, axis=(0, 1))
            offset = np.angle(CTFs[i])[center[0], center[1]]
            CTFs[i] *= np.exp(-1j * offset)
        pupil_masks = pupil_masks.astype(np.complex128) * CTFs

    F_E_reconstructed = fft2c(E_reconstructed)
    F_E_reconstructed *= pupil_masks

    if method == "average":
        F_E_stitched = np.mean(F_E_reconstructed, axis=0)
        pupil_masks_sum = np.sum(np.abs(pupil_masks), axis=0)
        effective_pupil = pupil_masks_sum > 0
        pupil_masks_sum[pupil_masks_sum == 0] = 1
        F_E_stitched /= pupil_masks_sum
    elif method == "nearest":
        num_images, H, W = F_E_reconstructed.shape
        grid_y, grid_x = np.indices((H, W))
        distance_maps = np.empty((num_images, H, W), dtype=float)
        for i in range(num_images):
            x_ctf = data.illum_px[0, i]
            y_ctf = data.illum_px[1, i]
            distance_maps[i] = np.sqrt((grid_x - x_ctf) ** 2 + (grid_y - y_ctf) ** 2)
        best_idx = np.argmin(distance_maps, axis=0)
        F_E_stitched = np.zeros((H, W), dtype=F_E_reconstructed.dtype)
        for i in range(num_images):
            mask = best_idx == i
            F_E_stitched[mask] = F_E_reconstructed[i][mask]
        effective_pupil = np.any(np.abs(pupil_masks) > 0, axis=0)
        F_E_stitched *= effective_pupil
    else:
        raise ValueError("Invalid method. Choose 'average' or 'nearest'.")

    E_stitched = ifft2c(F_E_stitched).conj()

    return E_reconstructed, pupil_masks, effective_pupil, E_stitched


def reconstruct(data: ImagingData, params: ReconParams) -> dict:
    """
    Perform the full reconstruction pipeline:
      1. Compute real part: 0.5*log(I)
      2. Directed Hilbert transform -> imaginary part
      3. Exponentiate to get complex field stack
      4. Optionally reconstruct aberration via get_ctf

    Returns a result dict containing:
      - 'E_stack': complex field stack (N, H, W)
      - 'aberration': CTF array if computed
    """
    # 1. Real component of log-field
    Re = 0.5 * np.log(data.I_low)
    # 2. Imaginary via Hilbert transform
    Im = directed_hilbert_transform_stack(Re, data.illum_px)
    # 3. Combine and exponentiate
    logE = Re + 1j * Im
    E_stack = np.exp(logE)

    result = {"E_stack": E_stack}

    # 4. Aberration
    CTF_abe = None
    if params.reconstruct_aberration:
        H, W = data.I_low.shape[1:]
        center = np.array([H, W]) // 2
        shifts = (data.illum_px.T - center).astype(int)
        F_E = fft2c(E_stack)
        CTF_abe = get_ctf(
            F_E, shifts, CTF_radius=data.system_na_px, useWeights=False, useZernike=True
        ).conj()
        result["aberration"] = CTF_abe

    # 5. Stitch the stack into a single field
    _, _, _, E_stitched = stitch(
        data, E_stack, CTF=CTF_abe, method=params.stitch_method
    )
    result["E_stitched"] = E_stitched

    return result
