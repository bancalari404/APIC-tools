import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

from pyAPIC.io.mat_reader import ImagingData
from pyAPIC.core.parameters import ReconParams
from pyAPIC.core.solve_ctf_operators import get_ctf

# Fourier helpers
FFT = lambda x: fftshift(fft2(x), axes=(-2, -1))
IFFT = lambda x: ifft2(ifftshift(x, axes=(-2, -1)))


def directed_hilbert_transform_stack(Re_stack: np.ndarray, freqXY_stack: np.ndarray) -> np.ndarray:
    """
    Perform the directed Hilbert transform on a stack of real-valued images.
    Returns the imaginary part stack.
    """
    # FFT of each frame
    F_Re = FFT(Re_stack)
    N, H, W = F_Re.shape

    # frequency grid centered
    center = np.array([H, W]) // 2
    u = np.arange(H) - center[0]
    v = np.arange(W) - center[1]
    U, V = np.meshgrid(u, v, indexing='xy')
    U = np.broadcast_to(U, (N, H, W))
    V = np.broadcast_to(V, (N, H, W))

    # LED positions in pixel coords relative to center
    kx = freqXY_stack[0].reshape(N, 1, 1) - center[0]
    ky = freqXY_stack[1].reshape(N, 1, 1) - center[1]

    dot = U * kx + V * ky
    # Hilbert mask: -1j * sign(dot)
    hilbert_mask = -1j * np.sign(dot)

    F_hilbert = F_Re * hilbert_mask
    Im_stack = IFFT(F_hilbert)
    return Im_stack


def pupil_mask_stack(shape: tuple, freqXY_stack: np.ndarray, NA_pix: float) -> np.ndarray:
    """
    Generate pupil masks for each LED frame.
    shape: (N, H, W)
    freqXY_stack: (2, N)
    """
    N, H, W = shape
    u = np.arange(H)
    v = np.arange(W)
    U, V = np.meshgrid(u, v, indexing='xy')
    U = np.broadcast_to(U, (N, H, W))
    V = np.broadcast_to(V, (N, H, W))

    kx = freqXY_stack[0].reshape(N, 1, 1)
    ky = freqXY_stack[1].reshape(N, 1, 1)

    R = np.sqrt((U - kx)**2 + (V - ky)**2)
    mask = np.zeros((N, H, W), dtype=float)
    mask[R <= NA_pix] = 1.0
    return mask


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
    Im = directed_hilbert_transform_stack(Re, data.freqXY_calib)
    # 3. Combine and exponentiate
    logE = Re + 1j * Im
    E_stack = np.exp(logE)

    result = {'E_stack': E_stack}

    # 4. Aberration
    if params.reconstruct_aberration:
        # Compute shifts relative to Fourier center
        H, W = data.I_low.shape[1:]
        center = np.array([H, W]) // 2
        shifts = (data.freqXY_calib.T - center).astype(int)
        # Compute CTF aberration using your operator
        F_E = FFT(E_stack)
        CTF_abe = get_ctf(F_E, shifts, CTF_radius=data.na_rp_cal,
                           useWeights=False, useZernike=True).conj()
        result['aberration'] = CTF_abe

    return result
