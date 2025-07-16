import math
import numpy as np
import matplotlib.pyplot as plt
from pyAPIC.io.mat_reader import ImagingData


def plot_initial(data: ImagingData, ncols: int = 5) -> None:
    """
    Plot the initial intensity stack as a grid of images.

    Args:
        data (ImagingData): Contains I_low stack of shape (N, H, W).
        ncols (int): Number of columns in the grid.
    """
    I_stack = data.I_low
    N, H, W = I_stack.shape
    ncols = min(ncols, N)
    nrows = math.ceil(N / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        if idx < N:
            ax.imshow(I_stack[idx], cmap='gray')
            ax.set_title(f'Image {idx}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_results(result: dict) -> None:
    """
    Plot the reconstructed complex field: amplitude and phase.

    Args:
        result (dict): Result dict from reconstruct(); must contain 'E_stack'.
    """
    E_stack = result.get('E_stack')
    if E_stack is None:
        raise ValueError("Result dict must contain 'E_stack'.")

    # Combine stack by averaging
    E = np.mean(E_stack, axis=0)
    amp = np.abs(E)
    phase = np.angle(E)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax1.imshow(amp, cmap='gray')
    ax1.set_title('Amplitude')
    ax1.axis('off')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax2.set_title('Phase')
    ax2.axis('off')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
