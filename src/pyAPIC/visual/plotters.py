import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pyAPIC.io.mat_reader import ImagingData


def plot_input(data: ImagingData, ncols: int = 5) -> None:
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

    Parameters
    ----------
    result : dict
        Result dict from :func:`reconstruct`. It should contain ``'E_stitched'``.
    """
    E = result.get("E_stitched")
    if E is None:
        raise ValueError("Result dict must contain 'E_stitched'.")

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


def plot_E_stack(result: dict) -> None:
    """
    Plot the E_stack complex field with a slider to navigate through images.

    Parameters
    ----------
    result : dict
        Result dict from :func:`reconstruct`. It should contain ``'E_stack'``.
    """
    E_stack = result.get("E_stack")
    if E_stack is None:
        raise ValueError("Result dict must contain 'E_stack'.")
    
    N = E_stack.shape[0]
    if N == 0:
        raise ValueError("E_stack is empty.")
    
    # Initial image index
    init_idx = 0
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.25)
    
    # Initial plot
    E_init = E_stack[init_idx]
    amp_init = np.abs(E_init)
    phase_init = np.angle(E_init)
    
    im1 = ax1.imshow(amp_init, cmap='gray')
    ax1.set_title(f'Amplitude - Image {init_idx}')
    ax1.axis('off')
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    im2 = ax2.imshow(phase_init, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax2.set_title(f'Phase - Image {init_idx}')
    ax2.axis('off')
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Create slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Image', 0, N-1, valinit=init_idx, valfmt='%d')
    
    def update(val):
        idx = int(slider.val)
        E = E_stack[idx]
        amp = np.abs(E)
        phase = np.angle(E)
        
        im1.set_array(amp)
        im1.set_clim(vmin=amp.min(), vmax=amp.max())
        ax1.set_title(f'Amplitude - Image {idx}')
        
        im2.set_array(phase)
        ax2.set_title(f'Phase - Image {idx}')
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # plt.tight_layout()
    plt.show()
