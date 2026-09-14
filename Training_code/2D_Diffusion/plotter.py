#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def load_xy_flux(filename):
    """
    Load triplet-form data:
        x  y  flux
    written in nested loops from Fortran.

    Returns:
        X, Y, F : 2D numpy arrays ready for plotting.
    """
    data = np.loadtxt(filename)

    x = data[:, 0]
    y = data[:, 1]
    f = data[:, 2]

    # Unique coordinates
    x_unique = np.unique(x)
    y_unique = np.unique(y)
   

    nx = len(x_unique)
    ny = len(y_unique)

    # Reshape based on Fortran ordering (i loops outer, j inner)
    X = x.reshape((ny, nx))
    Y = y.reshape((ny, nx))
    
    F = f.reshape((ny, nx))
    F = np.transpose(F)
    
    
    

    return X, Y, F



def plot_flux(X, Y, F):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: numerical flux ---
    ax = axes[0]
    im = ax.imshow(
        F,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin='lower',
        aspect='auto'
    )
    fig.colorbar(im, ax=ax, label="Neutron flux")

    ax.contour(
        X, Y, F,
        levels=10,
        colors='k',
        linewidths=0.5
    )
    ax.set_title("Computed Flux")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # --- Right: analytic solution ---
    ax2 = axes[1]
    analytic = np.cos(np.pi * X) * np.cos(np.pi * Y)

    cs = ax2.contour(
        X, Y, analytic
    )
    fig.colorbar(cs, ax=ax2)
    ax2.set_title("Analytic cos(πx)cos(πy)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    plt.tight_layout()
    plt.show()


def main():
    filename = "build/flux.txt"   # change if needed

    X, Y, F = load_xy_flux(filename)
    plot_flux(X, Y, F)

   


if __name__ == "__main__":
    main()


