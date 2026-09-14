import numpy as np
import matplotlib.pyplot as plt

def plot_multigroup(filename):
    # Load CSV into numpy array
    data = np.loadtxt(filename, delimiter=",", skiprows=1)

    # First column is x
    x = data[:, 0]

    # Remaining columns are fluxes for each group
    for g in range(1, data.shape[1]):
        plt.plot(x, data[:, g], label=f"Group {g}")

    plt.xlabel("x [cm]")
    plt.ylabel("Flux")
    plt.title("Multigroup Neutron Diffusion Solution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_multigroup("solution_multigroup.csv")
