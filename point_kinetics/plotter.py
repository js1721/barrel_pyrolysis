import numpy as np
import matplotlib.pyplot as plt

def load_cvode_output(filename):
    """
    Load CVODE output written as:
    # t y1 y2 ... yN
    data rows...
    """
    with open(filename, "r") as f:
        header = f.readline().strip()

    labels = header.lstrip("#").split()
    data = np.loadtxt(filename)

    t = data[:, 0]
    y = data[:, 1:]

    return t, y, labels[1:]



def plot_cvode(filename):
    t, y, labels = load_cvode_output(filename)

    plt.figure(figsize=(8, 5))

    for i in range(y.shape[1]):
        plt.plot(t, y[:, i], label=labels[i])

    plt.xlabel("Time")
    plt.ylabel("State variables")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_cvode("build/cvode_output.dat")
