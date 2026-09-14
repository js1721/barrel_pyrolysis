import numpy as np


def coupling_coefficient_r(v1: float,
                             v2: float,
                             mu_r: float) -> float:
    """
    Radial stochastic coupling coefficient.

    From Eq. (65) of document:
        nabla R|_{r=0} = -mu_r * r_hat - mu_z * z_hat

    G_r = -v1*v2*R'_r(0) = v1*v2*mu_r
    """
    return v1 * v2 * mu_r


def coupling_coefficient_z(v1: float,
                             v2: float,
                             mu_z: float) -> float:
    """
    Axial stochastic coupling coefficient.
    G_z = -v1*v2*R'_z(0) = v1*v2*mu_z
    """
    return v1 * v2 * mu_z