import numpy as np
from scipy.linalg import expm
from materials import DELAYED_GROUPS, BETA, I_GRP


def build_kinetics_matrix(rho: float,
                            Lambda: float) -> np.ndarray:
    """
    Build point kinetics matrix A of size (1+I, 1+I).

    From Eqs. (59)-(60) of document:
        d/dt [P, C1,...,CI]^T = A [P, C1,...,CI]^T
    """
    A = np.zeros((1 + I_GRP, 1 + I_GRP))

    # dP/dt row — Eq. (59)
    A[0, 0] = (rho - BETA) / Lambda
    for i, g in enumerate(DELAYED_GROUPS):
        A[0, i + 1] = g.lambda_i

    # dC_i/dt rows — Eq. (60)
    for i, g in enumerate(DELAYED_GROUPS):
        A[i + 1, 0]     =  g.beta_i / Lambda
        A[i + 1, i + 1] = -g.lambda_i

    return A


def advance_kinetics(P: float,
                      C: np.ndarray,
                      rho: float,
                      Lambda: float,
                      dt: float) -> tuple:
    """
    Advance point kinetics equations by dt using
    the matrix exponential — exact solution of the
    linear ODE system.

    Parameters
    ----------
    P      : current amplitude
    C      : precursor concentrations shape (I,)
    rho    : current reactivity
    Lambda : current prompt neutron lifetime s
    dt     : timestep s

    Returns
    -------
    P_new : float
    C_new : np.ndarray shape (I,)
    """
    A         = build_kinetics_matrix(rho, Lambda)
    state     = np.concatenate([[P], C])
    state_new = expm(A * dt) @ state

    P_new = max(float(state_new[0]), 0.0)
    C_new = np.maximum(state_new[1:], 0.0)

    return P_new, C_new


def initialise_precursors(P0: float,
                            Lambda: float) -> np.ndarray:
    """
    Initialise precursor concentrations at steady state.

    From Eq. (60) with dC/dt = 0:
        C_i = (beta_i / lambda_i / Lambda) * P0
    """
    return np.array([
        g.beta_i / (g.lambda_i * Lambda) * P0
        for g in DELAYED_GROUPS
    ])