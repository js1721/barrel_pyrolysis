"""
kinetics.py
===========
Point kinetics equations for amplitude evolution.

Implements Eqs. (93)-(94) of document:
    Lambda(t) dP/dt = (rho(t) - beta)*P(t)
                    + sum_j lambda_j * C_j(t)

    dC_j/dt = (beta_j/Lambda(t))*P(t) - lambda_j*C_j(t)

Solved via matrix exponential for exact ODE solution.
"""

import numpy as np
from scipy.linalg import expm

from materials import BETA_I, LAMBDA_I, BETA, I_GRP


def build_pk_matrix(rho: float,
                     Lambda: float) -> np.ndarray:
    """
    Build point kinetics matrix A of size (1+I, 1+I).

    d/dt [P, C1,...,CI]^T = A * [P, C1,...,CI]^T

    From Eqs. (93)-(94).
    """
    A = np.zeros((1 + I_GRP, 1 + I_GRP))

    # dP/dt row
    A[0, 0] = (rho - BETA) / Lambda
    for i in range(I_GRP):
        A[0, i+1] = LAMBDA_I[i]

    # dC_i/dt rows
    for i in range(I_GRP):
        A[i+1, 0]     =  BETA_I[i] / Lambda
        A[i+1, i+1]   = -LAMBDA_I[i]

    return A


def advance_kinetics(P: float,
                      C: np.ndarray,
                      rho: float,
                      Lambda: float,
                      dt: float) -> tuple:
    """
    Advance point kinetics by dt using matrix exponential.

    Exact solution of the linear ODE system.

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
    A         = build_pk_matrix(rho, Lambda)
    state     = np.concatenate([[P], C])
    state_new = expm(A * dt) @ state

    P_new = max(float(state_new[0]), 0.0)
    C_new = np.maximum(state_new[1:], 0.0)

    return P_new, C_new


def initialise_precursors(P0: float,
                            Lambda: float) -> np.ndarray:
    """
    Steady-state precursor concentrations.

    From Eq. (75):
        C_j(0) = beta_j / (lambda_j * Lambda(0))

    Parameters
    ----------
    P0     : initial amplitude
    Lambda : initial prompt neutron lifetime s

    Returns
    -------
    C : shape (I,)
    """
    return BETA_I / (LAMBDA_I * Lambda) * P0