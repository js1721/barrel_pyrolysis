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
                      dt: float,
                      q: float = 0.0) -> tuple:
    """
    Advance point kinetics by dt, EXACTLY, including an external source:

        dP/dt   = (rho - beta)/Lambda P + sum_j lambda_j C_j + q
        dC_j/dt = beta_j/Lambda P - lambda_j C_j

    q is the source in amplitude units per second (see
    neutronics.source_amplitude_rate), held constant over the step. The
    inhomogeneous system is solved with one matrix exponential of the
    augmented matrix [[A, b], [0, 0]], b = (q, 0, ..., 0), so the source
    is integrated exactly rather than by a quadrature. q = 0 reproduces
    the previous source-free update.
    """
    A = build_pk_matrix(rho, Lambda)
    n = A.shape[0]
    Aug = np.zeros((n + 1, n + 1))
    Aug[:n, :n] = A
    Aug[0, n] = q
    state = np.concatenate([[P], C, [1.0]])
    state_new = expm(Aug * dt) @ state
    P_new = max(float(state_new[0]), 0.0)
    C_new = np.maximum(state_new[1:n], 0.0)
    return P_new, C_new


def source_driven_steady_state(q: float, rho: float, Lambda: float) -> float:
    """
    Steady amplitude of a SUBCRITICAL system driven by source q:
    setting dP/dt = dC/dt = 0 gives sum_j lambda_j C_j = beta P/Lambda,
    hence 0 = rho P/Lambda + q, i.e.

        P_0 = -q Lambda / rho        (rho < 0).

    No steady state exists at or above critical: with a source, a
    critical system's power grows linearly and a supercritical one's
    exponentially.
    """
    if rho >= 0.0:
        raise ValueError(
            f"no source-driven steady state at rho = {rho:.4g} >= 0; start "
            f"from a subcritical composition, or set Config.P0 explicitly.")
    return -q * Lambda / rho


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