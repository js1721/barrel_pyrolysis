"""
keff_crosscheck.py
===================
Checks the one open question about the stochastic model's k_eff: is
the per-phase combination

    k_eff = v1*k1 + v2*k2

(static_shape_solve's convention, neutronics.py:364) a good proxy for
the ACTUAL dominant eigenvalue of the full coupled 2N system Eq. 89
represents? _build_self_and_cross already builds each phase's A_self /
Cross / F blocks straight from Eq. 89 -- no new derivation is needed,
just assembling them into one joint system:

    [A_self1  Cross1 ] [psi1]         [F1  0] [psi1]
    [Cross2   A_self2] [psi2] = (1/k) [0   0] [psi2]

and solving it with the same power-iteration pattern already used for
the per-phase solve, as the "gold standard" this model's own equations
imply. F2=0 (combustible has no fission), so this parallels
_solve_fissile_phase exactly, just on the stacked 2N system.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from mesh import Mesh1D
from materials import make_PuO, make_combustible
from neutronics import _build_self_and_cross, static_shape_solve


def joint_k_eff(phases, mesh, T, v, mu, maxiter=2000, tol=1e-12):
    N = mesh.N
    A1, Cross1, F1, _ = _build_self_and_cross(
        phases[0], phases[1], mesh, float(np.mean(T[0])),
        v[0], v[1], mu, is_phase0=True)
    A2, Cross2, F2, _ = _build_self_and_cross(
        phases[1], phases[0], mesh, float(np.mean(T[1])),
        v[1], v[0], mu, is_phase0=False)

    M = np.zeros((2 * N, 2 * N))
    M[:N, :N] = A1
    M[:N, N:] = Cross1
    M[N:, N:] = A2
    M[N:, :N] = Cross2

    Fb = np.zeros((2 * N, 2 * N))
    Fb[:N, :N] = F1   # F2 == 0 identically (combustible: no fission)

    lu_piv = lu_factor(M)
    psi = np.full(2 * N, 1.0 / N)
    k = 1.0
    for _ in range(maxiter):
        fission_old = Fb @ psi
        rhs = fission_old / k
        psi_new = lu_solve(lu_piv, rhs)
        fission_new = Fb @ psi_new
        denom = np.sum(fission_old)
        k_new = k * (np.sum(fission_new) / denom) if abs(denom) > 1e-300 else k
        scale = np.max(np.abs(psi_new))
        if scale > 1e-300:
            psi_new = psi_new / scale
        converged = abs(k_new - k) < tol * max(abs(k), 1e-30)
        psi, k = psi_new, k_new
        if converged:
            break

    return k, psi.reshape(2, N)


def compare(H=40.0, N=80, mu_values=(0.0, 0.01, 0.05, 0.1, 0.2, 0.5),
            v1=0.545, T0=300.0):
    phases = [make_PuO(), make_combustible()]
    v = np.array([v1, 1.0 - v1])
    mesh = Mesh1D(H, N)
    T = np.full((2, N), T0)

    print(f"{'mu':>6} | {'k_eff (v1*k1+v2*k2)':>20} | {'k_eff (joint 2N)':>18} | {'rel. diff':>10}")
    print("-" * 66)
    for mu in mu_values:
        k_combo, _ = static_shape_solve(phases, mesh, T, v, mu)
        k_joint, _ = joint_k_eff(phases, mesh, T, v, mu)
        rel = abs(k_combo - k_joint) / abs(k_joint) if k_joint != 0 else float("nan")
        print(f"{mu:6.3f} | {k_combo:20.6f} | {k_joint:18.6f} | {rel*100:9.3f}%")


if __name__ == "__main__":
    compare()
