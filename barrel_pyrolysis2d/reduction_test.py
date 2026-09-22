"""
Reduction test for the 2D stochastic thermal operator -- DOCUMENT closure.
(The solver default is the relaxational closure; this file verifies the
implementation of the document's equation, not its physics.)

DECISIVE CHECK: with Nr = 1, mu_r = 0 and constant mu_z, the operator
assembled by thermal.py must reproduce the document's Section 5 slab
equation

    K_i T_i'' + 2 v_k K_i mu T_i'
              - v_k mu (K_i + K_k) T_k'
              + v_k mu^2 (K_i v_k + K_k v_i)(T_i - T_k)

The reference below is written out directly from that equation, NOT
taken from any existing code, so this tests the derivation rather than
self-consistency. It is applied to a smooth manufactured field and
compared against the discrete operator recovered from thermal.py's own
matrix and explicit source.

Also checks:
  - |m|^2 = mu_r^2 + mu_z^2, i.e. the algebraic coupling at
    mu_r = mu_z = mu is TWICE the slab value, not the slab value.
  - The operator annihilates a uniform field (T_1 = T_2 = const), which
    it must: no drift, no coupling, no conduction in equilibrium.
"""

import numpy as np

from mesh import CylindricalMesh2D
from materials import make_PuO, make_combustible
from thermal import (build_self_operator, grad_upwind, div_T_m,
                      CM_TO_M, _as_field)

np.set_printoptions(precision=6, suppress=True)

phases = [make_PuO(), make_combustible()]
v = np.array([0.7, 0.3])
H, Nz = 40.0, 200
MU = 0.05           # cm^-1


def discrete_operator(mesh, mu_r, mu_z, T, dt=1.0):
    """
    Recover L[T] (units K/s) from thermal.py's own assembly:
    the matrix is (I - dt/(rCp) L_self), so
        L_self T = rCp/dt * (T - A T)
    and the cross terms come from the same expressions used in
    solve_thermal_step.
    """
    Nr, Nzl = mesh.Nr, mesh.Nz
    dr = mesh.dr * CM_TO_M
    dz = mesh.dz * CM_TO_M
    mr = _as_field(mu_r, (Nr, Nzl)) / CM_TO_M
    mz = _as_field(mu_z, (Nr, Nzl)) / CM_TO_M
    m2 = mr**2 + mz**2

    out = []
    for p, ph in enumerate(phases):
        k = 1 - p
        ph_o = phases[k]
        rCp = ph.rho * ph.Cp
        A = build_self_operator(ph, mesh, dt, mu_r, mu_z,
                                 v[p], v[k], ph_o.K, None, "document")
        self_part = rCp / dt * (T[p].ravel() - A.dot(T[p].ravel()))
        self_part = self_part.reshape(Nr, Nzl)

        dTk_dr, dTk_dz = grad_upwind(T[k], dr, dz)
        cross = (-v[k] * ph.K * div_T_m(T[k], mr, mz, mesh, dr, dz)
                 - v[k] * ph_o.K * (mr * dTk_dr + mz * dTk_dz)
                 + v[k] * (ph.K * v[k] + ph_o.K * v[p]) * m2 * (T[p] - T[k]))
        out.append(self_part + cross)
    return np.array(out)


def slab_reference(T, dz_m, mu_cm):
    """
    The document's written slab equation, discretised independently:
    central second difference, backward first differences (matching the
    operator's upwinding). Returns K/s-equivalent RHS (before /rho Cp).
    """
    mu = mu_cm / CM_TO_M
    out = []
    for p, ph in enumerate(phases):
        k = 1 - p
        ph_o = phases[k]
        Ti, Tk = T[p][0], T[k][0]

        d2 = np.zeros_like(Ti)
        d2[1:-1] = (Ti[2:] - 2 * Ti[1:-1] + Ti[:-2]) / dz_m**2

        # forward upwind, matching the operator (positive drift
        # coefficient => transport toward decreasing z)
        dTi = np.zeros_like(Ti); dTi[:-1] = (Ti[1:] - Ti[:-1]) / dz_m
        dTk = np.zeros_like(Tk); dTk[:-1] = (Tk[1:] - Tk[:-1]) / dz_m

        rhs = (ph.K * d2
               + 2 * v[k] * ph.K * mu * dTi
               - v[k] * mu * (ph.K + ph_o.K) * dTk
               + v[k] * mu**2 * (ph.K * v[k] + ph_o.K * v[p]) * (Ti - Tk))
        out.append(rhs[None, :])
    return np.array(out)


# ── Test 1: slab reduction ───────────────────────────────────────────
mesh1 = CylindricalMesh2D(R=1.0, H=H, Nr=1, Nz=Nz)
z = mesh1.z * CM_TO_M
T = np.array([
    (300.0 + 200.0 * np.exp(-z / 0.12) + 15.0 * np.sin(4 * np.pi * z / (H * CM_TO_M)))[None, :],
    (300.0 + 140.0 * np.exp(-z / 0.09) + 10.0 * np.cos(3 * np.pi * z / (H * CM_TO_M)))[None, :],
])

L_code = discrete_operator(mesh1, 0.0, MU, T)
L_ref = slab_reference(T, mesh1.dz * CM_TO_M, MU)

interior = slice(1, Nz - 1)
for p in range(2):
    a = L_code[p][0][interior]
    b = L_ref[p][0][interior]
    rel = np.abs(a - b).max() / max(np.abs(b).max(), 1e-30)
    print(f"phase {p+1}: max relative deviation from slab equation = {rel:.3e}")
    assert rel < 1e-10, "2D operator does NOT reduce to the slab equation"

# ── Test 2: |m|^2 doubling ───────────────────────────────────────────
Tc = np.array([np.full((1, Nz), 500.0), np.full((1, Nz), 300.0)])
L_ax = discrete_operator(mesh1, 0.0, MU, Tc)      # axial only
L_iso = discrete_operator(mesh1, MU, MU, Tc)      # mu_r = mu_z
# with a uniform field only the algebraic term survives
ratio = L_iso[0][0][Nz // 2] / L_ax[0][0][Nz // 2]
print(f"\nalgebraic coupling ratio (mu_r=mu_z) / (mu_r=0) = {ratio:.6f}"
      "   [expect 2.0: |m|^2 = mu_r^2 + mu_z^2]")
assert abs(ratio - 2.0) < 1e-10

# ── Test 3: equilibrium annihilation ─────────────────────────────────
mesh2 = CylindricalMesh2D(R=20.0, H=H, Nr=12, Nz=40)
Teq = np.array([np.full((12, 40), 350.0), np.full((12, 40), 350.0)])
L_eq = discrete_operator(mesh2, MU, MU, Teq)
print(f"max |L[T_1=T_2=const]| = {np.abs(L_eq).max():.3e}   [expect ~0]")
assert np.abs(L_eq).max() < 1e-6, \
    "operator does not annihilate thermal equilibrium"

print("\nAll operator reduction checks passed.")
