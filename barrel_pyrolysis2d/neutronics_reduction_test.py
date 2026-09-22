"""
neutronics_reduction_test.py
============================
2D neutronics checks against the verified 1D solver.

1. Nr = 1, reflective drum wall, mu_r = 0: the r-z problem has no radial
   structure and must reproduce the 1D (relaxational, joint-eigenvalue)
   k_eff, the flux ratio psi_2/psi_1, the generation time Lambda, and
   the ABSOLUTE source-driven flux phi = P psi -- the last is a check
   on units, since the two codes normalise psi differently (per length
   in 1D, per volume in 2D) and only a consistent treatment of the
   source and amplitude gives the same physical flux.
2. The vacuum drum wall only removes neutrons: k(vacuum) < k(reflective).
3. The fundamental mode is positive in both phases across mu.
"""

import importlib.util
import os
import sys
import warnings

import numpy as np

from mesh import CylindricalMesh2D
import materials as m2
import neutronics as n2

HERE = os.path.dirname(os.path.abspath(__file__))
ONE_D = os.path.join(os.path.dirname(HERE), "barrel_pyrolysis1d")


def load_1d(name):
    spec = importlib.util.spec_from_file_location(name + "_1d",
                                                  os.path.join(ONE_D, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name + "_1d"] = mod
    spec.loader.exec_module(mod)
    return mod


# The 1D neutronics imports `materials` and `mesh` by name; point those
# names at the 1D modules while it loads, then restore the 2D ones.
saved = {k: sys.modules.get(k) for k in ("materials", "mesh")}
sys.modules["materials"] = load_1d("materials")
sys.modules["mesh"] = load_1d("mesh")
n1 = load_1d("neutronics")
Mesh1D = sys.modules["mesh"].Mesh1D
ph1 = [sys.modules["materials"].make_PuO(), sys.modules["materials"].make_combustible()]
for k, mod in saved.items():
    sys.modules[k] = mod

failures = []


def check(name, value, tol):
    ok = value < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {value:.3e}  (tol {tol:.0e})")
    if not ok:
        failures.append(name)


ph2 = [m2.make_PuO(), m2.make_combustible()]
H, Nz = 40.0, 80
v = np.array([0.17229402357524248, 1 - 0.17229402357524248])
Q = 6.058e3                      # n/s/cm^3 of fuel, reactor grade

print("1. Nr = 1, reflective wall, mu_r = 0  vs  1D")
for mu in (0.0, 0.12, 0.5, 2.0):
    mesh1 = Mesh1D(H=H, N=Nz)
    T1 = np.full((2, Nz), 300.0)
    k1, psi1 = n1.static_shape_solve(ph1, mesh1, T1, v, mu, closure="relaxational")
    L1 = n1.compute_Lambda(ph1, mesh1, psi1, T1, v)
    q1 = n1.source_amplitude_rate(ph1, mesh1, psi1, v, Q)

    mesh2 = CylindricalMesh2D(R=10.0, H=H, Nr=1, Nz=Nz)
    T2 = np.full((2, 1, Nz), 300.0)
    k2, psi2 = n2.static_shape_solve(ph2, mesh2, T2, v, 0.0, mu, radial_bc="reflective")
    L2 = n2.compute_Lambda(ph2, mesh2, psi2, T2, v)
    q2 = n2.source_amplitude_rate(ph2, mesh2, psi2, v, Q)

    print(f"   mu = {mu}")
    check("k_eff rel. diff", abs(k2 - k1) / k1, 1e-9)
    if mu == 0.0:
        # decoupled: the non-fissile phase carries no flux in either code
        check("psi_2 identically zero at mu = 0 (2D)", float(np.abs(psi2[1]).max()), 1e-12)
    else:
        r1 = psi1[1, 1] / psi1[0, 1]
        r2 = psi2[1, 1, 0] / psi2[0, 1, 0]
        check("psi2/psi1 (thermal) max rel. diff", np.abs(r2 - r1).max() / np.abs(r1).max(), 1e-8)
    check("Lambda rel. diff", abs(L2 - L1) / L1, 1e-9)
    if k1 < 1.0:
        rho = (k1 - 1.0) / k1
        phi1 = (-q1 * L1 / rho) * psi1[0, 1]
        phi2 = (-q2 * L2 / rho) * psi2[0, 1, 0]
        check("absolute source-driven thermal flux rel. diff",
              np.abs(phi2 - phi1).max() / np.abs(phi1).max(), 1e-8)

print("\n2. Vacuum drum wall removes neutrons")
mesh = CylindricalMesh2D(R=20.0, H=H, Nr=20, Nz=40)
T = np.full((2, 20, 40), 300.0)
kv, _ = n2.static_shape_solve(ph2, mesh, T, v, 0.5, 0.5, radial_bc="vacuum")
kr, _ = n2.static_shape_solve(ph2, mesh, T, v, 0.5, 0.5, radial_bc="reflective")
print(f"   k(vacuum wall) = {kv:.5f},  k(reflective wall) = {kr:.5f}")
check("k(vacuum) < k(reflective)  [0 means yes]", 0.0 if kv < kr else 1.0, 0.5)

print("\n3. Positive fundamental mode, R = 20 cm drum")
worst = 0.0
for mu in (0.05, 0.5, 2.0, 10.0):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, psi = n2.static_shape_solve(ph2, mesh, T, v, mu, mu, radial_bc="vacuum")
    worst = max(worst, float(-psi.min() / np.abs(psi).max()))
check("no negative flux, mu in [0.05, 10]", max(worst, 0.0), 1e-12)

print()
if failures:
    print("FAILED:", ", ".join(failures))
    raise SystemExit(1)
print("All 2D neutronics checks passed.")
