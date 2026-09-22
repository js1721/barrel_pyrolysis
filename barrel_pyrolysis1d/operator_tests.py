"""
operator_tests.py
=================
Tests of the STOCHASTIC operators themselves, at mu != 0.

Tests 1-4 exercise the DOCUMENT closure (closure="document") against
the document's written slab equation -- they verify the implementation
of that equation, not its physics. Tests 5-7 exercise the RELAXATIONAL
closure (the solver default) against properties the realisation-
averaged benchmark (benchmark_markov.py) shows the physics must have.

verification_tests.py runs every case at mu = 0, which switches the
whole closure off: it verifies conduction and bare-slab neutronics but
cannot see the drift or coupling terms at all. Two sign errors (thermal
self drift, neutronic algebraic coupling) sat undetected behind that.
These tests exercise the closure directly.

Each discrete operator is recovered from the solver's own assembly and
compared against a reference written INDEPENDENTLY from the document's
active slab equation

    L_i = K_i T_i'' + 2 v_k K_i mu T_i' - v_k mu (K_i + K_k) T_k'
          + v_k mu^2 (K_i v_k + K_k v_i)(T_i - T_k)

(D, psi in place of K, T for neutronics).

  1. Constant mu, thermal:    matched stencils => machine precision.
  2. Constant mu, neutronics: matched stencils => machine precision.
  3. Equilibrium: T_1 = T_2 = const must give zero stochastic source,
     everywhere including the boundary cells.
  4. Varying mu(z): the operator must converge to the CONSERVATIVE
     continuum form, which carries the extra v_k K_i (T_i-T_k) mu'
     term. It must NOT converge to the pointwise form 2 v_k K_i mu T_i'
     (which the superseded code implemented and which is only valid
     for constant mu).

Run: python operator_tests.py
"""

import numpy as np

from mesh import Mesh1D
from materials import make_PuO, make_combustible
from thermal import (stochastic_self_operator, _stochastic_cross_source,
                      _banded_matvec, CM_TO_M)
import neutronics as nt

phases = [make_PuO(), make_combustible()]
v = np.array([0.7, 0.3])
H = 40.0
failures = []


def check(name, value, tol):
    ok = value < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {value:.3e}  (tol {tol:.0e})")
    if not ok:
        failures.append(name)


# ─────────────────────────────────────────────────────────────────────
def thermal_L(p, mesh, mu, T, closure="document"):
    """Full stochastic thermal operator for phase p, W/m^3, from the
    solver's own implicit matrix + explicit source."""
    k = 1 - p
    ph, po = phases[p], phases[k]
    N = mesh.N
    dz = mesh.dz * CM_TO_M
    mu_m = np.broadcast_to(np.atleast_1d(mu), (N,)).astype(float) / CM_TO_M
    L = stochastic_self_operator(ph, mesh, mu, v[k], closure)
    return (_banded_matvec(L, T[p])
            + _stochastic_cross_source(ph, po, T[p], T[k], mu_m, v[p], v[k], dz, closure))


def fields(z, Lz):
    T1 = 300 + 200 * np.exp(-z / (0.3 * Lz)) + 15 * np.sin(4 * np.pi * z / Lz)
    T2 = 300 + 140 * np.exp(-z / (0.22 * Lz)) + 10 * np.cos(3 * np.pi * z / Lz)
    return np.array([T1, T2])


# ── 1. constant mu, thermal ─────────────────────────────────────────
print("1. Thermal operator vs document slab equation (constant mu)")
N = 400
mesh = Mesh1D(H=H, N=N)
dz = mesh.dz * CM_TO_M
z = (np.arange(N) + 0.5) * dz
T = fields(z, H * CM_TO_M)
MU = 0.05
mu_m = MU / CM_TO_M

def fwd(f):
    d = np.zeros_like(f); d[:-1] = (f[1:] - f[:-1]) / dz; return d

def lap(f):
    d = np.zeros_like(f); d[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / dz**2; return d

for p in range(2):
    k = 1 - p
    Ki, Kk = phases[p].K, phases[k].K
    ref = (Ki * lap(T[p]) + 2 * v[k] * Ki * mu_m * fwd(T[p])
           - v[k] * mu_m * (Ki + Kk) * fwd(T[k])
           + v[k] * mu_m**2 * (Ki * v[k] + Kk * v[p]) * (T[p] - T[k]))
    got = thermal_L(p, mesh, MU, T)
    i = slice(1, N - 2)
    check(f"phase {p+1} rel. deviation", np.abs(got[i] - ref[i]).max()
          / np.abs(ref[i]).max(), 1e-10)

# ── 2. constant mu, neutronics ──────────────────────────────────────
print("\n2. Neutronic operator vs document slab equation (constant mu)")
Nn = 400
meshn = Mesh1D(H=H, N=Nn)
dzn = meshn.dz
zn = (np.arange(Nn) + 0.5) * dzn
psi = np.array([1.0 + 0.5 * np.exp(-zn / 5.0) + 0.1 * np.sin(3 * np.pi * zn / H),
                0.8 + 0.3 * np.exp(-zn / 4.0) + 0.05 * np.cos(2 * np.pi * zn / H)])
Dv = [1.2, 0.9]
MUn = 0.05

def cen(f):
    d = np.zeros_like(f); d[1:-1] = (f[2:] - f[:-2]) / (2 * dzn); return d

def lapn(f):
    d = np.zeros_like(f); d[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / dzn**2; return d

for p in range(2):
    k = 1 - p
    Di, Dk = Dv[p], Dv[k]
    A, C = nt._build_group_self_and_cross(Di, Dk, 0.0, meshn, v[p], v[k], MUn, p == 0, "document")
    got = -(A @ psi[p] + C @ psi[k])          # loss operator => L = -(...)
    ref = (Di * lapn(psi[p]) + 2 * v[k] * Di * MUn * cen(psi[p])
           - v[k] * MUn * (Di + Dk) * cen(psi[k])
           + v[k] * MUn**2 * (Di * v[k] + Dk * v[p]) * (psi[p] - psi[k]))
    i = slice(1, Nn - 1)
    check(f"phase {p+1} rel. deviation", np.abs(got[i] - ref[i]).max()
          / np.abs(ref[i]).max(), 1e-10)

# ── 3. equilibrium ───────────────────────────────────────────────────
print("\n3. Equilibrium: stochastic terms vanish for T_1 = T_2 = const")
Teq = np.full((2, N), 350.0)
mu_var = 0.02 + 0.06 * (z / z[-1])**2           # varying, to be strict
for p in range(2):
    got = thermal_L(p, mesh, mu_var * 1.0, Teq)  # conduction of const = 0
    # normalise by the size of the individual terms being cancelled
    # (the conduction stencil, K T / dz^2), so this measures cancellation
    # relative to what is being cancelled, i.e. machine precision
    scale = phases[p].K * 350.0 / dz**2
    check(f"thermal phase {p+1}, all cells incl. boundaries",
          np.abs(got).max() / scale, 1e-13)

psi_eq = np.full((2, Nn), 1.0)
for p in range(2):
    k = 1 - p
    A1, C1 = nt._build_group_self_and_cross(Dv[p], Dv[k], 0.0, meshn, v[p], v[k], 0.05, p == 0, "document")
    A0, C0 = nt._build_group_self_and_cross(Dv[p], Dv[k], 0.0, meshn, v[p], v[k], 0.0, p == 0, "document")
    stoch = (A1 - A0) @ psi_eq[p] + (C1 - C0) @ psi_eq[k]
    check(f"neutronic phase {p+1}, all cells incl. boundaries",
          np.abs(stoch).max(), 1e-12)

# ── 4. varying mu: conservative form, including mu' term ─────────────
print("\n4. Varying mu(z): converge to conservative form (with mu' term)")

def mu_of(zm):           # m^-1, smooth, strongly varying
    return (2.0 + 6.0 * (zm / (H * CM_TO_M))**2)

def exact_thermal(p, zm):
    """Continuum conservative operator, analytic derivatives."""
    import sympy as sp
    s = sp.Symbol('s')
    Lz = H * CM_TO_M
    T1 = 300 + 200 * sp.exp(-s / (0.3 * Lz)) + 15 * sp.sin(4 * sp.pi * s / Lz)
    T2 = 300 + 140 * sp.exp(-s / (0.22 * Lz)) + 10 * sp.cos(3 * sp.pi * s / Lz)
    Ts = [T1, T2]
    m = 2.0 + 6.0 * (s / Lz)**2
    k = 1 - p
    Ki, Kk = phases[p].K, phases[k].K
    cons = (Ki * sp.diff(Ts[p], s, 2)
            + v[k] * Ki * sp.diff(m * (Ts[p] - Ts[k]), s)
            + v[k] * Ki * m * sp.diff(Ts[p], s)
            - v[k] * Kk * m * sp.diff(Ts[k], s)
            + v[k] * (Ki * v[k] + Kk * v[p]) * m**2 * (Ts[p] - Ts[k]))
    point = (Ki * sp.diff(Ts[p], s, 2)
             + 2 * v[k] * Ki * m * sp.diff(Ts[p], s)
             - v[k] * m * (Ki + Kk) * sp.diff(Ts[k], s)
             + v[k] * (Ki * v[k] + Kk * v[p]) * m**2 * (Ts[p] - Ts[k]))
    fc = sp.lambdify(s, cons, 'numpy'); fp = sp.lambdify(s, point, 'numpy')
    return fc(zm), fp(zm)

print("      N    err vs conservative   err vs pointwise (superseded)")
errs = []
for Nv in (100, 200, 400, 800, 1600):
    mv = Mesh1D(H=H, N=Nv)
    dzv = mv.dz * CM_TO_M
    zv = (np.arange(Nv) + 0.5) * dzv
    Tv = fields(zv, H * CM_TO_M)
    got = thermal_L(0, mv, mu_of(zv) * CM_TO_M, Tv)   # mu passed in cm^-1
    ec, ep = exact_thermal(0, zv)
    i = slice(2, Nv - 2)
    scale = np.abs(ec[i]).max()
    e1 = np.abs(got[i] - ec[i]).max() / scale
    e2 = np.abs(got[i] - ep[i]).max() / scale
    errs.append(e1)
    print(f"  {Nv:5d}    {e1:.3e}             {e2:.3e}")
order = -np.polyfit(np.log([100, 200, 400, 800, 1600]), np.log(errs), 1)[0]
print(f"        observed order vs conservative form: {order:.2f}  (forward upwind => 1)")
check("convergence order to conservative form >= 0.8 (1 - order shown)",
      max(0.0, 1.0 - order), 0.2)
check("does NOT converge to pointwise form (residual stays O(1e-2+))",
      0.0 if e2 > 1e-3 else 1.0, 0.5)

# ── 5. relaxational thermal: equilibrium and the equilibration rate ──
print("\n5. Relaxational thermal closure: equilibrium + equilibration")
for p in range(2):
    got = thermal_L(p, mesh, mu_var, Teq, "relaxational")
    check(f"phase {p+1} annihilates T_1 = T_2 = const",
          np.abs(got).max() / (phases[p].K * 350.0 / dz**2), 1e-13)
from thermal import solve_thermal_step
N5 = 10
m5 = Mesh1D(H=H, N=N5)
MU5 = 0.5
T5 = np.array([np.full(N5, 310.0), np.full(N5, 300.0)])
rc = [q.rho * q.Cp for q in phases]
lam = (MU5 / CM_TO_M)**2 * (phases[0].K * v[1] + phases[1].K * v[0]) \
      * (v[1] / rc[0] + v[0] / rc[1])
dt5, nst = 0.05, 200
for _ in range(nst):
    T5 = solve_thermal_step(phases, m5, T5, np.zeros((2, N5)), np.zeros((2, 2, N5)),
                            v, MU5, dt5, T_f=300.0, h_conv=0.0, emissivity=0.0,
                            T_amb=None, closure="relaxational")
dT = (T5[0] - T5[1]).mean()
pred = 10.0 * np.exp(-lam * dt5 * nst)
check("isolated slab: T_1 - T_2 decays at rate lambda_c (rel. err.)",
      abs(dT - pred) / pred, 5e-3)
E0 = v[0] * rc[0] * 310.0 + v[1] * rc[1] * 300.0
E = v[0] * rc[0] * T5[0].mean() + v[1] * rc[1] * T5[1].mean()
check("isolated slab: total energy conserved (rel.)", abs(E - E0) / E0, 1e-10)

# ── 6. relaxational neutronics: joint eigenproblem is physical ───────
print("\n6. Relaxational neutronics: positive fundamental mode at all mu")
import warnings as _w
from neutronics import static_shape_solve
phase_pair = phases
m6 = Mesh1D(H=H, N=80)
T6 = np.full((2, 80), 300.0)
v6 = np.array([0.6744613390173657, 1 - 0.6744613390173657])
worst = 0.0
for mu6 in (0.0, 0.05, 0.5, 2.0, 10.0):
    with _w.catch_warnings():
        _w.simplefilter("error")                 # a sign change would raise
        k6, psi6 = static_shape_solve(phase_pair, m6, T6, v6, mu6,
                                      maxiter=20000, closure="relaxational")
    worst = max(worst, float(-psi6.min() / np.abs(psi6).max()))
check("no negative flux in either phase, mu in [0, 10]", max(worst, 0.0), 1e-12)

# ── 7. relaxational neutronics: fine-mixing (atomic-mix) limit ───────
print("\n7. Relaxational neutronics: psi_2 -> psi_1 as mu grows")
_, psi7 = static_shape_solve(phase_pair, m6, T6, v6, 10.0, maxiter=20000,
                             closure="relaxational")
mid = slice(30, 50)
check("thermal psi_2/psi_1 - 1 at mu = 10 cm^-1 (mid-slab)",
      abs(psi7[1, 1, mid].mean() / psi7[0, 1, mid].mean() - 1.0), 0.02)

# ── 8. pyrolysis energy conservation (isolated slab, fission off) ────
print("\n8. Pyrolysis: heat deposited equals chemical energy released")
from solver import Solver, Config
for dt8 in (0.05, 0.005):
    cfg8 = Config(H=40.0, N=40, mu=0.5, v1=0.17229402357524248, t_end=2.0, dt=dt8,
                  T0=560.0, T_f=560.0, h_conv=0.0, emissivity=0.0, T_amb=None,
                  P0=0.0, source_density=0.0, T_ceiling=1e9)
    s8 = Solver(cfg8); ph8, v8, dz8 = s8.phases, s8.v, s8.mesh.dz
    en = lambda st: (sum(v8[p] * ph8[p].rho * ph8[p].Cp * np.sum(st.T[p]) for p in range(2)) * dz8,
                     sum(v8[p] * ph8[p].q * np.sum(st.omega[p]) for p in range(2)) * dz8)
    st8 = s8.initialise(); a0, c0 = en(st8)
    while st8.t < 2.0 - 1e-9:
        st8 = s8.step(st8, dt8)
    a1, c1 = en(st8)
    check(f"dt = {dt8}: |heat gained / chemical released - 1|",
          abs((a1 - a0) / (c0 - c1) - 1.0), 1e-9)

print()
if failures:
    print("FAILED:", ", ".join(failures))
    raise SystemExit(1)
print("All operator tests passed.")
