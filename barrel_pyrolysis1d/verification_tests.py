"""
verification_tests.py
======================
Independent analytical verification of the two core solvers, each in
its fully decoupled (mu=0) limit where a closed-form solution exists.

1. Neutronics: bare-slab two-group k_eff (PuO alone, vacuum BCs),
   standard 2-group (fast=0, thermal=1, chi=[1,0], no upscatter)
   bare-reactor formula assuming both groups share one buckling B:
       k = [nuSf0 + nuSf1*Ss12/(D1*B^2+Sa1)] / (D0*B^2+Sa0+Ss12)
   compared against static_shape_solve() with v1=1.0 (isolates PuO's
   own eigenvalue k1 = k_eff, since k_eff = v1*k1 + v2*k2 and v2=0),
   across a mesh refinement sweep. The shared-B assumption is only
   exact when d_ext is negligible relative to H for BOTH groups; fast
   and thermal have quite different d_ext here (different removal
   cross sections), so this test uses a large H (2000cm) where that
   holds to <0.05% (checked directly: at H=40cm the same comparison is
   off by ~11%, purely from this shared-B approximation, not a solver
   bug -- see project history / keff_crosscheck-style residual check
   against the exact matrices, which confirms the solver itself
   reproduces its own system to ~1e-13).

2. Thermal: semi-infinite-solid transient conduction with a convective
   surface condition (Incropera & DeWitt, "plane wall, sudden change
   in surface condition" case), compared against solve_thermal_step()
   with mu=0 (decouples the two phases exactly), emissivity=0
   (isolates the pure-convective boundary term from the radiative
   one), Ef=0 (no fission heating), and PuO (k_arr=0, no pyrolysis
   heating) -- so the only physics left is 1D transient conduction
   with a convective BC, exactly matching the analytical case:

       (T(z,t)-T0)/(Tf-T0) = erfc(xi) - exp(a)*erfc(xi+eta)

   with xi = z/(2 sqrt(alpha t)), eta = h sqrt(alpha t)/K,
   a = h z/K + h^2 alpha t/K^2, alpha = K/(rho Cp). H is kept large
   enough that the diffusion depth sqrt(alpha t) stays well inside
   the slab over the test window, so the insulated far boundary at
   z=H never influences the near-surface solution being compared.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, erfcx

from mesh import Mesh1D
from materials import make_PuO, make_combustible
from neutronics import static_shape_solve
from thermal import solve_thermal_step

BLUE   = "#2a78d6"
ORANGE = "#eb6834"
GRAY   = "#8a8a86"
INK    = "#0b0b0b"

plt.rcParams.update({
    "figure.figsize":  (6.0, 4.0),
    "figure.dpi":       200,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "font.size":        12,
    "axes.labelsize":   12,
    "axes.edgecolor":   INK,
    "axes.linewidth":   0.8,
    "axes.grid":        True,
    "grid.color":       "#d8d8d4",
    "grid.linewidth":   0.6,
    "legend.fontsize":  10,
    "legend.frameon":   False,
    "lines.linewidth":  2.0,
})


FIGDIR = "figures"


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.tight_layout()
    path = f"{FIGDIR}/{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Test 1: bare-slab neutronics k_eff convergence ───────────────────

def neutronics_convergence(H=2000.0, N_values=(20, 40, 80, 160, 320, 640)):
    ph = make_PuO()
    D0, D1 = ph.D
    Sa0, Sa1 = ph.Sigma_a
    nuSf0, nuSf1 = ph.nu_Sf
    Ss12 = ph.Sigma_s12
    # Buckling from the THERMAL group's d_ext (dominant/reference
    # group, matching the single-group model's original convention);
    # see module docstring on why H must be large for this shared-B
    # formula to hold.
    # Milne extrapolation distance 0.7104*lambda_tr = 2.1312 D, the same
    # convention as neutronics._build_group_self_and_cross (the reference
    # must use the solver's boundary condition to test the solver).
    d_ext = 2.1312 * D1
    B = np.pi / (H + 2 * d_ext)
    B2 = B**2
    k_analytical = (nuSf0 + nuSf1 * Ss12 / (D1*B2 + Sa1)) / (D0*B2 + Sa0 + Ss12)

    phases = [make_PuO(), make_combustible()]
    v = np.array([1.0, 0.0])

    Ns, errs = [], []
    print(f"Analytical bare-slab k_eff = {k_analytical:.6f} "
          f"(d_ext={d_ext:.4f}cm, B={B:.6f}/cm)")
    for N in N_values:
        mesh = Mesh1D(H, N)
        T = np.full((2, N), 300.0)
        k, _ = static_shape_solve(phases, mesh, T, v, mu=0.0)
        err = abs(k - k_analytical) / k_analytical
        Ns.append(N)
        errs.append(err)
        print(f"  N={N:4d}: k_eff={k:.6f}  rel. error={err*100:.4f}%")

    Ns, errs = np.array(Ns, dtype=float), np.array(errs)
    # fitted convergence order: err ~ C * N^-p
    p, logC = np.polyfit(np.log(Ns), np.log(errs), 1)
    p = -p
    print(f"  Fitted convergence order: {p:.2f}")

    fig, ax = plt.subplots()
    ax.loglog(Ns, errs * 100, "o-", color=BLUE, label="Numerical")
    fit_line = np.exp(logC) * Ns**(-p) * 100
    ax.loglog(Ns, fit_line, "--", color=GRAY,
               label=fr"fit: $O(N^{{-{p:.2f}}})$")
    ax.set(xlabel="$N$ (mesh cells)",
           ylabel="Relative error in $k_\\mathrm{eff}$ (%)")
    ax.legend()
    _save(fig, "fig_verify_keff_convergence")
    return k_analytical, Ns, errs, p


# ── Test 2: semi-infinite slab, convective BC, transient conduction ──

def _analytical_T(z_m, t, T0, Tf, K, alpha, h):
    xi  = z_m / (2.0 * np.sqrt(alpha * t))
    eta = h * np.sqrt(alpha * t) / K
    a   = h * z_m / K + (h**2 * alpha * t) / K**2
    b   = xi + eta
    # exp(a)*erfc(b) rewritten via erfcx to avoid overflow:
    #   erfc(b) = erfcx(b)*exp(-b^2)  =>  exp(a)*erfc(b) = erfcx(b)*exp(a-b^2)
    correction = erfcx(b) * np.exp(a - b**2)
    return T0 + (Tf - T0) * (erfc(xi) - correction)


def thermal_profile_check(H=40.0, N=80, dt=0.05, t_test=300.0,
                           h_conv=50.0, T0=300.0, Tf=1200.0):
    phases = [make_PuO(), make_combustible()]
    ph = phases[0]
    alpha = (ph.K) / (ph.rho * ph.Cp)   # m^2/s

    mesh  = Mesh1D(H, N)
    T     = np.full((2, N), T0)
    omega = np.ones((2, N))
    phi   = np.zeros((2, 2, N))
    v     = np.array([1.0, 0.0])

    n_steps = int(round(t_test / dt))
    for _ in range(n_steps):
        T = solve_thermal_step(
            phases, mesh, T, omega, phi, v,
            mu=0.0, dt=dt, T_f=Tf, h_conv=h_conv,
            emissivity=0.0, Ef=0.0,
        )

    z_cm = mesh.z
    z_m  = z_cm * 1.0e-2
    T_analytical = _analytical_T(z_m, t_test, T0, Tf, ph.K, alpha, h_conv)

    max_err = np.max(np.abs(T[0] - T_analytical))
    print(f"Thermal profile check at t={t_test}s, N={N}: "
          f"max |T_num - T_analytical| = {max_err:.4f} K")

    fig, ax = plt.subplots()
    ax.plot(z_cm, T_analytical, color=GRAY, lw=3.0, label="Analytical")
    ax.plot(z_cm, T[0], "--", color=BLUE, label="Numerical")
    ax.set(xlabel="$z$ (cm)", ylabel="$T$ (K)",
           xlim=(0, min(H, 20.0)))
    ax.legend()
    _save(fig, "fig_verify_thermal_profile")
    return max_err


def thermal_convergence(H=40.0, N_values=(20, 40, 80, 160, 320),
                         dt=0.02, t_test=100.0, h_conv=50.0,
                         T0=300.0, Tf=1200.0):
    phases = [make_PuO(), make_combustible()]
    ph = phases[0]
    alpha = ph.K / (ph.rho * ph.Cp)
    v = np.array([1.0, 0.0])
    n_steps = int(round(t_test / dt))

    Ns, errs = [], []
    for N in N_values:
        mesh  = Mesh1D(H, N)
        T     = np.full((2, N), T0)
        omega = np.ones((2, N))
        phi   = np.zeros((2, 2, N))
        for _ in range(n_steps):
            T = solve_thermal_step(
                phases, mesh, T, omega, phi, v,
                mu=0.0, dt=dt, T_f=Tf, h_conv=h_conv,
                emissivity=0.0, Ef=0.0,
            )
        z_m = mesh.z * 1.0e-2
        T_analytical = _analytical_T(z_m, t_test, T0, Tf, ph.K, alpha, h_conv)
        err = np.max(np.abs(T[0] - T_analytical))
        Ns.append(N)
        errs.append(err)
        print(f"  N={N:4d}: max error = {err:.5f} K")

    Ns, errs = np.array(Ns, dtype=float), np.array(errs)
    p, logC = np.polyfit(np.log(Ns), np.log(errs), 1)
    p = -p
    print(f"  Fitted convergence order: {p:.2f}")

    fig, ax = plt.subplots()
    ax.loglog(Ns, errs, "o-", color=ORANGE, label="Numerical")
    fit_line = np.exp(logC) * Ns**(-p)
    ax.loglog(Ns, fit_line, "--", color=GRAY,
               label=fr"fit: $O(N^{{-{p:.2f}}})$")
    ax.set(xlabel="$N$ (mesh cells)",
           ylabel="Max $|T_\\mathrm{num}-T_\\mathrm{analytical}|$ (K)")
    ax.legend()
    _save(fig, "fig_verify_thermal_convergence")
    return Ns, errs, p


if __name__ == "__main__":
    print("=== Test 1: bare-slab neutronics k_eff ===")
    neutronics_convergence()
    print()
    print("=== Test 2: semi-infinite slab thermal conduction ===")
    thermal_profile_check()
    print()
    print("=== Test 2b: thermal convergence ===")
    thermal_convergence()
    print()
    # Tests 1-2b all run at mu = 0, which switches the stochastic closure
    # off entirely. Test 3 exercises the closure itself (mu != 0): it is
    # what catches sign or coefficient errors in the drift and coupling
    # terms, which Tests 1-2b cannot see.
    print("=== Test 3: stochastic operators at mu != 0 (operator_tests.py) ===")
    import os, runpy
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "operator_tests.py"), run_name="__main__")
