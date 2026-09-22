"""
percolation_demo.py
====================
Demonstrates the moderator-percolation submodel (percolation.py): melting,
gravity drainage, and flash-off of a mobile liquid component within the
combustible phase, and the resulting dynamic correlation length mu(z,t)
fed back into the neutronics/thermal solve in place of a constant mu.

Baseline config reuses solver.py's own defaults (mu=0.3, v1=0.35 --
PercolationConfig.ell0's default of 1/0.3 was explicitly chosen to match
this, per its own comment) at v1=0.545 for a moderate, safely-stable
composition (see main.py's config history for the v1 instability region
to avoid, roughly [0.83, 0.98]). This is NOT tuned to k_eff(0)=1 --
demonstrating percolation's effect on reactivity is the point here, not
a critical-run demo (see main.py for that).

Percolation parameters are illustrative (same "physically reasonable
order of magnitude, not fit to a library" convention as the rest of this
project's material data -- see materials.py's alpha_D comments), reusing
percolation.py's own self-test values directly.

NOTE on performance: many small (2N x 2N, N~80) LU factorisations in a
tight loop are dominated by BLAS thread-spawn overhead if left to the
default multi-threaded backend -- capping to 1 thread (must happen
before numpy is imported) turns a run that never finishes in any
practical time into one that completes in well under a minute.
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib.pyplot as plt

from solver import Solver, Config
from percolation import PercolationConfig, mu_field, correlation_length

BLUE   = "#2a78d6"
ORANGE = "#eb6834"
AQUA   = "#1baf7a"
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


def main():
    perc = PercolationConfig(
        enabled=True, theta_r=0.02, theta_s=0.30,
        T_melt=400.0, T_boil=900.0, k_melt=0.05, k_flash=0.02,
        L_fus=2.0e5, L_vap=8.0e5, rho_liquid=900.0,
        K_sat=0.05, n_perc=3.0,
        ell0=1.0 / 0.3, ell_max=15.0, theta_c=0.25, s0_perc=1.0, p_perc=4.0,
    )
    base = dict(
        H=40.0, N=80, mu=0.3, v1=0.545,
        # t_end=200, not 300: the per-phase fixed-point solve's inner
        # power iteration is known to become unreliable when the 2N
        # system develops near-degenerate eigenvalues (see main.py's
        # v1-instability comment for the same failure mode triggered a
        # different way, via v1 alone at constant mu). Here a spatially-
        # varying mu field from percolation triggers it too, once the
        # melt front develops enough structure -- confirmed by direct
        # inspection: k_eff is smooth and monotonic through t=200s,
        # then plunges discontinuously (0.621->0.615->0.566->0.467 over
        # four consecutive steps at t=220-224s) into chaotic bouncing
        # between ~0.46 and ~0.81 for the remainder of a 300s run. Not
        # a percolation.py bug -- the SAME known neutronics solver gap,
        # newly triggered by this mechanism. Truncating here keeps the
        # demo honest rather than showing a result that looks like
        # real physics but isn't.
        t_end=200.0, dt=0.05, T0=300.0, P0=1.0,
        T_f=1200.0, h_conv=50.0, emissivity=0.3, T_amb=300.0,
        Ef=3.2e-11,
    )

    print("=== With percolation ===")
    solver_on = Solver(Config(**base, percolation=perc))
    hist_on   = solver_on.run()

    print("\n=== Without percolation (constant mu=0.3) ===")
    solver_off = Solver(Config(**base, percolation=PercolationConfig(enabled=False)))
    hist_off   = solver_off.run()

    t_on    = np.array([h["t"] for h in hist_on])
    k_on    = np.array([h["k_eff"] for h in hist_on])
    P_on    = np.array([h["P"] for h in hist_on])
    T2_on   = np.array([h["T2_max"] for h in hist_on])
    thmax_on = np.array([h["theta_max"] for h in hist_on])
    thmean_on = np.array([h["theta_mean"] for h in hist_on])

    t_off   = np.array([h["t"] for h in hist_off])
    k_off   = np.array([h["k_eff"] for h in hist_off])
    P_off   = np.array([h["P"] for h in hist_off])
    T2_off  = np.array([h["T2_max"] for h in hist_off])

    # --- 1. Liquid saturation theta(t) -------------------------------
    fig, ax = plt.subplots()
    ax.plot(t_on, thmax_on, color=BLUE, label=r"$\theta_{\max}$")
    ax.plot(t_on, thmean_on, color=ORANGE, label=r"$\bar\theta$")
    ax.axhline(perc.theta_s, color=GRAY, ls="--", lw=1.2, label=r"$\theta_s$ (fully melted)")
    ax.axhline(perc.theta_r, color=GRAY, ls=":", lw=1.2, label=r"$\theta_r$ (irreducible)")
    ax.set(xlabel="$t$ (s)", ylabel=r"$\theta$ (liquid saturation)")
    ax.legend()
    _save(fig, "fig_percolation_theta_time")

    # --- 2. Spatial theta and mu profile at final time -----------------
    z = solver_on.mesh.z
    theta_final = solver_on.last_state.theta
    mu_final = mu_field(theta_final, perc)
    fig, ax1 = plt.subplots()
    ax1.plot(z, theta_final, color=BLUE, label=r"$\theta(z)$")
    ax1.set(xlabel="$z$ (cm)", ylabel=r"$\theta(z)$", ylim=[0, perc.theta_s * 1.1])
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax2 = ax1.twinx()
    ax2.plot(z, mu_final, color=ORANGE, label=r"$\mu(z)$")
    ax2.set_ylabel(r"$\mu(z)$ (cm$^{-1}$)", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.axhline(1.0 / perc.ell_max, color=ORANGE, ls=":", lw=1.0)
    fig.suptitle(f"Liquid saturation and correlation-length field at t={solver_on.last_state.t:.0f}s")
    _save(fig, "fig_percolation_theta_mu_spatial")

    # --- 3. k_eff comparison -------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(t_on, k_on, color=BLUE, label="With percolation")
    ax.plot(t_off, k_off, color=ORANGE, label="Without percolation (dry, $\\mu$=0.3)")
    ax.set(xlabel="$t$ (s)", ylabel=r"$k_\mathrm{eff}$")
    ax.legend()
    _save(fig, "fig_percolation_compare_keff")

    # --- 4. Power comparison ---------------------------------------------
    fig, ax = plt.subplots()
    ax.semilogy(t_on, P_on, color=BLUE, label="With percolation")
    ax.semilogy(t_off, P_off, color=ORANGE, label="Without percolation")
    ax.set(xlabel="$t$ (s)", ylabel="$P(t)$")
    ax.legend()
    _save(fig, "fig_percolation_compare_power")

    # --- 5. Temperature comparison (combustible phase max) -------------
    fig, ax = plt.subplots()
    ax.plot(t_on, T2_on, color=BLUE, label="With percolation")
    ax.plot(t_off, T2_off, color=ORANGE, label="Without percolation")
    ax.axhline(perc.T_melt, color=GRAY, ls=":", lw=1.0, label=r"$T_\mathrm{melt}$")
    ax.axhline(perc.T_boil, color=GRAY, ls="--", lw=1.0, label=r"$T_\mathrm{boil}$")
    ax.set(xlabel="$t$ (s)", ylabel="Combustible $T_\\mathrm{max}$ (K)")
    ax.legend(fontsize=9)
    _save(fig, "fig_percolation_compare_temperature")

    # --- 6. Correlation length constitutive law -------------------------
    theta_sweep = np.linspace(perc.theta_r, perc.theta_s, 200)
    ell_sweep = correlation_length(theta_sweep, perc)
    fig, ax = plt.subplots()
    ax.plot(theta_sweep, ell_sweep, color=AQUA)
    ax.axvline(perc.theta_c, color=GRAY, ls="--", lw=1.0, label=r"$\theta_c$ (percolation threshold)")
    ax.set(xlabel=r"$\theta$ (liquid saturation)", ylabel=r"$\ell(\theta)$ (cm)")
    ax.legend()
    _save(fig, "fig_percolation_correlation_law")

    print(f"\nWith percolation:    k_eff(0)={k_on[0]:.4f}  k_eff(end)={k_on[-1]:.4f}  "
          f"theta_max(end)={thmax_on[-1]:.4f}")
    print(f"Without percolation: k_eff(0)={k_off[0]:.4f}  k_eff(end)={k_off[-1]:.4f}")


if __name__ == "__main__":
    main()
