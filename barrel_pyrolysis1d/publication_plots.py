"""
publication_plots.py
=====================
Clean, single-purpose figures for the document -- one plot per file,
not the multi-panel diagnostic dashboards in main.py.

Runs the solver once with the current demo config and saves each
figure as its own PNG, ready to \\includegraphics directly.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from solver import Solver, Config
from materials import BETA

# ── Validated categorical palette (dataviz skill reference) ───────
BLUE   = "#2a78d6"   # series 1: PuO
ORANGE = "#eb6834"   # series 2: Combustible
AQUA   = "#1baf7a"   # series 3: volume-weighted combined
GRAY   = "#8a8a86"   # reference lines / secondary ink
INK    = "#0b0b0b"

plt.rcParams.update({
    "figure.figsize":    (6.0, 4.0),
    "figure.dpi":        200,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "font.size":         12,
    "axes.labelsize":    12,
    "axes.edgecolor":    INK,
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "grid.color":        "#d8d8d4",
    "grid.linewidth":    0.6,
    "legend.fontsize":   10,
    "legend.frameon":    False,
    "xtick.color":       INK,
    "ytick.color":       INK,
    "text.color":        INK,
    "axes.labelcolor":   INK,
    "lines.linewidth":   2.0,
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
    cfg = Config(
        H=40.0, N=80, mu=0.5, v1=0.17068927633801423,   # k0 = 0.98, source-driven -- see main.py
        t_end=300.0, dt=0.05,
        T0=300.0,
        T_f=1200.0, h_conv=50.0, emissivity=0.3, T_amb=300.0,
        Ef=3.2e-11,
    )
    solver  = Solver(cfg)
    history = solver.run()
    state   = solver.last_state
    z       = solver.mesh.z

    t     = np.array([h["t"]      for h in history])
    k_eff = np.array([h["k_eff"]  for h in history])
    rho   = np.array([h["rho"]    for h in history])
    P     = np.array([h["P"]      for h in history])
    T1    = np.array([h["T1_max"] for h in history])
    T2    = np.array([h["T2_max"] for h in history])

    # --- 1. Multiplication factor vs time -------------------------
    fig, ax = plt.subplots()
    ax.plot(t, k_eff, color=BLUE)
    ax.axhline(1.0, color=GRAY, ls="--", lw=1.2, label="Critical")
    ax.set(xlabel="$t$ (s)", ylabel=r"$k_\mathrm{eff}$")
    ax.legend()
    _save(fig, "fig_k_eff")

    # --- 2. Reactivity vs time --------------------------------------
    fig, ax = plt.subplots()
    ax.plot(t, rho * 1e5, color=BLUE, label=r"$\rho$")
    ax.axhline(BETA * 1e5, color=GRAY, ls="--", lw=1.2, label=r"$\beta$")
    ax.set(xlabel="$t$ (s)", ylabel=r"Reactivity ($\times 10^{-5}$)")
    ax.legend()
    _save(fig, "fig_reactivity")

    # --- 3. Neutron population vs time -------------------------------
    fig, ax = plt.subplots()
    ax.semilogy(t, P, color=BLUE)
    ax.set(xlabel="$t$ (s)", ylabel="$P(t)$")
    _save(fig, "fig_power")

    # --- 4. Temperature vs time (both phases, max) -------------------
    fig, ax = plt.subplots()
    ax.plot(t, T1, color=BLUE,   label="PuO")
    ax.plot(t, T2, color=ORANGE, label="Combustible")
    ax.axhline(cfg.T_f, color=GRAY, ls=":", lw=1.2, label="$T_f$")
    ax.set(xlabel="$t$ (s)", ylabel="$T$ (K)")
    ax.legend()
    _save(fig, "fig_temperature_time")

    # --- 5. Spatial temperature profile at final time -----------------
    T_mix = state.v[0] * state.T[0] + state.v[1] * state.T[1]
    fig, ax = plt.subplots()
    ax.plot(z, state.T[0], color=BLUE,   label="PuO")
    ax.plot(z, state.T[1], color=ORANGE, label="Combustible")
    ax.plot(z, T_mix,      color=AQUA,   ls="--",
            label=r"$v_1T_1+v_2T_2$")
    ax.axhline(cfg.T_f, color=GRAY, ls=":", lw=1.2, label="$T_f$")
    ax.set(xlabel="$z$ (cm)", ylabel="$T$ (K)")
    ax.legend()
    _save(fig, "fig_temperature_spatial")

    # --- 6a/6b. Spatial flux shape at final time, fast and thermal ----
    # psi is (2, 2, N) [phase, group, space]
    psi_mix = state.v[0] * state.psi[0] + state.v[1] * state.psi[1]   # (2, N) [group, space]

    fig, ax = plt.subplots()
    ax.plot(z, state.psi[0, 0], color=BLUE,   label=r"$\psi_{1,\mathrm{fast}}$ PuO")
    ax.plot(z, state.psi[1, 0], color=ORANGE, label=r"$\psi_{2,\mathrm{fast}}$ Combustible")
    ax.plot(z, psi_mix[0],      color=AQUA,   ls="--",
            label=r"$v_1\psi_{1,f}+v_2\psi_{2,f}$")
    ax.set(xlabel="$z$ (cm)", ylabel=r"$\psi_\mathrm{fast}(z)$")
    ax.legend()
    _save(fig, "fig_flux_spatial_fast")

    fig, ax = plt.subplots()
    ax.plot(z, state.psi[0, 1], color=BLUE,   label=r"$\psi_{1,\mathrm{th}}$ PuO")
    ax.plot(z, state.psi[1, 1], color=ORANGE, label=r"$\psi_{2,\mathrm{th}}$ Combustible")
    ax.plot(z, psi_mix[1],      color=AQUA,   ls="--",
            label=r"$v_1\psi_{1,th}+v_2\psi_{2,th}$")
    ax.set(xlabel="$z$ (cm)", ylabel=r"$\psi_\mathrm{thermal}(z)$")
    ax.legend()
    _save(fig, "fig_flux_spatial_thermal")

    # --- 7. Combustible content density, spatial, at final time ---------
    fig, ax = plt.subplots()
    ax.plot(z, state.omega[0], color=BLUE,   label=r"$\omega_1$ PuO")
    ax.plot(z, state.omega[1], color=ORANGE, label=r"$\omega_2$ Combustible")
    omega0_max = max(ph.omega0 for ph in solver.phases)
    ax.set(xlabel="$z$ (cm)", ylabel=r"$\omega(z)$ (kg/m$^3$)",
           ylim=[0, 1.05 * omega0_max])
    ax.legend()
    _save(fig, "fig_fuel_fraction")

    print(f"\nFinal state: t={state.t:.1f}s  k_eff={state.k_eff:.5f}  "
          f"rho={state.rho:.5f}")


if __name__ == "__main__":
    main()
