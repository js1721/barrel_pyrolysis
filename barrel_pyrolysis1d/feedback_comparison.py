"""
feedback_comparison.py
=======================
One-off comparison: the current critical demo config (main.py) run
with temperature feedback on the neutronics solve ON vs OFF
(Config.T_feedback). With it off, k_eff is pinned at its t=0 value for
the whole run (v does not evolve either, so nothing else can move it),
isolating point kinetics' pure exponential response from any Doppler /
1/v-thermal-averaging self-limiting feedback.
"""

import numpy as np
import matplotlib.pyplot as plt

from solver import Solver, Config
from materials import BETA

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
    fig.tight_layout()
    path = f"{FIGDIR}/{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    base = dict(
        H=40.0, N=80, mu=0.15, v1=0.5757877395611601,
        t_end=300.0, dt=0.05, T0=300.0, P0=1.0,
        T_f=1200.0, h_conv=50.0, emissivity=0.3, T_amb=300.0,
        Ef=3.2e-11,
    )

    print("=== With temperature feedback ===")
    solver_on = Solver(Config(**base, T_feedback=True))
    hist_on   = solver_on.run()

    print("\n=== Without temperature feedback ===")
    solver_off = Solver(Config(**base, T_feedback=False))
    hist_off   = solver_off.run()

    t_on   = np.array([h["t"] for h in hist_on])
    k_on   = np.array([h["k_eff"] for h in hist_on])
    P_on   = np.array([h["P"] for h in hist_on])

    t_off  = np.array([h["t"] for h in hist_off])
    k_off  = np.array([h["k_eff"] for h in hist_off])
    P_off  = np.array([h["P"] for h in hist_off])

    fig, ax = plt.subplots()
    ax.plot(t_on, k_on, color=BLUE, label="With T feedback")
    ax.plot(t_off, k_off, color=ORANGE, label="Without T feedback")
    ax.axhline(1.0, color=GRAY, ls="--", lw=1.2, label="Critical")
    ax.set(xlabel="$t$ (s)", ylabel=r"$k_\mathrm{eff}$")
    ax.legend()
    _save(fig, "fig_feedback_compare_keff")

    fig, ax = plt.subplots()
    ax.plot(t_on, P_on, color=BLUE, label="With T feedback")
    ax.plot(t_off, P_off, color=ORANGE, label="Without T feedback")
    ax.set(xlabel="$t$ (s)", ylabel="$P(t)$")
    ax.legend()
    _save(fig, "fig_feedback_compare_power")

    # --- Flux shape at final time, both cases -------------------------
    z = solver_on.mesh.z
    psi_on  = solver_on.last_state.psi
    psi_off = solver_off.last_state.psi

    fig, ax = plt.subplots()
    ax.plot(z, psi_on[0],  color=BLUE,   ls="-",  label=r"$\psi_1$ PuO, with feedback")
    ax.plot(z, psi_on[1],  color=ORANGE, ls="-",  label=r"$\psi_2$ Combustible, with feedback")
    ax.plot(z, psi_off[0], color=BLUE,   ls="--", label=r"$\psi_1$ PuO, without feedback")
    ax.plot(z, psi_off[1], color=ORANGE, ls="--", label=r"$\psi_2$ Combustible, without feedback")
    ax.set(xlabel="$z$ (cm)", ylabel=r"$\psi(z)$",
           title=f"Flux shape at t={solver_on.last_state.t:.0f}s")
    ax.legend(fontsize=9)
    _save(fig, "fig_feedback_compare_flux")

    print(f"\nWith feedback:    k_eff(0)={k_on[0]:.6f}  k_eff(end)={k_on[-1]:.6f}  "
          f"P(end)={P_on[-1]:.4f}")
    print(f"Without feedback: k_eff(0)={k_off[0]:.6f}  k_eff(end)={k_off[-1]:.6f}  "
          f"P(end)={P_off[-1]:.4f}")


if __name__ == "__main__":
    main()
