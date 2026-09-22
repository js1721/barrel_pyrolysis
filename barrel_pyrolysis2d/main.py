"""
main.py  (2D axisymmetric r-z)
==============================
Furnace transient of a 200-litre PCM drum (R = 28.6 cm, H = 88 cm).

Starting point: a subcritical STORED drum, k_eff(0) = 0.98, whose
initial power is the source-driven steady state of the reactor-grade
Pu intrinsic neutron source (no power is chosen). v1 = 0.1599 is the
fuel-lean root of k_eff = 0.98 at mu = 0.5 cm^-1; this drum's k peaks
near 1.22 around v1 ~ 0.5 (optimum moderation). A smaller 20 x 40 cm
drum cannot reach 0.98 at any composition (k peaks ~0.93): radial
leakage dominates.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver import Solver, Config
from benchmark_markov2d import safety_report_2d, print_safety_report_2d

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def plot(history, solver):
    os.makedirs(FIGDIR, exist_ok=True)
    t = np.array([h["t"] for h in history])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(t, [h["k_eff"] for h in history]); ax[0].set(xlabel="t (s)", ylabel="$k_{eff}$")
    ax[1].semilogy(t, [h["P"] for h in history]); ax[1].set(xlabel="t (s)", ylabel="P (n cm/s)")
    ax[2].plot(t, [h["T_fuel_mean"] for h in history], label="fuel (mean)")
    ax[2].plot(t, [h["T_mod_mean"] for h in history], label="combustible (mean)")
    ax[2].plot(t, [h["T_max"] for h in history], "k:", label="max")
    ax[2].set(xlabel="t (s)", ylabel="T (K)"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig2d_timeseries.png"), dpi=150)

    st, m = solver.last_state, solver.mesh
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for a, (fld, title) in zip(ax, [(st.T[0], "fuel T (K)"), (st.T[1], "combustible T (K)"),
                                    (st.psi[0, 1] * st.P, "fuel thermal flux (n/cm$^2$/s)")]):
        pc = a.pcolormesh(m.r, m.z, fld.T, shading="auto")
        fig.colorbar(pc, ax=a); a.set(xlabel="r (cm)", ylabel="z (cm)", title=f"{title}, t={st.t:.0f}s")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig2d_fields.png"), dpi=150)
    print(f"Saved: {FIGDIR}/fig2d_timeseries.png, fig2d_fields.png")


if __name__ == "__main__":
    cfg = Config(R=28.6, H=88.0, Nr=22, Nz=64, mu=0.5, v1=0.15988287397030493,
                 t_end=300.0, dt=1.0, T0=300.0, T_f=1200.0, h_conv=50.0,
                 emissivity=0.3, T_amb=300.0)
    solver = Solver(cfg)
    history = solver.run()
    h0, h1 = history[0], history[-1]
    print(f"\nk_eff {h0['k_eff']:.5f} -> {h1['k_eff']:.5f};  P {h0['P']:.3e} -> {h1['P']:.3e} "
          f"(x{h1['P']/h0['P']:.2f});  T_max {h1['T_max']:.1f} K")

    # See benchmark_markov2d.py's module docstring: this closed-form
    # model's k_eff is a biased point estimate, and the realisation
    # ensemble below is itself only a partial (radially-independent)
    # benchmark -- likely a lower bound on the true clumping risk.
    rep = safety_report_2d(cfg.mu, cfg.v1, solver.mesh, model_k=h0["k_eff"])
    print_safety_report_2d(rep)

    plot(history, solver)
