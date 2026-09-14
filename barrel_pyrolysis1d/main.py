"""
main.py
=======
Entry point and plotting — 1D slab, two energy groups (fast/thermal).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from solver import Solver, Config, State
from materials import BETA

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)


def plot_timeseries(history: list, cfg: Config):
    """Plot scalar diagnostics vs time."""
    t        = np.array([h["t"]          for h in history])
    k_eff    = np.array([h["k_eff"]      for h in history])
    rho      = np.array([h["rho"]        for h in history])
    P        = np.array([h["P"]          for h in history])
    T1_max   = np.array([h["T1_max"]     for h in history])
    T2_max   = np.array([h["T2_max"]     for h in history])
    T1_base  = np.array([h["T1_base"]    for h in history])
    T2_base  = np.array([h["T2_base"]    for h in history])
    omega2   = np.array([h["omega2"]     for h in history])
    omega2mn = np.array([h["omega2_min"] for h in history])
    omega2mx = np.array([h["omega2_max"] for h in history])
    Lambda   = np.array([h["Lambda"]     for h in history])

    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(2, 3, figure=fig)
    fig.suptitle(
        f"1D Slab Stochastic Barrel Pyrolysis "
        f"(Two Group)\n"
        f"N={cfg.N}, mu={cfg.mu}, v1={cfg.v1}, "
        f"T_f={cfg.T_f}K",
        fontsize=12
    )

    # k_eff
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, k_eff, 'b')
    ax.axhline(1.0, color='r', ls='--', label='Critical')
    ax.set(xlabel='t (s)', ylabel=r'$k_\mathrm{eff}$',
            title='Multiplication Factor')
    ax.legend()
    ax.grid(True)

    # Reactivity
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, rho * 1e5, 'b', label=r'$\rho$')
    ax.axhline(BETA * 1e5, color='r', ls='--',
                label=r'$\beta$')
    ax.set(xlabel='t (s)',
            ylabel=r'Reactivity ($\times10^{-5}$)',
            title='Reactivity')
    ax.legend()
    ax.grid(True)

    # Amplitude
    ax = fig.add_subplot(gs[0, 2])
    ax.semilogy(t, P, 'g')
    ax.set(xlabel='t (s)', ylabel='P(t)',
            title='Neutron Population')
    ax.grid(True)

    # Temperatures
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, T1_max,  'r-',  label='PuO max')
    ax.plot(t, T2_max,  'b-',  label='Comb. max')
    ax.plot(t, T1_base, 'r--', label='PuO z=0')
    ax.plot(t, T2_base, 'b--', label='Comb. z=0')
    ax.axhline(cfg.T_f, color='k', ls=':',
                label=r'$T_f$')
    ax.set(xlabel='t (s)', ylabel='T (K)',
            title='Temperatures')
    ax.legend(fontsize=7)
    ax.grid(True)

    # Combustible content density with spatial spread
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, omega2, 'b', label=r'$\bar\omega_2$')
    ax.fill_between(t, omega2mn, omega2mx,
                     alpha=0.3, color='b',
                     label='min/max')
    ax.set(xlabel='t (s)', ylabel=r'$\omega_2$ (kg/m$^3$)',
            title='Combustible Content Density')
    ax.legend()
    ax.grid(True)

    # Prompt neutron lifetime
    ax = fig.add_subplot(gs[1, 2])
    ax.semilogy(t, Lambda, 'purple')
    ax.set(xlabel='t (s)', ylabel=r'$\Lambda$ (s)',
            title='Prompt Neutron Lifetime')
    ax.grid(True)

    plt.tight_layout()
    path = f"{FIGDIR}/timeseries_1g.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")


def plot_spatial(solver: Solver, state: State):
    """
    Plot spatial profiles at a given state.
    Four panels:
      1. Temperature both phases
      2. Flux shape both phases
      3. Fuel fraction
      4. Fission heat source
    """
    z      = solver.mesh.z
    phases = solver.phases
    cfg    = solver.cfg

    # Fission source: Ef * sum_g Sigma_f,g(T) * phi_g
    S_f = np.array([
        cfg.Ef * (
            phases[p].Sf_T(float(np.mean(state.T[p])))[0] * state.phi[p][0]
            + phases[p].Sf_T(float(np.mean(state.T[p])))[1] * state.phi[p][1]
        )
        for p in range(2)
    ])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Spatial Profiles at t = {state.t:.1f} s\n"
        f"k_eff = {state.k_eff:.5f},  "
        f"rho = {state.rho:.5f},  "
        f"P = {state.P:.3e}",
        fontsize=12
    )

    T_mix   = state.v[0] * state.T[0]   + state.v[1] * state.T[1]
    # psi is (2, 2, N) [phase, group, space]; mix across phases, keep groups separate
    psi_mix = state.v[0] * state.psi[0] + state.v[1] * state.psi[1]   # (2, N) [group, space]

    # --- Temperature ---
    ax = axes[0, 0]
    ax.plot(z, state.T[0], 'r', label='PuO')
    ax.plot(z, state.T[1], 'b', label='Combustible')
    ax.plot(z, T_mix, 'g--', label=r'$v_1T_1+v_2T_2$')
    ax.axvline(0, color='k', ls=':', lw=0.8,
                label='z=0 (heated)')
    ax.axhline(cfg.T_f, color='gray', ls='--',
                lw=0.8, label=r'$T_f$')
    ax.set(xlabel='z (cm)', ylabel='T (K)',
            title='Temperature')
    ax.legend(fontsize=8)
    ax.grid(True)

    # --- Flux shape (fast solid, thermal dashed) ---
    ax = axes[0, 1]
    ax.plot(z, state.psi[0, 0], 'r-',  label=r'$\psi_{1,\mathrm{fast}}$ PuO')
    ax.plot(z, state.psi[0, 1], 'r--', label=r'$\psi_{1,\mathrm{th}}$ PuO')
    ax.plot(z, state.psi[1, 0], 'b-',  label=r'$\psi_{2,\mathrm{fast}}$ Combustible')
    ax.plot(z, state.psi[1, 1], 'b--', label=r'$\psi_{2,\mathrm{th}}$ Combustible')
    ax.plot(z, psi_mix[0], 'g-',  label=r'$v_1\psi_{1,f}+v_2\psi_{2,f}$')
    ax.plot(z, psi_mix[1], 'g--', label=r'$v_1\psi_{1,th}+v_2\psi_{2,th}$')
    ax.set(xlabel='z (cm)', ylabel=r'$\psi(z)$',
            title='Flux Shape Function (fast solid / thermal dashed)')
    ax.legend(fontsize=6)
    ax.grid(True)

    # --- Combustible content density ---
    ax = axes[1, 0]
    ax.plot(z, state.omega[1], 'b',
             label=r'$\omega_2$ Combustible')
    ax.plot(z, state.omega[0], 'r',
             label=r'$\omega_1$ PuO')
    omega0_max = max(ph.omega0 for ph in phases)
    ax.set(xlabel='z (cm)', ylabel=r'$\omega(z)$ (kg/m$^3$)',
            title='Combustible Content Density',
            ylim=[0, 1.05 * omega0_max])
    ax.legend(fontsize=8)
    ax.grid(True)

    # --- Fission heat source ---
    ax = axes[1, 1]
    ax.plot(z, S_f[0], 'r', label=r'$S_{f,1}$ PuO')
    ax.plot(z, S_f[1], 'b',
             label=r'$S_{f,2}$ Combustible')
    ax.set(xlabel='z (cm)',
            ylabel=r'$S_\mathrm{fission}$ (W/m$^3$)',
            title='Fission Heat Source')
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    path = f"{FIGDIR}/spatial_1g_{state.t:.0f}s.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")


def print_summary(history: list, solver: Solver):
    """Print final state summary."""
    h   = history[-1]
    cfg = solver.cfg
    print("\n" + "="*50)
    print("1D SLAB TWO GROUP — SUMMARY")
    print("="*50)
    print(f"Final time   : {h['t']:.2f} s")
    print(f"Final k_eff  : {h['k_eff']:.6f}")
    print(f"Final rho    : {h['rho']:.6f}")
    print(f"Beta         : {BETA:.6f}")
    print(f"Final P(t)   : {h['P']:.4e}")
    print(f"Final Lambda : {h['Lambda']:.4e} s")
    print(f"Final T_max  : {h['T2_max']:.1f} K")
    print(f"Final T_base : {h['T2_base']:.1f} K")
    print(f"Final omega2 : {h['omega2']:.2f} kg/m^3")
    print(f"Timesteps    : {len(history)}")
    print(f"\nConfig:")
    print(f"  H={cfg.H}cm, N={cfg.N}")
    print(f"  mu={cfg.mu}, v1={cfg.v1}")
    print(f"  T_f={cfg.T_f}K, h={cfg.h_conv}W/m2K")
    print("="*50)


if __name__ == "__main__":

    cfg = Config(
        H           = 40.0,
        N           = 80,
        mu          = 0.0291785563199653,
        v1          = 0.7,
                              # Re-tuned for the two-group model: bisected
                              # mu (v1=0.7 fixed) for k_eff(0)=1.000000.
                              # v1=0.7 was chosen deliberately, not just
                              # for convenience -- the per-phase fixed-
                              # point solve's inner power iteration was
                              # found to become numerically unstable
                              # (k(v1) discontinuous, even briefly
                              # negative) for v1 roughly in [0.83, 0.98]
                              # at mu~0.1-0.2, most likely two of the 2N
                              # system's eigenvalues becoming closely
                              # spaced/near-degenerate there and breaking
                              # the plain power iteration's dominant-
                              # eigenvalue assumption (same failure mode
                              # family as the earlier "fuller form"
                              # spurious-eigenvalue issue from project
                              # history, now via a different mechanism --
                              # energy groups instead of drift-term
                              # discretisation). v1=0.7 sits well inside
                              # the confirmed-smooth, confirmed-stable
                              # region (checked v1 in [0.1,0.82] and
                              # +/-0.01 around this exact point). This
                              # solver robustness gap is a known
                              # limitation to revisit -- see project
                              # history -- not something papered over by
                              # this parameter choice.
        t_end       = 300.0,
        dt          = 0.05,
        T0          = 300.0,
        P0          = 1.0,
        T_f         = 1200.0,
        h_conv      = 50.0,
        emissivity  = 0.3,
        T_amb       = 300.0,  # z=H now loses heat convectively+radiatively
                               # to ambient (previously insulating) --
                               # reuses h_conv/emissivity by default.
        Ef          = 3.2e-11,
    )

    solver  = Solver(cfg)
    history = solver.run()

    print_summary(history, solver)

    plot_timeseries(history, cfg)
    plot_spatial(solver, solver.last_state)