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
from percolation import PercolationConfig
from benchmark_markov import safety_report, print_safety_report

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

    # TODO: PLACEHOLDER -- copied verbatim from percolation.py's own
    # self-test (see its __main__ block). These are illustrative values
    # chosen to exercise melting/drainage/flashing in a synthetic test,
    # NOT fitted to the real PVC-dominated waste case. Do not treat any
    # number below as a physical result until it's been replaced with a
    # fitted/sourced value. See HANDOFF.md, "Outstanding" item 2.
    # DISABLED: with ell0 = 1/0.3 cm the dry correlation length gives
    # mu0 = 0.3 cm^-1 at t=0, above the stochastic closure's breakdown
    # (the algebraic coupling exceeds the combustible thermal-group
    # removal near mu ~ 0.2 cm^-1 at this composition), so
    # static_shape_solve now raises ClosureBreakdownError immediately.
    # Re-enable only with an ell0 (and ell_max) that keep
    # mu = 1/ell below that threshold -- see neutronics.closure_margin.
    percolation_cfg = PercolationConfig(
        enabled=False,
        theta_r=0.02, theta_s=0.30,
        T_melt=400.0, T_boil=900.0,
        k_melt=0.05, k_flash=0.02,
        L_fus=2.0e5, L_vap=8.0e5, rho_liquid=900.0,
        K_sat=0.05, n_perc=3.0,
        ell0=1.0/0.3, ell_max=15.0, theta_c=0.25, s0_perc=1.0, p_perc=4.0,
    )

    cfg = Config(
        percolation = percolation_cfg,
        H           = 40.0,
        N           = 80,
        # mu and v1 (relaxational closure, joint eigenvalue, Marshak BC):
        # criticality by composition exists only for mu > ~0.071 cm^-1.
        # Below that the fuel chords exceed ~14 cm, a single fuel chunk is
        # supercritical on its own, and k > 1 for every composition. v1 is
        # bisected for k_eff(0) = 0.98 at the chosen mu (source-driven start).
        mu          = 0.5,        # chords ~3-6 cm; model k ~6% below benchmark here
        v1          = 0.17229402357524248,    # subcritical storage state, k_eff(0) = 0.98 (a choice -- change v1 to change it); the initial power follows from the reactor-grade Pu intrinsic source, see Config.P0 / sources.py
        t_end       = 300.0,
        dt          = 0.05,
        T0          = 300.0,
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

    # The closed-form (relaxational) closure above is a biased point
    # estimate of the true composition's criticality, not a substitute for
    # it -- see HANDOFF.md "Open" items 1-2 and benchmark_markov.py. v1 is
    # a fixed composition parameter (not a solver state variable), so the
    # initial k_eff is representative of this check for the whole run.
    rep = safety_report(cfg.mu, cfg.v1, H=cfg.H, model_k=history[0]["k_eff"])
    print_safety_report(rep)

    plot_timeseries(history, cfg)
    plot_spatial(solver, solver.last_state)