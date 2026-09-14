import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from solver import Solver, Config
from materials import BETA


def plot_timeseries(history: list, cfg: Config):
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

    fig = plt.figure(figsize=(15, 10))
    gs  = GridSpec(3, 3, figure=fig)
    fig.suptitle(
        "Stochastic Barrel Pyrolysis — 2D Cylindrical\n"
        f"T_f={cfg.T_f}K, h={cfg.h_conv}W/m²K, "
        f"ε={cfg.emissivity}",
        fontsize=12
    )

    # k_eff
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, k_eff, 'b')
    ax.axhline(1.0, color='r', ls='--', label='Critical')
    ax.set(xlabel='t (s)', ylabel=r'$k_\mathrm{eff}$',
            title='Multiplication Factor')
    ax.legend(); ax.grid(True)

    # Reactivity
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, rho*1e5, 'b', label=r'$\rho$')
    ax.axhline(BETA*1e5, color='r', ls='--',
                label=r'$\beta$')
    ax.set(xlabel='t (s)',
            ylabel=r'Reactivity ($\times10^{-5}$)',
            title='Reactivity')
    ax.legend(); ax.grid(True)

    # Amplitude
    ax = fig.add_subplot(gs[0, 2])
    ax.semilogy(t, P, 'g')
    ax.set(xlabel='t (s)', ylabel='P(t)',
            title='Neutron Population')
    ax.grid(True)

    # Temperatures — max
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, T1_max, 'r-',  label='PuO max')
    ax.plot(t, T2_max, 'b-',  label='Comb. max')
    ax.plot(t, T1_base,'r--', label='PuO base (z=0)')
    ax.plot(t, T2_base,'b--', label='Comb. base (z=0)')
    ax.axhline(cfg.T_f, color='k', ls=':', label='T_f')
    ax.set(xlabel='t (s)', ylabel='T (K)',
            title='Temperatures')
    ax.legend(fontsize=7); ax.grid(True)

    # Fuel fraction
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, omega2, 'b', label=r'$\bar\omega_2$')
    ax.fill_between(t, omega2mn, omega2mx,
                     alpha=0.3, color='b',
                     label='min/max')
    ax.set(xlabel='t (s)', ylabel=r'$\omega_2$',
            title='Combustible Fuel Fraction')
    ax.legend(); ax.grid(True)

    # Lambda
    ax = fig.add_subplot(gs[1, 2])
    ax.semilogy(t, Lambda, 'purple')
    ax.set(xlabel='t (s)', ylabel=r'$\Lambda$ (s)',
            title='Prompt Neutron Lifetime')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("timeseries.png", dpi=150,
                bbox_inches='tight')
    plt.show()


def plot_spatial_fields(solver: Solver, state):
    mesh = solver.mesh

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Spatial Fields at t = {state.t:.1f} s",
        fontsize=13
    )

    fields = [
        (state.T[0],
         "PuO Temperature (K)", 'hot'),
        (state.T[1],
         "Combustible Temp (K)", 'hot'),
        (state.psi[0],
         r"PuO Shape $\psi_1$", 'viridis'),
        (state.psi[1],
         r"Comb. Shape $\psi_2$", 'viridis'),
        (state.omega[1],
         r"Fuel Fraction $\omega_2$", 'Blues_r'),
        (state.phi[0],
         r"PuO Flux $\phi_1$", 'plasma'),
    ]

    for ax, (field, title, cmap) in zip(
            axes.ravel(), fields):
        im = ax.pcolormesh(
            mesh.Z2D, mesh.R2D, field,
            cmap=cmap, shading='auto'
        )
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set(xlabel='z (cm)', ylabel='r (cm)',
               title=title)

    plt.tight_layout()
    plt.savefig(f"spatial_{state.t:.0f}s.png",
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_axial_profiles(solver: Solver, state):
    mesh = solver.mesh
    z    = mesh.z

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"Axial Profiles (centreline) at t={state.t:.1f} s",
        fontsize=12
    )

    axes[0].plot(z, state.T[0][0,:], 'r-', label='PuO')
    axes[0].plot(z, state.T[1][0,:], 'b-', label='Comb.')
    axes[0].axvline(0, color='k', ls=':', label='z=0 (heated)')
    axes[0].set(xlabel='z (cm)', ylabel='T (K)',
                 title='Temperature')
    axes[0].legend(fontsize=8); axes[0].grid(True)

    axes[1].plot(z, state.psi[0][0,:], 'r-',
                  label=r'$\psi_1$')
    axes[1].plot(z, state.psi[1][0,:], 'b-',
                  label=r'$\psi_2$')
    axes[1].set(xlabel='z (cm)', ylabel=r'$\psi$',
                 title='Flux Shape')
    axes[1].legend(fontsize=8); axes[1].grid(True)

    axes[2].plot(z, state.omega[1][0,:], 'b-',
                  label='centre')
    axes[2].plot(z, state.omega[1][-1,:], 'b--',
                  label='edge')
    axes[2].set(xlabel='z (cm)',
                 ylabel=r'$\omega_2$',
                 title='Fuel Fraction')
    axes[2].legend(fontsize=8); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f"axial_{state.t:.0f}s.png",
                dpi=150, bbox_inches='tight')
    plt.show()


def print_summary(history: list, solver: Solver):
    h   = history[-1]
    cfg = solver.cfg
    print("\n" + "="*55)
    print("SIMULATION SUMMARY")
    print("="*55)
    print(f"Final time        : {h['t']:.2f} s")
    print(f"Final k_eff       : {h['k_eff']:.6f}")
    print(f"Final rho         : {h['rho']:.6f}")
    print(f"Beta (Pu-239)     : {BETA:.6f}")
    print(f"Final P(t)        : {h['P']:.4e}")
    print(f"Final Lambda      : {h['Lambda']:.4e} s")
    print(f"Final T_base mean : {h['T2_base']:.1f} K")
    print(f"Final T_max       : {h['T2_max']:.1f} K")
    print(f"Final omega2 mean : {h['omega2']:.5f}")
    print(f"Final omega2 min  : {h['omega2_min']:.5f}")
    print(f"Timesteps         : {len(history)}")
    print(f"\nGeometry  : R={cfg.R}cm, H={cfg.H}cm")
    print(f"Mesh      : Nr={cfg.Nr}, Nz={cfg.Nz}")
    print(f"mu        : {cfg.mu} cm^-1")
    print(f"v1        : {cfg.v1}")
    print(f"T_f       : {cfg.T_f} K")
    print(f"h_conv    : {cfg.h_conv} W/m^2/K")
    print(f"emissivity: {cfg.emissivity}")
    print("="*55)


if __name__ == "__main__":

    cfg = Config(
        # Geometry
        R           = 20.0,
        H           = 40.0,
        Nr          = 25,
        Nz          = 50,

        # Stochastics (mu_r = mu_z = mu)
        mu          = 0.3,
        v1          = 0.35,

        # Time
        t_end       = 200.0,
        dt          = 0.1,

        # ICs (Eqs. 78-80)
        T0          = 300.0,
        P0          = 1.0,

        # Heating BC at z=0 (Eq. 66)
        T_f         = 1200.0,
        h_conv      = 50.0,
        emissivity  = 0.9,

        # Physics
        neutron_speed = 2.2e5,
        Ef            = 3.2e-11,
    )

    solver  = Solver(cfg)
    history = solver.run()

    print_summary(history, solver)

    # Spatial plots at final state
    state = solver.initialise()
    for _ in range(min(len(history)-1, 100)):
        state = solver.step(state)

    plot_timeseries(history, cfg)
    plot_spatial_fields(solver, state)
    plot_axial_profiles(solver, state)