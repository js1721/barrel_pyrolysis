import numpy as np
from dataclasses import dataclass, field
from typing import List

from materials import (make_PuO, make_combustible,
                        BETA, DELAYED_GROUPS)
from mesh import CylindricalMesh2D
from thermal import (solve_thermal_step,
                      advance_pyrolysis,
                      initialise_omega)
from neutronics import static_shape_solve, compute_Lambda
from kinetics import advance_kinetics, initialise_precursors

Rg = 8.314


@dataclass
class Config:
    """
    Solver configuration.
    All boundary conditions from Section 4.1 of document.
    """
    # --- Geometry ---
    R:    float = 20.0
    H:    float = 40.0
    Nr:   int   = 30
    Nz:   int   = 60

    # --- Stochastics (mu_r = mu_z = mu, document note) ---
    mu:   float = 0.5
    v1:   float = 0.4

    # --- Time ---
    t_end: float = 200.0
    dt:    float = 0.1

    # --- Initial conditions (Eqs. 78-80) ---
    T0:   float = 300.0   # Eq. (78)
    P0:   float = 1.0     # Eq. (79)

    # --- Heating BC at z=0 (Eq. 66) ---
    T_f:        float = 1200.0   # furnace temperature K
    h_conv:     float = 50.0     # convective HTC W/m^2/K
    emissivity: float = 0.9      # surface emissivity

    # --- Physics ---
    neutron_speed: float = 2.2e5
    Ef:            float = 3.2e-11


@dataclass
class State:
    """Complete system state."""
    t:      float
    T:      np.ndarray    # (2, Nr, Nz)
    omega:  np.ndarray    # (2, Nr, Nz)
    phi:    np.ndarray    # (2, Nr, Nz)
    psi:    np.ndarray    # (2, Nr, Nz)
    P:      float
    C:      np.ndarray    # (I_GRP,)
    k_eff:  float
    rho:    float
    Lambda: float
    v:      np.ndarray    # (2,)


class Solver:
    """
    2D cylindrical two-phase stochastic pyrolysis
    and neutronics solver.

    Boundary conditions from Section 4.1 of document:
      Thermal z=0  : convective + radiative (Eqs. 76-77)
      Thermal other: stochastic Neumann (Eqs. 72-73)
      Neutronics   : stochastic vacuum (Eqs. 83-84)
      Neutronics r=0: symmetry

    Operator split:
      A. Point kinetics
      B. Thermal solve  (with BCs 76-77 at z=0)
      C. Pyrolysis
      D. Static shape   (with BCs 83-84)
    """

    def __init__(self, config: Config):
        self.cfg  = config
        self.mesh = CylindricalMesh2D(
            config.R, config.H, config.Nr, config.Nz
        )
        self.mats    = [make_PuO(), make_combustible()]
        self.history: List[dict] = []

    def initialise(self) -> State:
        cfg    = self.cfg
        mesh   = self.mesh
        mats   = self.mats
        Nr, Nz = mesh.Nr, mesh.Nz

        v = np.array([cfg.v1, 1.0 - cfg.v1])

        # Eq. (78): T(r,z,0) = 300 K
        T = np.full((2, Nr, Nz), cfg.T0)

        # omega(r,z,0) = omega0 (uniform)
        omega = initialise_omega(mats, mesh)

        # Initial shape solve with BCs (83)-(84)
        k_eff, psi = static_shape_solve(
            mats, mesh, T, v, cfg.mu
        )
        Lambda = compute_Lambda(
            mats, mesh, psi, T, cfg.neutron_speed
        )

        # Eqs. (79)-(80)
        P = cfg.P0
        C = initialise_precursors(P, Lambda)

        phi = P * psi
        rho = (k_eff - 1.0) / k_eff

        state = State(
            t=0.0, T=T, omega=omega, phi=phi,
            psi=psi, P=P, C=C,
            k_eff=k_eff, rho=rho, Lambda=Lambda,
            v=v
        )
        self.history.append(self._record(state))
        return state

    def step(self, state: State) -> State:
        cfg  = self.cfg
        dt   = cfg.dt
        mats = self.mats
        mesh = self.mesh

        # ── A: Point kinetics ──────────────────────────────────
        P_new, C_new = advance_kinetics(
            state.P, state.C,
            state.rho, state.Lambda, dt
        )

        # ── B: Thermal solve ───────────────────────────────────
        # BCs: Eqs. (76)-(77) at z=0
        #      Eqs. (72)-(73) at all other boundaries
        phi_new = P_new * state.psi
        T_new   = solve_thermal_step(
            mats, mesh,
            state.T, state.omega,
            phi_new, state.v,
            cfg.mu, dt,
            cfg.T_f, cfg.h_conv, cfg.emissivity,
            cfg.Ef
        )

        # ── C: Pyrolysis ───────────────────────────────────────
        omega_new = advance_pyrolysis(
            mats, T_new, state.omega, dt
        )

        # ── D: Static shape solve ──────────────────────────────
        # BCs: Eqs. (83)-(84) stochastic vacuum
        k_new, psi_new = static_shape_solve(
            mats, mesh, T_new, state.v, cfg.mu
        )
        Lambda_new = compute_Lambda(
            mats, mesh, psi_new, T_new, cfg.neutron_speed
        )
        rho_new = (k_new - 1.0) / k_new

        return State(
            t      = state.t + dt,
            T      = T_new,
            omega  = omega_new,
            phi    = P_new * psi_new,
            psi    = psi_new,
            P      = P_new,
            C      = C_new,
            k_eff  = k_new,
            rho    = rho_new,
            Lambda = Lambda_new,
            v      = state.v.copy(),
        )

    def _adaptive_dt(self, state: State) -> float:
        """Adaptive timestep."""
        dt = self.cfg.dt

        # Near prompt critical
        margin = BETA - state.rho
        if 0.0 < margin < 0.1 * BETA:
            dt = min(dt,
                     0.01 * abs(state.Lambda / margin))

        # Thermal CFL
        K_max  = max(m.K for m in self.mats)
        rC_min = min(m.rho * m.Cp for m in self.mats)
        dt_r   = 0.4 * rC_min * self.mesh.dr**2 / K_max
        dt_z   = 0.4 * rC_min * self.mesh.dz**2 / K_max
        dt     = min(dt, dt_r, dt_z)

        # Pyrolysis stability
        mat2     = self.mats[1]
        T2_max   = float(state.T[1].max())
        rate_max = mat2.k_arr * np.exp(
            -mat2.E_act / (Rg * max(T2_max, 1.0))
        )
        if rate_max > 1e-30:
            dt = min(dt, 0.1 / rate_max)

        return max(dt, 1e-6)

    def _record(self, state: State) -> dict:
        return {
            "t":          state.t,
            "k_eff":      state.k_eff,
            "rho":        state.rho,
            "P":          state.P,
            "Lambda":     state.Lambda,
            "T1_max":     float(state.T[0].max()),
            "T2_max":     float(state.T[1].max()),
            "T1_mean":    float(state.T[0].mean()),
            "T2_mean":    float(state.T[1].mean()),
            # Track temperature at heated base z=0
            "T1_base":    float(state.T[0][:, 0].mean()),
            "T2_base":    float(state.T[1][:, 0].mean()),
            "omega1":     float(state.omega[0].mean()),
            "omega2":     float(state.omega[1].mean()),
            "omega2_min": float(state.omega[1].min()),
            "omega2_max": float(state.omega[1].max()),
            "v1":         float(state.v[0]),
            "v2":         float(state.v[1]),
        }

    def _check_safety(self, state: State) -> bool:
        if state.rho >= BETA:
            print(f"\n*** PROMPT CRITICAL ***")
            print(f"    t     = {state.t:.4f} s")
            print(f"    rho   = {state.rho:.6f}")
            print(f"    beta  = {BETA:.6f}")
            print(f"    k_eff = {state.k_eff:.6f}")
            return True
        return False

    def run(self) -> List[dict]:
        state = self.initialise()

        print(f"Boundary conditions:")
        print(f"  z=0  : h={self.cfg.h_conv} W/m2/K, "
              f"T_f={self.cfg.T_f} K, "
              f"eps={self.cfg.emissivity}")
        print(f"  other: stochastic Neumann (Eqs.72-73)")
        print(f"  neutronics: stochastic vacuum (Eqs.83-84)")
        print()
        print(f"{'t':>8} {'k_eff':>9} {'rho':>10} "
              f"{'P':>10} {'T_base':>8} {'T_max':>8}")
        print("-" * 60)

        while state.t < self.cfg.t_end:
            dt    = self._adaptive_dt(state)
            state = self.step(state)
            self.history.append(self._record(state))

            if self._check_safety(state):
                break

            if len(self.history) % 10 == 0:
                T_base = max(state.T[0][:,0].mean(),
                             state.T[1][:,0].mean())
                T_max  = max(state.T[0].max(),
                             state.T[1].max())
                print(
                    f"{state.t:8.2f} "
                    f"{state.k_eff:9.5f} "
                    f"{state.rho:10.6f} "
                    f"{state.P:10.3e} "
                    f"{T_base:8.1f} "
                    f"{T_max:8.1f}"
                )

        return self.history