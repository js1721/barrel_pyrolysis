"""
solver.py  (2D axisymmetric r-z)
================================
Coupled thermal / neutronic / kinetic transient for a cylindrical drum,
the 2D counterpart of barrel_pyrolysis1d/solver.py on the same basis:
relaxational stochastic closure, one eigenvalue for the coupled phases,
Marshak vacuum boundaries, neutron-temperature thermal group, absolute
flux amplitude and a source-driven initial state.

Per step (operator split, each piece explicit in the others):
  A. point kinetics with the intrinsic source   (kinetics.py)
  B. moderator percolation, if enabled          (percolation.py)
  D. pyrolysis, exact over the step, energy-consistent heat release
  C. two-phase conduction                       (thermal.py)
  E. new shape, k_eff and Lambda                (neutronics.py)

Geometry and boundaries: axis r = 0 symmetric; heated floor z = 0 at
T_f (convective + radiative); top z = H losing to T_amb; the drum wall
r = R held at T_amb (thermal) and a vacuum boundary (neutronic).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from mesh import CylindricalMesh2D
from materials import make_PuO, make_combustible
from thermal import solve_thermal_step, initialise_omega
from neutronics import (static_shape_solve, compute_Lambda,
                        source_amplitude_rate)
from kinetics import (advance_kinetics, initialise_precursors,
                      source_driven_steady_state)
from sources import source_density as pu_source_density
from percolation import (PercolationConfig, initialise_theta,
                         advance_percolation, adaptive_dt, thermal_sink,
                         mu_field)


@dataclass
class Config:
    # geometry (cm)
    R: float = 20.0
    H: float = 40.0
    Nr: int = 20
    Nz: int = 40
    # mixture: inverse correlation length (cm^-1; mu_r = mu_z = mu for
    # isotropic chunks) and fuel volume fraction
    mu: float = 0.5
    v1: float = 0.3
    percolation: PercolationConfig = field(default_factory=PercolationConfig)
    # time (s)
    t_end: float = 300.0
    dt: float = 0.5
    # initial / boundary
    T0: float = 300.0
    T_f: float = 1200.0
    h_conv: float = 50.0
    emissivity: float = 0.3
    T_amb: float = 300.0
    Ef: float = 3.2e-11
    radial_bc: str = "vacuum"
    # power scale: P0 = None -> source-driven steady state (needs k0 < 1);
    # phi = P psi in n/cm^2/s, psi's fuel phase unit volume integral
    P0: float = None
    isotopics: object = "reactor"
    source_density: float = None


@dataclass
class State:
    t: float
    T: np.ndarray        # (2, Nr, Nz)
    omega: np.ndarray    # (2, Nr, Nz)
    theta: np.ndarray    # (Nr, Nz)
    psi: np.ndarray      # (2 phases, 2 groups, Nr, Nz)
    P: float
    C: np.ndarray
    k_eff: float
    rho: float
    Lambda: float


class Solver:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mesh = CylindricalMesh2D(R=cfg.R, H=cfg.H, Nr=cfg.Nr, Nz=cfg.Nz)
        self.phases = [make_PuO(), make_combustible()]
        self.v = np.array([cfg.v1, 1.0 - cfg.v1])
        self.q_density = (float(cfg.source_density) if cfg.source_density is not None
                          else pu_source_density(cfg.isotopics, self.phases[0].rho * 1e-3))
        self.history: List[dict] = []

    def _mu(self, theta):
        if self.cfg.percolation.enabled:
            return mu_field(theta, self.cfg.percolation)
        return np.full((self.cfg.Nr, self.cfg.Nz), self.cfg.mu)

    def _shape(self, T, theta, psi_init=None):
        mu = self._mu(theta)
        k, psi = static_shape_solve(self.phases, self.mesh, T, self.v, mu, mu,
                                    psi_init=psi_init, radial_bc=self.cfg.radial_bc)
        return k, psi, compute_Lambda(self.phases, self.mesh, psi, T, self.v)

    def initialise(self) -> State:
        c, m = self.cfg, self.mesh
        T = np.full((2, m.Nr, m.Nz), c.T0)
        omega = initialise_omega(self.phases, m.Nr, m.Nz)
        theta = initialise_theta(c.percolation, m.Nr, m.Nz)
        k, psi, Lam = self._shape(T, theta)
        rho = (k - 1.0) / k
        q = source_amplitude_rate(self.phases, m, psi, self.v, self.q_density)
        P = source_driven_steady_state(q, rho, Lam) if c.P0 is None else c.P0
        st = State(0.0, T, omega, theta, psi, P, initialise_precursors(P, Lam),
                   k, rho, Lam)
        self.history.append(self._record(st))
        return st

    def step(self, st: State, dt: float) -> State:
        c, m, ph = self.cfg, self.mesh, self.phases
        # A. kinetics
        q = source_amplitude_rate(ph, m, st.psi, self.v, self.q_density)
        P, C = advance_kinetics(st.P, st.C, st.rho, st.Lambda, dt, q)
        # B. percolation -> latent-heat sink on the combustible phase
        S_extra = None
        theta = st.theta
        if c.percolation.enabled:
            theta, Gm, Gf = advance_percolation(st.theta, st.T[1], m.dz, dt, c.percolation)
            S_extra = np.zeros_like(st.T)
            S_extra[1] = -thermal_sink(Gm, Gf, c.percolation)
        # D (before C). Pyrolysis, integrated EXACTLY over the step at the
        # step's starting temperature, omega_new = omega exp(-kappa dt),
        # with its heat released as exactly the energy of the material
        # consumed, q (omega - omega_new) / dt. An explicit q*omega*rate
        # source over-releases energy whenever rate*dt > 1 -- at 1200 K
        # the rate is ~360 1/s, so a 1 s step released ~360x the energy
        # present and drove T far above the furnace temperature.
        omega = np.empty_like(st.omega)
        S_c = np.zeros_like(st.T)
        for p_, q_ph in enumerate(ph):
            kap = (q_ph.k_arr * np.exp(-q_ph.E_act / (8.314 * np.maximum(st.T[p_], 1.0)))
                   if q_ph.k_arr > 0 else np.zeros_like(st.T[p_]))
            omega[p_] = st.omega[p_] * np.exp(-kap * dt)
            S_c[p_] = q_ph.q * (st.omega[p_] - omega[p_]) / dt
        S_extra = S_c if S_extra is None else S_extra + S_c
        # C. conduction (fluxes at the step's new amplitude); combustion is
        # supplied through S_extra, so the thermal step's own explicit
        # combustion term is switched off by passing zero omega.
        mu = self._mu(st.theta)
        T = solve_thermal_step(ph, m, st.T, np.zeros_like(st.omega), P * st.psi, self.v,
                               mu, mu, dt, c.T_f, c.h_conv, c.emissivity, c.Ef,
                               T_amb=c.T_amb, S_extra=S_extra, closure="relaxational")
        # E. shape
        k, psi, Lam = self._shape(T, theta, psi_init=st.psi)
        return State(st.t + dt, T, omega, theta, psi, P, C, k, (k - 1.0) / k, Lam)

    def _record(self, st: State) -> dict:
        return dict(t=st.t, k_eff=st.k_eff, rho=st.rho, P=st.P, Lambda=st.Lambda,
                    T_max=float(st.T.max()), T_fuel_mean=float(st.T[0].mean()),
                    T_mod_mean=float(st.T[1].mean()), omega2=float(st.omega[1].mean()))

    def run(self, verbose: bool = True) -> list:
        c = self.cfg
        st = self.initialise()
        if verbose:
            print(f"2D drum R={c.R} cm H={c.H} cm ({c.Nr}x{c.Nz}), mu={c.mu}, v1={c.v1:.4f}")
            print(f"  k0={st.k_eff:.5f} rho0={st.rho:+.5f} Lambda={st.Lambda:.3e} s "
                  f"P0={st.P:.4e} (source {self.q_density:.3e} n/s/cm^3 fuel)")
            print(f"{'t':>7} {'k_eff':>8} {'rho':>9} {'P':>10} {'T_max':>7} "
                  f"{'T_fuel':>7} {'T_mod':>7}")
        n_print = max(1, int(round(10.0 / c.dt)))
        n = 0
        while st.t < c.t_end - 1e-9:
            dt = min(c.dt, c.t_end - st.t)
            if c.percolation.enabled:
                dt = min(dt, adaptive_dt(st.theta, st.T[1], self.mesh.dz, c.percolation))
            st = self.step(st, dt)
            self.history.append(self._record(st))
            n += 1
            if verbose and n % n_print == 0:
                h = self.history[-1]
                print(f"{h['t']:7.1f} {h['k_eff']:8.5f} {h['rho']:+9.5f} {h['P']:10.3e} "
                      f"{h['T_max']:7.1f} {h['T_fuel_mean']:7.1f} {h['T_mod_mean']:7.1f}")
        self.last_state = st
        return self.history
