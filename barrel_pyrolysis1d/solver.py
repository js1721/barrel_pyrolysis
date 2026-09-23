"""
solver.py
=========
Main coupled solver — 1D slab, two energy groups (fast/thermal).

Operator split per timestep (Section 5.1):
    A. Point kinetics  (Eqs. 93-94)
    B. Thermal solve   (Eq. 95)
    C. Pyrolysis       (Eq. 96)
    D. Static shape    (Eqs. 85-92)

    A fifth, optional stage -- moderator percolation (theta, the mobile
    liquid saturation) -- runs alongside B/C when Config.percolation is
    enabled: see percolation.py's module docstring for the physical
    picture. It is folded into the SAME operator split rather than
    given its own numbered stage because it plays the same role B/C
    already do -- an explicit update evaluated at the step's starting
    state, feeding back into the (previously constant) mu used by B and
    D -- not because it's a lesser concern.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List

from materials import (make_PuO, make_combustible,
                        BETA, I_GRP)
from mesh import Mesh1D
from thermal import (solve_thermal_step,
                      advance_pyrolysis,
                      pyrolysis_step,
                      initialise_omega,
                      arrhenius_rate)
from neutronics import (static_shape_solve,
                         compute_Lambda,
                         source_amplitude_rate)
from kinetics import source_driven_steady_state
from sources import source_density as _pu_source_density
from kinetics import (advance_kinetics,
                       initialise_precursors)
from percolation import (PercolationConfig,
                          initialise_theta,
                          advance_percolation,
                          mu_field,
                          thermal_sink,
                          adaptive_dt as percolation_adaptive_dt)

Rg = 8.314


@dataclass
class Config:
    # Geometry
    H:    float = 40.0
    N:    int   = 100

    # Stochastics
    mu:   float = 0.3
    v1:   float = 0.35

    # Stochastic closure, used by BOTH thermal and neutronics:
    #   "relaxational" (default) -- relaxational inter-phase exchange,
    #       no odd-order drift terms; checked against the realisation-
    #       averaged benchmark (benchmark_markov.py): correct sign and
    #       rate scaling and positive fluxes, but k ~6% low (non-
    #       conservative) at mu = 0.5-2 cm^-1.
    #   "document" -- the document's active slab equation (odd-order
    #       drift terms + anti-diffusive exchange). Kept for comparison;
    #       produces a negative combustible flux, diverging phase
    #       temperatures, and no fundamental mode at moderate mu.
    closure: str = "relaxational"

    # Moderator percolation: a dynamic, per-cell correlation length
    # driven by liquid-moderator melting/drainage/flash-off, replacing
    # the constant `mu` above wherever it's consumed (thermal.py,
    # neutronics.py) once enabled. PercolationConfig()'s own default
    # (enabled=False) makes this an EXACT no-op -- mu_field()/
    # thermal_sink()/advance_percolation() all degenerate to their
    # disabled branches, and Solver._mu_field() below bypasses
    # percolation entirely and returns the plain scalar `mu` above
    # unchanged -- so existing Config instances that never touch this
    # field reproduce prior behaviour bit-for-bit.
    percolation: PercolationConfig = field(default_factory=PercolationConfig)

    # Time
    t_end: float = 300.0
    dt:    float = 0.05

    # ICs
    T0:   float = 300.0   # Eq. (73)
    # ── Power scale ────────────────────────────────────────────────
    # The amplitude P is ABSOLUTE: phi = P * psi is the neutron flux in
    # n/cm^2/s (psi's fuel phase normalised to unit integral), and the
    # fission heating is converted to W/m^3 in thermal.py.
    #
    # P0 = None (default): the initial state is the SOURCE-DRIVEN steady
    #   state of a subcritical drum, P0 = -q Lambda / rho0, with q the
    #   Pu intrinsic neutron source. Requires k_eff(0) < 1.
    # P0 = <number>: explicit initial amplitude (n/cm/s), overriding the
    #   source-driven value (the source still acts during the transient).
    P0:   float = None
    # Intrinsic neutron source of the fuel phase: a preset ("reactor",
    # "weapons") or a dict of Pu (and Am-241) mass fractions, see
    # sources.py. source_density overrides it directly, in n/s/cm^3 of
    # fuel phase; 0.0 switches the source off.
    isotopics: object = "reactor"
    source_density: float = None

    # Heating BC at z=0 (Eq. 61)
    T_f:        float = 1200.0
    h_conv:     float = 50.0
    emissivity: float = 0.9

    # Heat-loss BC at z=H -- None keeps the previous insulating
    # boundary; set T_amb (e.g. 300.0) to allow convective+radiative
    # loss to an ambient exterior there. h_conv_amb/emissivity_amb
    # default to the z=0 values (h_conv/emissivity) when left None.
    T_amb:          float = None
    h_conv_amb:     float = None
    emissivity_amb: float = None

    # Physics
    Ef: float = 3.2e-11

    # Local hot-spot ignition ceiling. This model's combustible-phase
    # kinetics (Arrhenius, fast) genuinely outrun its own conduction
    # (diffusion timescale ~9s/cell vs sub-ms local burn-through once
    # ignited -- confirmed mesh-independent, i.e. a real finite-time
    # blow-up of the underlying reaction-diffusion system, not a
    # discretization artifact: refining N=80->320 made the blow-up
    # happen SOONER, not later/disappear). 3000K is well above real
    # hydrocarbon flame temperatures (~1500-2500K) -- past it the
    # local solid-conduction model has already broken down physically
    # (real material would have fully reacted/ablated), so this stops
    # the run at the point the result stops being physically
    # meaningful rather than let T diverge to numerically absurd
    # values chasing a singularity.
    T_ceiling: float = 3000.0

    # When False, the neutronics solve (static_shape_solve/
    # compute_Lambda) is always fed a temperature frozen at T0, instead
    # of the actual evolving T -- this removes ALL temperature
    # feedback on reactivity (both the Doppler term and the general
    # 1/v thermal-averaging in Sa_T/Sf_T/nuSf_T), not just Doppler
    # specifically, since there is no separate "Doppler-only" switch
    # in this single-group model. Thermal conduction and pyrolysis
    # still evolve normally (T, omega are unaffected) -- only what the
    # neutronics solve SEES is frozen. Since v does not evolve either,
    # k_eff becomes exactly constant for the whole run when this is
    # False, isolating point kinetics' pure exponential response from
    # any self-limiting feedback.
    T_feedback: bool = True


@dataclass
class State:
    t:      float
    T:      np.ndarray   # (2, N)
    omega:  np.ndarray   # (2, N)
    psi:    np.ndarray   # (2, 2, N) [phase, energy group (fast/thermal), space]
    phi:    np.ndarray   # (2, 2, N) [phase, energy group, space]
    P:      float
    C:      np.ndarray   # (I_GRP,)
    k_eff:  float
    rho:    float
    Lambda: float
    v:      np.ndarray   # (2,)
    theta:  np.ndarray   # (N,) mobile-moderator liquid saturation
                          # (percolation.py); stays at its initial
                          # theta_r floor for the whole run whenever
                          # Config.percolation.enabled is False.


class Solver:

    def __init__(self, config: Config):
        self.cfg    = config
        self.mesh   = Mesh1D(config.H, config.N)
        self.phases = [make_PuO(), make_combustible()]
        self.v      = np.array([config.v1,
                                 1.0 - config.v1])
        # intrinsic source, n/s/cm^3 of fuel phase
        if config.source_density is not None:
            self.q_density = float(config.source_density)
        else:
            rho_fuel_g_cc = self.phases[0].rho * 1.0e-3      # kg/m^3 -> g/cm^3
            self.q_density = _pu_source_density(config.isotopics, rho_fuel_g_cc)
        self.history: List[dict] = []

    def _mu_field(self, theta: np.ndarray):
        """
        Per-cell inverse correlation length fed to thermal.py's
        solve_thermal_step and neutronics.py's static_shape_solve.

        Percolation disabled (Config.percolation.enabled=False, the
        default) returns Config.mu completely unchanged -- a plain
        scalar, exactly as before this submodel existed, with theta
        not consulted at all. Enabling percolation replaces that
        constant with percolation.mu_field(theta, cfg.percolation), a
        per-cell (N,) array driven by the evolving liquid saturation.
        Both thermal.py and neutronics.py already accept either a
        scalar or an (N,) array transparently (see their own mu
        docstrings), so this is the only place the two paths need to
        be told apart.
        """
        if not self.cfg.percolation.enabled:
            return self.cfg.mu
        return mu_field(theta, self.cfg.percolation)

    def initialise(self) -> State:
        cfg    = self.cfg
        mesh   = self.mesh
        phases = self.phases
        v      = self.v

        T     = np.full((2, mesh.N), cfg.T0)
        omega = initialise_omega(phases, mesh)
        theta = initialise_theta(cfg.percolation, mesh.N)

        k_eff, psi = static_shape_solve(
            phases, mesh, T, v, self._mu_field(theta),
            closure=cfg.closure
        )
        Lambda = compute_Lambda(phases, mesh, psi, T, v)
        rho    = (k_eff - 1.0) / k_eff
        q0     = source_amplitude_rate(phases, mesh, psi, v, self.q_density)
        if cfg.P0 is None:
            if q0 <= 0.0:
                raise ValueError("P0=None needs an intrinsic source; set "
                                 "Config.P0 or a nonzero source.")
            P = source_driven_steady_state(q0, rho, Lambda)
        else:
            P = cfg.P0
        C      = initialise_precursors(P, Lambda)
        phi    = P * psi

        state = State(
            t=0.0, T=T, omega=omega,
            psi=psi, phi=phi,
            P=P, C=C,
            k_eff=k_eff, rho=rho,
            Lambda=Lambda, v=v,
            theta=theta,
        )
        self.history.append(self._record(state))
        return state

    def step(self, state: State, dt: float = None) -> State:
        cfg    = self.cfg
        dt     = cfg.dt if dt is None else dt
        phases = self.phases
        mesh   = self.mesh

        # A: Point kinetics, with the intrinsic source (constant over
        # the step, evaluated on the step's starting shape)
        q = source_amplitude_rate(phases, mesh, state.psi, state.v,
                                  self.q_density)
        P_new, C_new = advance_kinetics(
            state.P, state.C,
            state.rho, state.Lambda, dt, q
        )

        # Percolation (evaluated at the step's STARTING theta/T2, same
        # convention as thermal.py's own explicit source terms, e.g.
        # S_c -- see solve_thermal_step). The caller (this method) is
        # responsible for respecting percolation's own transport CFL,
        # same division of responsibility as every other explicit
        # piece here: _adaptive_dt folds percolation_adaptive_dt into
        # the single dt used for the whole step, so by the time dt
        # reaches this call it has already been bounded appropriately
        # (a no-op bound when percolation is disabled or K_sat<=0 --
        # see percolation.adaptive_dt's docstring).
        theta_new, Gamma_melt, Gamma_flash = advance_percolation(
            state.theta, state.T[1], mesh.dz, dt, cfg.percolation
        )
        S_extra = None
        if cfg.percolation.enabled:
            S_extra = np.zeros((2, mesh.N))
            # thermal_sink() returns the latent heat to SUBTRACT from
            # phase 2's balance (see its own docstring); solve_thermal_
            # step's S_extra is instead ADDED into S_total, so the sign
            # flips here.
            S_extra[1] = -thermal_sink(Gamma_melt, Gamma_flash, cfg.percolation)

        # B: Thermal solve. Uses mu computed from the OLD theta
        # (state.theta) -- consistent with T, omega, phi/psi all being
        # the step's starting values here too.
        # Pyrolysis BEFORE conduction, exact over the step at the step's
        # starting temperature, its heat released as exactly the energy
        # of the volatiles consumed (thermal.pyrolysis_step). The thermal
        # step's own explicit combustion term is switched off by passing
        # zero omega, so the heat enters once, through S_extra.
        omega_new, S_c = pyrolysis_step(phases, state.T, state.omega, dt)
        S_extra = S_c if S_extra is None else S_extra + S_c
        mu_thermal = self._mu_field(state.theta)
        phi_new = P_new * state.psi
        T_new   = solve_thermal_step(
            phases, mesh,
            state.T, np.zeros_like(state.omega),
            phi_new, state.v,
            mu_thermal, dt,
            cfg.T_f, cfg.h_conv, cfg.emissivity,
            cfg.Ef,
            T_amb=cfg.T_amb,
            h_conv_amb=cfg.h_conv_amb,
            emissivity_amb=cfg.emissivity_amb,
            closure=cfg.closure,
            S_extra=S_extra,
        )

        # C: Pyrolysis

        # D: Static shape solve (warm-started from the previous
        # timestep's converged shape/k_eff -- the fixed-point solve
        # converges in a handful of passes from a nearby solution,
        # vs. hundreds from a cold uniform guess)
        #
        # T_feedback=False freezes what the neutronics solve sees at
        # T0, regardless of the actual (still-evolving) T_new -- see
        # Config.T_feedback. mu here uses the freshly-advanced
        # theta_new -- the most current information available by this
        # point in the step, matching T_for_neutronics's own use of
        # T_new rather than state.T.
        T_for_neutronics = T_new if cfg.T_feedback else np.full_like(T_new, cfg.T0)
        mu_shape = self._mu_field(theta_new)
        k_new, psi_new = static_shape_solve(
            phases, mesh, T_for_neutronics, state.v, mu_shape,
            psi_init=state.psi, k_init=state.k_eff,
            closure=cfg.closure
        )
        Lambda_new = compute_Lambda(
            phases, mesh, psi_new, T_for_neutronics, state.v
        )
        rho_new = (k_new - 1.0) / k_new

        return State(
            t      = state.t + dt,
            T      = T_new,
            omega  = omega_new,
            psi    = psi_new,
            phi    = P_new * psi_new,
            P      = P_new,
            C      = C_new,
            k_eff  = k_new,
            rho    = rho_new,
            Lambda = Lambda_new,
            v      = state.v.copy(),
            theta  = theta_new,
        )

    def _adaptive_dt(self, state: State) -> float:
        dt     = self.cfg.dt
        phases = self.phases
        mesh   = self.mesh

        # Near (or past) prompt critical: refine dt whenever the local
        # prompt-kinetics growth rate is fast, whether approaching
        # prompt critical from below (margin > 0) or already past it
        # (margin <= 0, rho >= beta) -- the old "0 < margin" guard let
        # a supercritical excursion (margin < 0) skip this refinement
        # entirely and blow up within a single fixed-size step.
        margin = BETA - state.rho
        if margin < 0.1 * BETA:
            scale = max(abs(margin), 1e-8)
            dt    = min(dt, 0.01 * abs(state.Lambda) / scale)

        # Thermal CFL (K/rho/Cp are SI per-metre; mesh.dz is in cm)
        K_max  = max(ph.K for ph in phases)
        rC_min = min(ph.rho * ph.Cp for ph in phases)
        dz_m   = mesh.dz * 1.0e-2
        dt_cfl = 0.4 * rC_min * dz_m**2 / K_max
        dt     = min(dt, dt_cfl)

        # Reaction-rate CFL: advance_pyrolysis()'s own omega update is
        # unconditionally stable (implicit Euler) regardless of dt, but
        # solve_thermal_step's combustion source S_c = q*rate (rate is
        # a volumetric mass-consumption rate, kg/m^3/s, since omega is
        # a density -- see materials.py's omega0 comment) is evaluated
        # EXPLICITLY at the step's starting omega/T and held constant
        # over the whole dt -- if the local Arrhenius rate is fast, a
        # too-large dt overshoots the actual heat release before omega
        # has a chance to deplete, producing spurious temperature
        # swings.
        #
        # Bound dt by the resulting TEMPERATURE CHANGE per step, not by
        # dt*rate directly -- dt ~ 1/rate makes dt*rate a constant, so
        # the per-step dT = dt*S_c/rCp = dt*rate*(q/Cp) does NOT shrink
        # as rate grows (q/Cp is a fixed, large ratio here); bounding
        # dt ~ 1/rate alone still let T climb by a fixed, unbounded-in-
        # count sequence of large jumps every step (verified: T reached
        # 1e5+ K while dt collapsed to microseconds under that
        # formula). Capping the actual per-step dT directly is the
        # correct bound.
        dT_max = 5.0   # K per step, conservative given q/Cp's scale
        for p, ph in enumerate(phases):
            if ph.k_arr < 1e-30:
                continue
            rate = arrhenius_rate(ph, state.T[p], state.omega[p])
            S_c_max = float(np.max(ph.q * rate))
            if S_c_max > 1e-30:
                rCp = ph.rho * ph.Cp
                dt = min(dt, dT_max * rCp / S_c_max)

        # Percolation transport CFL (Godunov, sampled on both the
        # demand and supply flux slopes -- see percolation.adaptive_dt's
        # docstring). A true no-op (returns dt unchanged) when
        # percolation is disabled or K_sat<=0, so this cannot affect
        # any run that doesn't use the submodel.
        if self.cfg.percolation.enabled:
            dt = percolation_adaptive_dt(
                state.theta, state.T[1], mesh.dz,
                self.cfg.percolation, dt_cap=dt
            )

        # NOTE: no dt floor here (previously 1e-9). Near the local
        # thermal-explosion singularity the reaction CFL above can
        # legitimately demand dt << 1e-9s (rate approaches ~1e9/s and
        # beyond) -- a floor there was found to silently override the
        # dT_max bound by orders of magnitude right as T crossed
        # T_ceiling, producing a ~4800K single-step overshoot instead
        # of the intended ~5K. Since dT_max bounds temperature change
        # per step (not time per step), the number of steps needed to
        # cross any fixed T range stays finite even as dt shrinks
        # without bound approaching the singularity -- so removing the
        # floor does not risk an infinite loop, it just lets dt go as
        # small as the physics actually requires.
        return max(dt, 1e-300)

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
            "T1_base":    float(state.T[0, 0]),
            "T2_base":    float(state.T[1, 0]),
            "omega2":     float(state.omega[1].mean()),
            "omega2_min": float(state.omega[1].min()),
            "omega2_max": float(state.omega[1].max()),
            "phi1_fast_mean":    float(state.phi[0, 0].mean()),
            "phi1_thermal_mean": float(state.phi[0, 1].mean()),
            "phi2_fast_mean":    float(state.phi[1, 0].mean()),
            "phi2_thermal_mean": float(state.phi[1, 1].mean()),
            # Percolation diagnostics -- theta stays pinned at its
            # initial theta_r floor (and mu_min==mu_max==Config.mu)
            # for the whole run whenever percolation is disabled, so
            # these are harmless/uninformative rather than wrong in
            # that case, and always present so history's dict shape
            # doesn't depend on the enabled flag.
            "theta_max":  float(state.theta.max()),
            "theta_mean": float(state.theta.mean()),
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

    def _check_fuel_exhausted(self, state: State) -> bool:
        # Once the combustible phase's bulk fuel fraction is used up,
        # its combustion source (S_c, thermal.py) vanishes identically
        # (S_c is proportional to omega) -- there is nothing left to
        # pyrolyse, so the run is complete. Stopping here also avoids
        # continuing to evolve the stiff, already-fuel-starved system
        # after the physically interesting part of the transient is
        # over. Epsilon threshold since floating-point omega rarely
        # hits exactly 0.0.
        if state.omega[1].mean() < 1e-6:
            print(f"\n*** FUEL EXHAUSTED ***")
            print(f"    t       = {state.t:.4f} s")
            print(f"    omega2  = {state.omega[1].mean():.3e}")
            return True
        return False

    def _check_temperature_ceiling(self, state: State) -> bool:
        # See Config.T_ceiling docstring: this model's combustible-
        # phase kinetics genuinely outrun its own conduction, producing
        # a mesh-independent (i.e. real, not discretization-artifact)
        # finite-time local blow-up once ignition threshold is crossed.
        # Stop cleanly once the result is no longer physically
        # meaningful, rather than chase the singularity to numerically
        # absurd values.
        T_max = max(state.T[0].max(), state.T[1].max())
        if T_max >= self.cfg.T_ceiling:
            hot_phase = 0 if state.T[0].max() >= state.T[1].max() else 1
            hot_idx   = int(np.argmax(state.T[hot_phase]))
            print(f"\n*** LOCAL IGNITION CEILING REACHED ***")
            print(f"    t         = {state.t:.4f} s")
            print(f"    T_max     = {T_max:.1f} K "
                  f"(phase {hot_phase}, cell {hot_idx}, z={self.mesh.z[hot_idx]:.2f}cm)")
            print(f"    T_ceiling = {self.cfg.T_ceiling:.1f} K")
            return True
        return False

    def run(self) -> List[dict]:
        state = self.initialise()
        cfg   = self.cfg
        self.last_state = state

        print(f"1D Slab — Two Group (fast/thermal)")
        print(f"  N={cfg.N}, H={cfg.H}cm")
        print(f"  mu={cfg.mu} cm^-1, v1={cfg.v1}")
        print(f"  percolation={'ON' if cfg.percolation.enabled else 'off'}")
        print(f"  T_f={cfg.T_f}K, h={cfg.h_conv}W/m2K")
        print(f"  k_eff(0) = {state.k_eff:.5f}")
        print(f"  rho(0)   = {state.rho:.6f}")
        print(f"  beta     = {BETA:.6f}")
        print()
        print(f"{'t':>8} {'k_eff':>9} {'rho':>10} "
              f"{'P':>10} {'T_max':>8} {'w2':>8}")
        print("-" * 58)

        while state.t < cfg.t_end:
            dt    = self._adaptive_dt(state)
            state = self.step(state, dt)
            self.last_state = state
            self.history.append(self._record(state))

            if self._check_safety(state):
                break

            if self._check_fuel_exhausted(state):
                break

            if self._check_temperature_ceiling(state):
                break

            if len(self.history) % 20 == 0:
                T_max = max(state.T[0].max(),
                            state.T[1].max())
                print(
                    f"{state.t:8.2f} "
                    f"{state.k_eff:9.5f} "
                    f"{state.rho:10.6f} "
                    f"{state.P:10.3e} "
                    f"{T_max:8.1f} "
                    f"{state.omega[1].mean():8.2f}"
                )

        return self.history