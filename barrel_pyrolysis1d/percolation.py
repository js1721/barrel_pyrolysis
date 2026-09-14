"""
percolation.py
===============
Moderator percolation: melting, gravity drainage, and flash-off of a
mobile liquid component within the combustible phase, and the resulting
dynamic correlation length mu(z,t) fed back into thermal.py and
neutronics.py in place of the constant Config.mu.

Physical picture (barrel document's Section "Moderator Percolation and
a Dynamic Correlation Length"): as the combustible phase heats past its
melting point, a mobile liquid component separates from the solid
matrix and drains under gravity through the packed bed, concentrating
locally before flashing off once the local temperature exceeds its
boiling point. Concentrating liquid increases the effective clustering
scale of the two-phase mixture, so the correlation length ell = 1/mu is
made a function of the local liquid saturation theta(z,t).

Geometry / sign convention
---------------------------
z=0 is the barrel's HEATED bottom (the furnace boundary, Eq. 61 of the
document); z=H is the top. Gravity therefore drains liquid toward
DECREASING z (toward the heated floor), not increasing z. This is a
genuine physical choice this module makes explicit (the document's
draft text says only "drains under gravity" without pinning a
direction) -- it means melted moderator pools preferentially at the
bottom regardless of where in z it melted, which is the physically
interesting "concentration" effect, and it reinforces rather than
competes with the fact that z=0 is already the hottest, fastest-
melting region.

Transport equation
-------------------
    d(theta)/dt = d/dz[K(theta)] + Gamma_melt - Gamma_flash

with the flux K(theta) >= 0 representing transport in the -z direction
(i.e. the physical flux vector is -K(theta) * z_hat), Brooks-Corey
relative conductivity

    K(theta) = K_sat * ((theta - theta_r)/(theta_s - theta_r))^n_perc,
               theta > theta_r,  else 0

Both boundaries (z=0 and z=H) are CLOSED to liquid transport: no liquid
enters from outside the top, and none exits through the solid barrel
floor at the bottom -- it pools in the bottom cell instead. This is
solved explicitly with a first-order upwind (Godunov) finite-volume
scheme; since dK/dtheta >= 0 the characteristic speed d(-K)/dtheta <= 0
everywhere, so the correct upwind face value is always taken from the
cell on the +z side of the face.

Sources
-------
Melting (relaxation to theta_s once T_2 > T_melt) and flash-off
(depletion once T_2 > T_boil) are both simple Heaviside-triggered
relaxations, matching the Arrhenius-kinetics level of rigour already
used for omega in thermal.py -- not a full enthalpy/Stefan treatment.
Both carry latent heat that must be returned to phase 2's energy
balance as an explicit sink (see thermal_sink()); a caller that ignores
this return value would silently violate energy conservation by
manufacturing melt/flash heat capacity for free.

Correlation length coupling
-----------------------------
    s(z,t)    = theta(z,t) / theta_c
    ell(z,t)  = ell0 + (ell_max - ell0) * s^p / (s0^p + s^p)
    mu(z,t)   = 1 / ell(z,t)

theta_c is the percolation threshold saturation (where the liquid is
taken to just form a connected pathway through the bed); ell0 is the
dry/baseline correlation length (Config.mu's old constant value,
inverted); ell_max is a physically capped maximum (pool depth or
barrel radius); s0 ~ 1 locates the transition near threshold, p
controls its sharpness. This is a saturating sigmoid in s, not the
literal critical-exponent divergence of percolation theory -- see the
document section for why that distinction matters.

Units: z in cm (matching mesh.py/thermal.py's convention), mu in
cm^-1, K_sat in cm/s, T in K, latent heats in J/kg via rho_liquid
(kg/m^3) converted internally to a per-cm^3 quantity where needed by
thermal.py's caller (thermal_sink() returns W/m^3, matching thermal.py's
existing source terms -- see its own CM_TO_M handling).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PercolationConfig:
    """
    All parameters governing the mobile-moderator submodel. Disabled by
    default (theta_s = 0.0), which makes melt_source, flash_sink, and
    the transport step all identically zero and mu_field() return a
    uniform mu0 everywhere -- an exact no-op reproducing the previous
    constant-mu behaviour when percolation isn't wanted.
    """
    enabled: bool = False

    # Saturation bounds (dimensionless, fraction of the combustible
    # phase's local pore/bulk volume occupied by mobile liquid)
    theta_r: float = 0.02     # irreducible (immobile) saturation
    theta_s: float = 0.0      # saturation when fully melted; 0 = disabled

    # Melting / flashing
    T_melt:   float = np.inf  # K; np.inf => melting never triggers
    T_boil:   float = np.inf  # K; np.inf => flashing never triggers
    k_melt:   float = 0.0     # melting relaxation rate, s^-1
    k_flash:  float = 0.0     # flash-off relaxation rate, s^-1
    L_fus:    float = 0.0     # latent heat of fusion, J/kg
    L_vap:    float = 0.0     # latent heat of vaporisation, J/kg
    rho_liquid: float = 900.0 # liquid moderator density, kg/m^3

    # Gravity drainage (Brooks-Corey)
    K_sat:  float = 0.0       # saturated hydraulic conductivity, cm/s
    n_perc: float = 3.0       # pore-structure exponent

    # Correlation-length constitutive law
    ell0:    float = 1.0 / 0.3   # dry correlation length, cm (matches
                                   # the old default mu0=0.3 cm^-1)
    ell_max: float = 20.0     # capped max correlation length, cm
    theta_c: float = 0.3      # percolation threshold saturation
    s0_perc: float = 1.0      # sigmoid centre (in units of theta/theta_c)
    p_perc:  float = 4.0      # sigmoid sharpness


def initialise_theta(cfg: PercolationConfig, N: int) -> np.ndarray:
    """theta(z, 0) = theta_r uniformly (bone dry above the irreducible
    floor)."""
    return np.full(N, cfg.theta_r)


def _K_brooks_corey(theta: np.ndarray, cfg: PercolationConfig) -> np.ndarray:
    """Relative hydraulic conductivity, cm/s. Zero at/below theta_r."""
    frac = np.clip((theta - cfg.theta_r) / max(cfg.theta_s - cfg.theta_r, 1e-30),
                    0.0, 1.0)
    return cfg.K_sat * frac ** cfg.n_perc


def melt_source(theta: np.ndarray, T2: np.ndarray,
                 cfg: PercolationConfig) -> np.ndarray:
    """Gamma_melt(z,t), s^-1 saturation units. Zero wherever T2 <= T_melt."""
    if not cfg.enabled:
        return np.zeros_like(theta)
    active = T2 > cfg.T_melt
    return np.where(active, cfg.k_melt * (cfg.theta_s - theta), 0.0)


def flash_sink(theta: np.ndarray, T2: np.ndarray,
               cfg: PercolationConfig) -> np.ndarray:
    """Gamma_flash(z,t), s^-1 saturation units. Zero wherever T2 <= T_boil."""
    if not cfg.enabled:
        return np.zeros_like(theta)
    active = T2 > cfg.T_boil
    return np.where(active, cfg.k_flash * theta, 0.0)


def thermal_sink(Gamma_melt: np.ndarray, Gamma_flash: np.ndarray,
                  cfg: PercolationConfig) -> np.ndarray:
    """
    Latent-heat sink to subtract from phase 2's energy balance, W/m^3.

    Gamma_melt/Gamma_flash are saturation rates (s^-1); multiplying by
    rho_liquid (kg/m^3) converts to a volumetric mass rate (kg/m^3/s),
    then by L_fus/L_vap (J/kg) gives W/m^3 -- matching thermal.py's
    S_f/S_c convention exactly (no extra phase-density factor, for the
    same reason omega's q*rate needs none there: this is already a
    per-total-volume quantity).
    """
    if not cfg.enabled:
        return np.zeros_like(Gamma_melt)
    return cfg.rho_liquid * (cfg.L_fus * Gamma_melt + cfg.L_vap * Gamma_flash)


def _limited_face_fluxes(theta: np.ndarray, cfg: PercolationConfig) -> np.ndarray:
    """
    Supply/demand-limited face fluxes (Cell Transmission Model style --
    Daganzo 1994 -- standard fix for exactly this problem in kinematic-
    wave/traffic-flow numerics), cm/s, returned as F_face shape (N+1,)
    with F_face[n] the flux at the -z face of cell n (F_face[0] and
    F_face[N] are the closed outer boundaries, always 0).

    A plain upwind scheme (demand-only: flux = K(theta[donor])) lets a
    cell keep receiving inflow forever even after it reaches theta_s,
    since K(theta) depends only on the SENDING cell -- verified in this
    module's self-test to either overshoot theta_s (if clipped
    post-hoc, silently destroying mass) or force the adaptive timestep
    to collapse toward zero forever once any cell saturates under
    continued upstream inflow (a PERSISTENT condition, unlike the
    transient reaction singularities solver.py's dt_max handles, so it
    never recovers). The fix is to also cap flux by the RECEIVING
    cell's remaining capacity:

        demand[n]   = K(theta[n])                     -- donor's supply
        supply[n-1] = K_sat*(1 - relsat[n-1]^n_perc)   -- receiver's
                                                           remaining
                                                           capacity (0
                                                           once
                                                           theta[n-1]
                                                           reaches
                                                           theta_s)
        flux[face]  = min(demand[n], supply[n-1])

    This makes saturation a genuine steady state (flux -> 0 exactly,
    not just slowly) rather than a numerical singularity, and keeps
    the scheme exactly mass-conservative for any dt within the
    transport CFL bound.
    """
    N = theta.size
    demand = _K_brooks_corey(theta, cfg)
    relsat = np.clip((theta - cfg.theta_r) / max(cfg.theta_s - cfg.theta_r, 1e-30),
                      0.0, 1.0)
    supply = cfg.K_sat * (1.0 - relsat ** cfg.n_perc)

    F_face = np.zeros(N + 1)
    # Face between cell n-1 (receiver, below) and cell n (donor, above),
    # n=1..N-1: flow is downward (donor -> receiver).
    F_face[1:N] = -np.minimum(demand[1:N], supply[0:N-1])
    return F_face


def _rhs(theta: np.ndarray, T2: np.ndarray, dz: float,
         cfg: PercolationConfig) -> tuple:
    """
    Shared RHS builder used by both advance_percolation() and
    adaptive_dt(), so the dt bound is always computed from the exact
    same explicit operator that will then be applied -- if these ever
    drifted out of sync (e.g. someone updates the flux scheme in one
    place but not the other) the CFL bound would silently stop meaning
    anything.

    Returns
    -------
    rhs         : shape (N,), d(theta)/dt from transport + sources
    Gamma_melt  : shape (N,)
    Gamma_flash : shape (N,)
    """
    Gamma_melt  = melt_source(theta, T2, cfg)
    Gamma_flash = flash_sink(theta, T2, cfg)

    F_face = _limited_face_fluxes(theta, cfg)
    dtheta_transport = -(F_face[1:] - F_face[:-1]) / dz
    rhs = dtheta_transport + Gamma_melt - Gamma_flash
    return rhs, Gamma_melt, Gamma_flash


def adaptive_dt(theta: np.ndarray, T2: np.ndarray, dz: float,
                 cfg: PercolationConfig, dt_cap: float = np.inf) -> float:
    """
    Safe explicit timestep for advance_percolation(): the supply/demand
    flux limiter in _limited_face_fluxes() makes saturation a genuine
    steady state rather than a source of unbounded stiffness, so the
    only bound needed here is ordinary Godunov transport CFL,
    dt <= 0.5*dz / max(dK/dtheta) sampled on demand AND supply's slopes
    (both can be as steep as K_sat*n_perc near their respective
    saturation limits).

    Mirrors solver.py's _adaptive_dt pattern: called BEFORE the step,
    not enforced internally by advance_percolation.
    """
    if not cfg.enabled or cfg.K_sat <= 0:
        return dt_cap

    eps = 1e-6
    demand_hi = _K_brooks_corey(theta + eps, cfg)
    demand_lo = _K_brooks_corey(theta, cfg)
    speed_max = float(np.max(np.abs(demand_hi - demand_lo))) / eps

    relsat = np.clip((theta - cfg.theta_r) / max(cfg.theta_s - cfg.theta_r, 1e-30),
                      0.0, 1.0)
    relsat_hi = np.clip((theta + eps - cfg.theta_r) / max(cfg.theta_s - cfg.theta_r, 1e-30),
                         0.0, 1.0)
    supply = cfg.K_sat * (1.0 - relsat ** cfg.n_perc)
    supply_hi = cfg.K_sat * (1.0 - relsat_hi ** cfg.n_perc)
    speed_max = max(speed_max, float(np.max(np.abs(supply_hi - supply))) / eps)

    if speed_max < 1e-30:
        return dt_cap
    return min(dt_cap, 0.5 * dz / speed_max)


def advance_percolation(theta: np.ndarray, T2: np.ndarray,
                         dz: float, dt: float,
                         cfg: PercolationConfig) -> tuple:
    """
    Advance theta by dt with explicit upwind finite volumes.

    Parameters
    ----------
    theta : shape (N,), current liquid saturation
    T2    : shape (N,), combustible-phase temperature (drives melt/flash)
    dz    : cell width, cm
    dt    : timestep, s -- caller MUST respect adaptive_dt() (not
            enforced internally, same division of responsibility as
            thermal.py's own step functions vs. solver.py's
            _adaptive_dt). Using a dt larger than adaptive_dt() returns
            can overshoot theta_s/theta_r and break mass conservation
            (see adaptive_dt's docstring).
    cfg   : PercolationConfig

    Returns
    -------
    theta_new   : shape (N,)
    Gamma_melt  : shape (N,) -- returned so the caller can apply
                  thermal_sink() to the SAME source used here, rather
                  than recomputing it from theta_new and risking it
                  drifting out of sync.
    Gamma_flash : shape (N,)
    """
    if not cfg.enabled:
        z = np.zeros_like(theta)
        return theta.copy(), z, z

    rhs, Gamma_melt, Gamma_flash = _rhs(theta, T2, dz, cfg)
    theta_new = theta + dt * rhs

    # Safety-net clip only: with dt chosen via adaptive_dt() this should
    # be a no-op to within floating-point noise, not the mechanism that
    # enforces the bounds (see adaptive_dt's docstring for why a
    # clip-as-primary-mechanism silently breaks mass conservation).
    theta_new = np.clip(theta_new, cfg.theta_r, cfg.theta_s if cfg.theta_s > 0 else 1.0)

    return theta_new, Gamma_melt, Gamma_flash


def correlation_length(theta: np.ndarray, cfg: PercolationConfig) -> np.ndarray:
    """ell(z,t), cm. Saturating sigmoid in s=theta/theta_c -- see module
    docstring. Returns a uniform ell0 field when disabled."""
    if not cfg.enabled or cfg.theta_c <= 0:
        return np.full_like(theta, cfg.ell0)
    s = theta / cfg.theta_c
    sp = s ** cfg.p_perc
    return cfg.ell0 + (cfg.ell_max - cfg.ell0) * sp / (cfg.s0_perc ** cfg.p_perc + sp)


def mu_field(theta: np.ndarray, cfg: PercolationConfig) -> np.ndarray:
    """mu(z,t) = 1/ell(z,t), cm^-1 -- drop-in replacement for the old
    scalar Config.mu everywhere it's consumed (thermal.py,
    neutronics.py)."""
    return 1.0 / correlation_length(theta, cfg)


# ─────────────────────────────────────────────────────────────────────
# Standalone self-test (pure numpy, no scipy needed -- run directly:
# `python percolation.py`). Exercises mass conservation, drainage
# direction, and the mu(theta) mapping in isolation from the rest of
# the solver, which does depend on scipy.
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    N, H = 40, 40.0     # cm
    dz = H / N
    z  = np.linspace(dz / 2, H - dz / 2, N)

    cfg = PercolationConfig(
        enabled=True,
        theta_r=0.02, theta_s=0.30,
        T_melt=400.0, T_boil=900.0,
        k_melt=0.05, k_flash=0.02,
        L_fus=2.0e5, L_vap=8.0e5, rho_liquid=900.0,
        K_sat=0.05, n_perc=3.0,
        ell0=1.0/0.3, ell_max=15.0, theta_c=0.25, s0_perc=1.0, p_perc=4.0,
    )

    theta = initialise_theta(cfg, N)

    # Synthetic temperature field: hot at z=0 (furnace end), cooling
    # toward z=H, rising slowly in time -- enough to trigger melting
    # near the bottom early, and (eventually) flashing there once hot
    # enough, without needing the full coupled thermal solver.
    def T2_of(t):
        return 300.0 + 700.0 * np.exp(-z / 12.0) * np.tanh(t / 40.0)

    dt_report = 20.0
    t, t_next_report = 0.0, 0.0
    total_melted = 0.0
    total_flashed = 0.0
    # NOTE: the conserved quantity for this finite-VOLUME scheme is the
    # plain cell sum dz*sum(theta), not np.trapezoid(theta, z) -- trapz
    # is a different quadrature rule (endpoint-weighted) that doesn't
    # match a box/Riemann sum over cell-centred values, so using it
    # here would report a spurious O(dz) "leak" that isn't actually
    # there. Same distinction applies to the melt/flash source
    # integrals below, for consistency.
    mass_in_domain_prev = dz * np.sum(theta)
    theta_t40 = None   # snapshot before flashing starts, for the
                        # drainage-direction check below

    print(f"{'t':>7} {'max theta':>10} {'z@max':>7} {'mu min..max (cm^-1)':>22} "
          f"{'sum melt':>10} {'sum flash':>10}")

    t_end = 300.0
    n_steps = 0
    while t < t_end:
        n_steps += 1
        if n_steps > 200_000:
            raise RuntimeError(f"self-test did not converge in time: t={t}, dt collapsing")
        T2 = T2_of(t)
        dt = min(adaptive_dt(theta, T2, dz, cfg), 0.5, t_end - t)
        theta, Gm, Gf = advance_percolation(theta, T2, dz, dt, cfg)
        total_melted  += dz * np.sum(Gm) * dt
        total_flashed += dz * np.sum(Gf) * dt
        t += dt

        if theta_t40 is None and t >= 40.0:
            theta_t40 = theta.copy()

        if t >= t_next_report:
            mu = mu_field(theta, cfg)
            imax = int(np.argmax(theta))
            print(f"{t:7.1f} {theta[imax]:10.4f} {z[imax]:7.2f} "
                  f"[{mu.min():8.4f}, {mu.max():8.4f}]      "
                  f"{total_melted:10.4f} {total_flashed:10.4f}")
            t_next_report += dt_report

    # --- sanity checks ---
    mass_final = dz * np.sum(theta)
    balance_residual = (mass_final - mass_in_domain_prev
                         - (total_melted - total_flashed))
    print(f"\nMass balance residual (should be ~0): {balance_residual:.3e}")
    print(f"theta profile (bottom->top): min={theta.min():.4f} "
          f"max={theta.max():.4f} argmax_z={z[np.argmax(theta)]:.2f}cm")

    # Drainage-direction check uses the t=40s snapshot, BEFORE flash-off
    # (T_boil=900K) has had time to deplete the hot bottom cells --
    # later in the run the peak legitimately migrates upward as the
    # bottom flashes off faster than it's resupplied, which is correct
    # physics (see the flash sum climbing in the printed table above),
    # not a transport bug.
    assert z[np.argmax(theta_t40)] < H / 4, (
        "expected early-time drainage to concentrate theta toward z=0 "
        "(the heated bottom), but the peak at t=40s is not near it"
    )
    assert abs(balance_residual) < 1e-8 * max(mass_final, 1.0), (
        "mass balance residual too large -- transport scheme is not "
        "conservative"
    )
    mu_disabled = mu_field(theta, PercolationConfig(enabled=False, ell0=cfg.ell0))
    assert np.allclose(mu_disabled, 1.0 / cfg.ell0), (
        "disabled config should return the uniform dry mu0 everywhere"
    )
    print("\nAll self-checks passed.")
