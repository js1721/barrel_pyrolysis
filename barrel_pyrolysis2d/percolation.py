"""
percolation.py  (2D axisymmetric r-z)
======================================
Moderator percolation in the 2D cylindrical tree: melting, gravity
drainage, and flash-off of a mobile liquid component within the
combustible phase, and the resulting dynamic correlation length
mu(r,z,t) fed back into thermal.py and neutronics.py in place of the
constant Config.mu.

This is the 2D generalization of barrel_pyrolysis1d/percolation.py.
The physics, the constitutive laws, and the Cell Transmission Model
flux limiter are unchanged; what follows documents precisely what the
extra dimension does and does NOT change.

Why transport stays one-dimensional
------------------------------------
The document's transport equation is a kinematic-wave (Green-Ampt /
Brooks-Corey) reduction of Darcy's law, valid when capillary pressure
gradients are negligible against the gravitational term. Gravity is
along -z; in an axisymmetric r-z geometry there is NO radial body
force. So under the document's own stated approximation there is no
radial driving term to generalize: a radial liquid flux would be
purely capillary, which is exactly what the kinematic-wave reduction
discards.

Consequently the transport operator remains

    d(theta)/dt = d/dz[K(theta)] + Gamma_melt - Gamma_flash

applied INDEPENDENTLY along each radial ring i. This is not a
simplifying approximation layered on top of the document -- it is what
the document's approximation already implies once the geometry is
fixed.

The rings are NOT physically decoupled, however: theta(r,z,t) is a
genuinely two-dimensional field, because Gamma_melt carries
H(T_2 - T_melt) and T_2(r,z,t) is two-dimensional. Radial structure in
theta arises entirely through the temperature field, not through
liquid flux. A column that never gets hot never melts, regardless of
what its neighbours do.

Why the cylindrical metric does not appear
-------------------------------------------
Because the flux is purely axial, its divergence is

    div(K z_hat) = d(K)/dz

with no 1/r * d(r * .)/dr metric factor -- the annular cell volume
r_{i+1/2}^2 - r_{i-1/2}^2 is common to both faces of an axial flux and
cancels out of the finite-volume balance. The 1D scheme (including the
supply/demand flux limiter and its CFL bound) therefore transfers
VERBATIM per ring, and remains exactly mass-conservative ring by ring.

Had a radial flux been retained, this would not hold: the radial
finite-volume balance carries r_edge face areas, the flux limiter's
supply/demand pairing would have to be rederived on annular control
volumes, and the CFL bound would pick up a dr-dependent term. That
work is deliberately not done here, because the model has no radial
flux to justify it.

Array convention
-----------------
All fields are shape (Nr, Nz), matching CylindricalMesh2D's (i,j)
ordering: i indexes r, j indexes z. Transport acts along axis=1.
j=0 is the HEATED bottom (z = dz/2) and gravity drains toward
DECREASING j, exactly as in 1D.

Units: r,z in cm, mu in cm^-1, K_sat in cm/s, T in K, latent heats in
J/kg via rho_liquid (kg/m^3); thermal_sink() returns W/m^3 to match
thermal.py's existing source-term convention.
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

    Identical field-for-field to the 1D PercolationConfig: the
    parameters are constitutive properties of the waste form, not of
    the mesh, so nothing here is dimension-dependent.
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
    ell0:    float = 1.0 / 0.3   # dry correlation length, cm
    ell_max: float = 20.0     # capped max correlation length, cm
    theta_c: float = 0.3      # percolation threshold saturation
    s0_perc: float = 1.0      # sigmoid centre (in units of theta/theta_c)
    p_perc:  float = 4.0      # sigmoid sharpness


def initialise_theta(cfg: PercolationConfig, Nr: int, Nz: int) -> np.ndarray:
    """theta(r, z, 0) = theta_r uniformly (bone dry above the
    irreducible floor). Shape (Nr, Nz)."""
    return np.full((Nr, Nz), cfg.theta_r)


def _K_brooks_corey(theta: np.ndarray, cfg: PercolationConfig) -> np.ndarray:
    """Relative hydraulic conductivity, cm/s. Zero at/below theta_r.
    Elementwise -- shape-agnostic."""
    frac = np.clip((theta - cfg.theta_r) / max(cfg.theta_s - cfg.theta_r, 1e-30),
                    0.0, 1.0)
    return cfg.K_sat * frac ** cfg.n_perc


def melt_source(theta: np.ndarray, T2: np.ndarray,
                 cfg: PercolationConfig) -> np.ndarray:
    """Gamma_melt(r,z,t), s^-1 saturation units. Zero wherever
    T2 <= T_melt. Elementwise, so the (Nr,Nz) field is handled with no
    change from the 1D form."""
    if not cfg.enabled:
        return np.zeros_like(theta)
    active = T2 > cfg.T_melt
    return np.where(active, cfg.k_melt * (cfg.theta_s - theta), 0.0)


def flash_sink(theta: np.ndarray, T2: np.ndarray,
               cfg: PercolationConfig) -> np.ndarray:
    """Gamma_flash(r,z,t), s^-1 saturation units. Zero wherever
    T2 <= T_boil."""
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
    S_f/S_c convention exactly (a per-total-volume quantity, so no
    extra phase-density factor).
    """
    if not cfg.enabled:
        return np.zeros_like(Gamma_melt)
    return cfg.rho_liquid * (cfg.L_fus * Gamma_melt + cfg.L_vap * Gamma_flash)


def _limited_face_fluxes(theta: np.ndarray, cfg: PercolationConfig) -> np.ndarray:
    """
    Supply/demand-limited AXIAL face fluxes (Cell Transmission Model,
    Daganzo 1994), cm/s, shape (Nr, Nz+1) with F_face[i, j] the flux at
    the -z face of cell (i, j). F_face[:, 0] and F_face[:, Nz] are the
    closed outer boundaries, always 0.

    A plain upwind scheme (demand-only: flux = K(theta[donor])) lets a
    cell keep receiving inflow forever even after it reaches theta_s,
    since K(theta) depends only on the SENDING cell -- which either
    overshoots theta_s (silently destroying mass when clipped post-hoc)
    or forces the adaptive timestep to collapse toward zero forever
    once any cell saturates under continued upstream inflow. The fix is
    to also cap flux by the RECEIVING cell's remaining capacity:

        demand[i,j]   = K(theta[i,j])                      -- donor supply
        supply[i,j-1] = K_sat*(1 - relsat[i,j-1]^n_perc)   -- receiver's
                                                              remaining
                                                              capacity
        flux[face]    = min(demand[i,j], supply[i,j-1])

    This makes saturation a genuine steady state (flux -> 0 exactly)
    and keeps the scheme exactly mass-conservative per ring for any dt
    within the transport CFL bound.

    The radial index is a pure spectator here: each ring's axial
    balance is independent (see module docstring), so this is the 1D
    operator broadcast over axis 0, not a new scheme.
    """
    Nr, Nz = theta.shape
    demand = _K_brooks_corey(theta, cfg)
    relsat = np.clip((theta - cfg.theta_r) / max(cfg.theta_s - cfg.theta_r, 1e-30),
                      0.0, 1.0)
    supply = cfg.K_sat * (1.0 - relsat ** cfg.n_perc)

    F_face = np.zeros((Nr, Nz + 1))
    # Face between cell j-1 (receiver, below) and cell j (donor, above),
    # j=1..Nz-1: flow is downward (donor -> receiver), hence negative.
    F_face[:, 1:Nz] = -np.minimum(demand[:, 1:Nz], supply[:, 0:Nz-1])
    return F_face


def _rhs(theta: np.ndarray, T2: np.ndarray, dz: float,
         cfg: PercolationConfig) -> tuple:
    """
    Shared RHS builder used by both advance_percolation() and
    adaptive_dt(), so the dt bound is always computed from the exact
    same explicit operator that will then be applied.

    Returns
    -------
    rhs         : shape (Nr, Nz), d(theta)/dt from transport + sources
    Gamma_melt  : shape (Nr, Nz)
    Gamma_flash : shape (Nr, Nz)
    """
    Gamma_melt  = melt_source(theta, T2, cfg)
    Gamma_flash = flash_sink(theta, T2, cfg)

    F_face = _limited_face_fluxes(theta, cfg)
    # Axial divergence only -- no 1/r metric factor, see module docstring.
    dtheta_transport = -(F_face[:, 1:] - F_face[:, :-1]) / dz
    rhs = dtheta_transport + Gamma_melt - Gamma_flash
    return rhs, Gamma_melt, Gamma_flash


def adaptive_dt(theta: np.ndarray, T2: np.ndarray, dz: float,
                 cfg: PercolationConfig, dt_cap: float = np.inf) -> float:
    """
    Safe explicit timestep for advance_percolation(): ordinary Godunov
    transport CFL, dt <= 0.5*dz / max(dK/dtheta), with the slope
    sampled on BOTH demand and supply (each can be as steep as
    K_sat*n_perc near its respective saturation limit).

    The maximum is taken over the whole (Nr, Nz) field, i.e. the most
    restrictive ring sets the timestep for all of them. Since the rings
    share the global solver timestep anyway, a per-ring dt would buy
    nothing.

    Note only dz enters: with no radial flux there is no dr-dependent
    CFL contribution, so REFINING THE RADIAL MESH COSTS NOTHING in
    timestep terms. That is a direct consequence of the axial-only
    transport and is worth keeping in mind when choosing Nr.

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
    Advance theta by dt with explicit upwind finite volumes, per ring.

    Parameters
    ----------
    theta : shape (Nr, Nz), current liquid saturation
    T2    : shape (Nr, Nz), combustible-phase temperature
    dz    : axial cell width, cm
    dt    : timestep, s -- caller MUST respect adaptive_dt() (not
            enforced internally, same division of responsibility as
            thermal.py's step functions vs. solver.py's _adaptive_dt).
    cfg   : PercolationConfig

    Returns
    -------
    theta_new   : shape (Nr, Nz)
    Gamma_melt  : shape (Nr, Nz) -- returned so the caller applies
                  thermal_sink() to the SAME source used here, rather
                  than recomputing it from theta_new and risking drift.
    Gamma_flash : shape (Nr, Nz)
    """
    if not cfg.enabled:
        z = np.zeros_like(theta)
        return theta.copy(), z, z

    rhs, Gamma_melt, Gamma_flash = _rhs(theta, T2, dz, cfg)
    theta_new = theta + dt * rhs

    # Safety-net clip only: with dt from adaptive_dt() this is a no-op
    # to within floating-point noise, not the mechanism enforcing bounds.
    theta_new = np.clip(theta_new, cfg.theta_r, cfg.theta_s if cfg.theta_s > 0 else 1.0)

    return theta_new, Gamma_melt, Gamma_flash


def correlation_length(theta: np.ndarray, cfg: PercolationConfig) -> np.ndarray:
    """ell(r,z,t), cm. Saturating sigmoid in s=theta/theta_c. Returns a
    uniform ell0 field when disabled. Elementwise -- shape-agnostic."""
    if not cfg.enabled or cfg.theta_c <= 0:
        return np.full_like(theta, cfg.ell0)
    s = theta / cfg.theta_c
    sp = s ** cfg.p_perc
    return cfg.ell0 + (cfg.ell_max - cfg.ell0) * sp / (cfg.s0_perc ** cfg.p_perc + sp)


def mu_field(theta: np.ndarray, cfg: PercolationConfig) -> np.ndarray:
    """
    mu(r,z,t) = 1/ell(r,z,t), cm^-1, shape (Nr, Nz).

    Drop-in replacement for the old scalar Config.mu everywhere it is
    consumed (thermal.py, neutronics.py). Note the correlation length
    is a SCALAR microstructural property at each point, so the mixture
    stays locally isotropic: mu_r = mu_z = mu(r,z,t) pointwise, which
    is the document's cylindrical note (R = exp(-mu_r r - mu_z z) with
    mu_r = mu_z) promoted to a spatially varying field. It does NOT
    become an anisotropic tensor just because drainage has a preferred
    direction -- drainage sets the theta distribution, and theta then
    sets an isotropic ell at each point.
    """
    return 1.0 / correlation_length(theta, cfg)
