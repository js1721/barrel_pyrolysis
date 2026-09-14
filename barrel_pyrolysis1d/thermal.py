"""
thermal.py
==========
Single-group 1D slab heat conduction and pyrolysis.

Thermal equation (Eq. 95 of document):

    rho_i Cp_i dT_i/dt = K_i d^2T_i/dz^2
                        + mu * dT_i/dz
                        - mu * sum_k v_k * dT_k/dz
                        - mu * sum_{l!=i} (K_l * dT_l/dz
                                           + mu * T_l)
                        + Ef * sum_g Sigma_f,i,g * phi_i,g
                        - q_i * d(omega_i)/dt

    mu may now be a spatially-varying field (N,), driven by
    percolation.py's dynamic correlation length, rather than the
    single constant Config.mu -- everywhere below that used to read
    "the" mu now reads mu(z) at that cell. Passing a plain scalar still
    works unchanged (broadcast to a uniform field), so this is a
    backward-compatible generalisation, not a behaviour change, for
    any caller that keeps passing a scalar.

    NOTE on units: rho_i here (and rho_i*Cp_i on the LHS) is the WHOLE
    material's bulk density (materials.py Phase.rho) -- omega_i is NOT
    a dimensionless mass fraction, it is the density (kg/m^3) of just
    the COMBUSTIBLE/volatile fraction of that bulk material (distinct
    from rho_i, since e.g. combustible waste is only partly volatile
    organic content -- the rest is inert filler/residue that never
    pyrolyses; see materials.py's omega0 comment). d(omega_i)/dt is
    therefore already a volumetric mass-consumption rate (kg/m^3/s),
    so q_i*d(omega_i)/dt = W/m^3 directly, matching rho_i*Cp_i's per-
    volume convention with no extra rho_i factor needed on this term.
    (An earlier version of this code used a dimensionless omega in
    [0,1] and needed an explicit rho_i factor here instead -- both the
    equation and materials.py's omega0 were changed together.)

Expanding for two phases (i=1, k,l=2 and i=2, k,l=1):

    Phase 1:
        rho_1 Cp_1 dT_1/dt = K_1 d^2T_1/dz^2
            + mu * dT_1/dz
            - mu * (v_1*dT_1/dz + v_2*dT_2/dz)
            - mu * (K_2 * dT_2/dz + mu * T_2)
            + S_f,1 + S_c,1

    Phase 2:
        rho_2 Cp_2 dT_2/dt = K_2 d^2T_2/dz^2
            + mu * dT_2/dz
            - mu * (v_1*dT_1/dz + v_2*dT_2/dz)
            - mu * (K_1 * dT_1/dz + mu * T_1)
            + S_f,2 + S_c,2

Collecting terms for phase 1:
    = K_1 d^2T_1/dz^2
      + mu*(1-v_1)*dT_1/dz
      - mu*(v_2 + K_2/K_1... wait, keep separate)
      - mu*v_2*dT_2/dz
      - mu*K_2*dT_2/dz
      - mu^2*T_2
      + sources

i.e. the gradient terms from the other phase enter
with coefficient -(mu*v_2 + mu*K_2) = -mu*(v_2 + K_2).

Pyrolysis (Eq. 96):
    d(omega_i)/dt = -k_i * omega_i * exp(-Ea_i/Rg/T_i)

Boundary conditions:
    z=0 : Eqs. (71)-(72) convective + radiative (heated, T_f)
    z=H : Eqs. (67)-(68) insulating by default; optionally convective +
          radiative heat loss to an ambient exterior (T_amb) via the
          same H_flux form as z=0 -- see solve_thermal_step's T_amb
          parameter.
"""

import numpy as np
from scipy.linalg import solve_banded

from materials import Phase
from mesh import Mesh1D

Rg       = 8.314
SIGMA_SB = 5.67e-8


def _banded_matvec(AB: np.ndarray, x: np.ndarray) -> np.ndarray:
    """A @ x for a tridiagonal A stored in solve_banded((1,1),...) format."""
    y = AB[1, :] * x
    y[:-1] += AB[0, 1:] * x[1:]
    y[1:]  += AB[2, :-1] * x[:-1]
    return y

# mesh.dz/mu are in cm (shared with neutronics.py), but Phase.K/rho/Cp
# are SI (per-metre). Convert lengths to metres before using them here.
CM_TO_M = 1.0e-2


def H_flux(T_surf: float,
            T_f: float,
            h_conv: float,
            emissivity: float) -> float:
    """
    Combined convective + radiative flux at z=0.

    From Eq. (61):
        H(T) = h*(T_f - T) + sigma*eps*(T_f^4 - T^4)

    Positive = heat into barrel.
    """
    return (h_conv * (T_f - T_surf)
            + SIGMA_SB * emissivity
            * (T_f**4 - T_surf**4))


def arrhenius_rate(ph: Phase,
                    T: np.ndarray,
                    omega: np.ndarray) -> np.ndarray:
    """
    Pointwise Arrhenius pyrolysis rate.

    From Eq. (96): -d(omega)/dt = k*omega*exp(-Ea/Rg/T)

    omega is a density (kg/m^3 of combustible content, not a
    dimensionless fraction -- see materials.py's omega0 comment), so
    this returns a volumetric mass-consumption rate, kg/m^3/s (the
    ODE's form is unchanged by omega's scale, since it's linear/
    homogeneous in omega).
    """
    exponent = np.where(
        T > 1.0,
        -ph.E_act / (Rg * T),
        -1e10
    )
    return ph.k_arr * omega * np.exp(exponent)


def advance_pyrolysis(phases: list,
                       T: np.ndarray,
                       omega: np.ndarray,
                       dt: float) -> np.ndarray:
    """
    Advance omega via implicit Euler.

    d(omega_i)/dt = -k_i * omega_i * exp(-Ea_i/Rg/T_i)

    omega is the density (kg/m^3) of combustible content remaining,
    not a dimensionless fraction -- see materials.py's omega0 comment.

    IC: omega(z,0) = omega0 set at initialisation.
    """
    omega_new = omega.copy()
    for p, ph in enumerate(phases):
        if ph.k_arr < 1e-30:
            continue
        rate         = arrhenius_rate(ph, T[p], omega[p])
        omega_new[p] = omega[p] / (1.0 + dt * rate)
        omega_new[p] = np.clip(omega_new[p], 0.0, ph.omega0)
    return omega_new


def initialise_omega(phases: list,
                      mesh: Mesh1D) -> np.ndarray:
    """omega(z,0) = omega0 uniformly. Shape (2, N)."""
    omega = np.zeros((2, mesh.N))
    for p, ph in enumerate(phases):
        omega[p] = np.full(mesh.N, ph.omega0)
    return omega


def _compute_gradients(T: np.ndarray,
                        dz: float) -> np.ndarray:
    """
    Compute dT/dz at cell centres via central differences.
    One-sided at boundaries.

    Parameters
    ----------
    T  : temperature field shape (N,)
    dz : cell width cm

    Returns
    -------
    dTdz : shape (N,)
    """
    dTdz       = np.zeros_like(T)
    # Interior: central difference
    dTdz[1:-1] = (T[2:] - T[:-2]) / (2.0 * dz)
    # Boundaries: one-sided
    dTdz[0]    = (T[1]  - T[0])  / dz
    dTdz[-1]   = (T[-1] - T[-2]) / dz
    return dTdz


def _build_conduction_matrix(ph: Phase,
                               mesh: Mesh1D,
                               dt: float,
                               mu,
                               v_self: float) -> np.ndarray:
    """
    Build banded Crank-Nicolson matrix for 1D conduction
    including the self-phase stochastic drift term.

    The equation for phase i contains:
        K_i d^2T_i/dz^2 + mu*(1 - v_i)*dT_i/dz

    The term mu*(1-v_i)*dT_i/dz is a first-order
    advection term. We discretise it with upwind
    differencing for stability (mu > 0, so flow is
    in +z direction):

        mu*(1-v_i) * (T_n - T_{n-1}) / dz   (upwind)

    Combined with CN diffusion this gives a
    non-symmetric tridiagonal system.

    Parameters
    ----------
    ph     : Phase
    mesh   : Mesh1D
    dt     : timestep s
    mu     : inverse correlation length cm^-1 -- scalar OR shape (N,).
             A scalar is broadcast to a uniform field; the discretely
             correct treatment below needs mu indexed PER CELL once
             it's spatially varying (see the AB[2,:-1] note), so it's
             promoted to an (N,) array immediately regardless.
    v_self : volume fraction of this phase

    Returns
    -------
    AB : banded matrix (3, N) for solve_banded
    """
    N    = mesh.N
    dz   = mesh.dz * CM_TO_M
    K    = ph.K
    rCp  = ph.rho * ph.Cp
    mu   = np.broadcast_to(np.atleast_1d(mu), (N,)).astype(float)
    mu_m = mu / CM_TO_M   # cm^-1 -> m^-1, now shape (N,)

    # Diffusion coefficient
    r = K * dt / (rCp * dz**2)

    # Self-phase drift coefficient: mu*(1 - v_self), per cell now
    # Upwind: positive drift in +z direction
    s = mu_m * (1.0 - v_self) * dt / (rCp * dz)   # shape (N,)

    AB        = np.zeros((3, N))

    # Main diagonal: diffusion + upwind drift (cell n's own mu[n])
    AB[1, :]  = 1.0 + r + 0.5 * s

    # Upper diagonal (n+1): diffusion only (CN) -- upwind puts no
    # drift contribution here regardless of mu, unaffected by mu being
    # spatially varying.
    AB[0, 1:] = -0.5 * r

    # Lower diagonal (n-1): diffusion + upwind drift. AB[2, j] is the
    # coefficient in ROW j+1's equation multiplying cell j (scipy's
    # solve_banded((1,1),...) convention: AB[2,j] = a[j+1,j]) -- so the
    # drift coefficient here must be evaluated at the ROW's cell,
    # mu[j+1], i.e. s[1:], NOT s[:-1]. With a uniform/scalar mu this
    # distinction was invisible (s was the same everywhere); it matters
    # now that mu can vary cell-to-cell.
    AB[2,:-1] = -0.5 * r - 0.5 * s[1:]

    # --- z=0 boundary ---
    # One-sided upwind for drift, one-sided diffusion
    # Heat flux BC applied via RHS
    AB[1, 0]  = 1.0 + 0.5 * r
    AB[0, 1]  = -0.5 * r

    # --- z=H boundary ---
    # Insulating: dT/dz = 0, so drift term vanishes
    AB[1, -1] = 1.0 + 0.5 * r
    AB[2, -2] = -0.5 * r

    return AB


def solve_thermal_step(phases: list,
                        mesh: Mesh1D,
                        T: np.ndarray,
                        omega: np.ndarray,
                        phi: np.ndarray,
                        v: np.ndarray,
                        mu,
                        dt: float,
                        T_f: float,
                        h_conv: float,
                        emissivity: float,
                        Ef: float = 3.2e-11,
                        T_amb: float = None,
                        h_conv_amb: float = None,
                        emissivity_amb: float = None,
                        S_extra: np.ndarray = None) -> np.ndarray:
    """
    Advance temperature for both phases by dt.

    Implements Eq. (95) exactly:

        rho_i Cp_i dT_i/dt = K_i d^2T_i/dz^2
            + mu * dT_i/dz
            - mu * sum_k v_k * dT_k/dz
            - mu * sum_{l!=i} (K_l * dT_l/dz + mu*T_l)
            + Ef * Sigma_f,i * phi_i
            - q_i * d(omega_i)/dt

    The implicit matrix handles:
        K_i d^2T_i/dz^2  +  mu*(1-v_i)*dT_i/dz

    The explicit RHS handles:
        - mu * v_other * dT_other/dz
        - mu * K_other * dT_other/dz
        - mu^2 * T_other
        + sources

    Parameters
    ----------
    phases         : [Phase1, Phase2]
    mesh           : Mesh1D
    T              : temperatures shape (2, N)
    omega          : density of combustible content, kg/m^3, shape (2, N)
    phi            : flux shape (2, 2, N) [phase, group, space]
    v              : volume fractions shape (2,)
    mu             : inverse correlation length cm^-1 -- scalar OR
                     shape (N,). A spatially-varying field (e.g. from
                     percolation.mu_field()) is used as-is; a scalar is
                     broadcast to a uniform field, reproducing the
                     original constant-mu behaviour exactly.
    dt             : timestep s
    T_f            : furnace temperature K (z=0, heated boundary)
    h_conv         : convective HTC W/m^2/K (z=0)
    emissivity     : surface emissivity (z=0)
    T_amb          : ambient temperature K (z=H, heat-loss boundary).
                     Defaults to T_f's counterpart -- pass explicitly
                     for a genuinely cool exterior (e.g. 300.0).
                     None disables the z=H flux term (recovers the
                     previous purely-insulating boundary).
    h_conv_amb     : convective HTC W/m^2/K (z=H). Defaults to h_conv.
    emissivity_amb : surface emissivity (z=H). Defaults to emissivity.
    S_extra        : optional additional explicit volumetric source,
                     W/m^3, shape (2, N) -- e.g. percolation.py's
                     thermal_sink() (already negative/subtracted, i.e.
                     pass -thermal_sink(...) if it represents a loss;
                     see solver.py). None (default) adds nothing,
                     reproducing the previous behaviour exactly.

    Returns
    -------
    T_new : shape (2, N)
    """
    N     = mesh.N
    dz    = mesh.dz * CM_TO_M
    mu    = np.broadcast_to(np.atleast_1d(mu), (N,)).astype(float)
    mu_m  = mu / CM_TO_M   # cm^-1 -> m^-1, shape (N,)
    T_new = T.copy()
    if h_conv_amb is None:
        h_conv_amb = h_conv
    if emissivity_amb is None:
        emissivity_amb = emissivity

    # Compute gradients of both phases explicitly
    # These are used in the cross-phase coupling terms
    dTdz = np.array([
        _compute_gradients(T[p], dz)
        for p in range(2)
    ])   # shape (2, N)

    for p, ph in enumerate(phases):
        p_other = 1 - p
        ph_oth  = phases[p_other]
        rCp     = ph.rho * ph.Cp

        # Build implicit matrix:
        # handles K_i d^2T_i/dz^2 + mu*(1-v_i)*dT_i/dz
        AB = _build_conduction_matrix(
            ph, mesh, dt, mu, v[p]
        )

        # ── Explicit source terms ──────────────────────────

        # 1. Cross-phase drift from sum_k v_k dT_k/dz
        #    The v_i*dT_i/dz part is in the implicit matrix
        #    (absorbed into the (1-v_i) coefficient).
        #    Here we add -mu * v_other * dT_other/dz
        S_drift_other = -mu_m * v[p_other] * dTdz[p_other]

        # 2. Conductivity-weighted gradient of other phase
        #    From: -mu * sum_{l!=i} K_l * dT_l/dz
        S_K_other = -mu_m * ph_oth.K * dTdz[p_other]

        # 3. Direct temperature coupling
        #    From: -mu * sum_{l!=i} mu * T_l = -mu^2 * T_other
        S_T_other = -mu_m**2 * T[p_other]

        # 4. Fission source: Ef * sum_g Sigma_f,g(T) * phi_g
        #    phi is now per-group (fast, thermal) -- sum the fission
        #    heat contribution across both groups.
        T_p = float(np.mean(T[p]))
        Sf  = ph.Sf_T(T_p)   # [fast, thermal]
        S_f = Ef * (Sf[0] * phi[p][0] + Sf[1] * phi[p][1])

        # 5. Combustion source: q * (-d(omega)/dt)
        #    q is heat of combustion per unit MASS (J/kg, see
        #    materials.py); omega is the density (kg/m^3) of
        #    combustible content, NOT a dimensionless fraction, so
        #    arrhenius_rate already returns a volumetric mass-
        #    consumption rate (kg/m^3/s) -- q*rate is W/m^3 directly,
        #    matching S_f and rCp's per-volume convention with no
        #    separate rho factor needed here (rho is the WHOLE
        #    material's bulk density, a distinct quantity -- see
        #    materials.py's omega0 comment).
        S_c = ph.q * arrhenius_rate(ph, T[p], omega[p])

        # Total explicit source
        S_total = (S_drift_other
                   + S_K_other
                   + S_T_other
                   + S_f
                   + S_c)

        if S_extra is not None:
            S_total = S_total + S_extra[p]

        # AB is a Crank-Nicolson matrix (I - 0.5*L), so the CN update
        # AB@T_new = (I + 0.5*L)@T_old + dt/rCp*S_total needs the
        # explicit half of the SAME diffusion/drift operator applied
        # to T_old -- (I+0.5L) = 2I-AB -- not just T_old alone (that
        # would silently drop the old-timestep diffusion/drift
        # contribution, verified against an analytical semi-infinite
        # conduction solution to diverge with mesh/timestep refinement
        # rather than converge -- see verification_tests.py).
        rhs = 2.0 * T[p] - _banded_matvec(AB, T[p]) + dt / rCp * S_total

        # ── Boundary conditions ────────────────────────────

        # z=0: Eqs. (71)-(72)
        # Phase 1: -K1*T1' - K1*(1-v)*mu*(T1-T2) = H(T1)
        # Phase 2: -K2*T2' + K2*v*mu*(T1-T2)     = H(T2)
        # H(T) plus the cross-phase term enter as flux into the first
        # cell. "v" here is always v1 (= v[p_other] for both phases,
        # since v[p_other] is v2 for phase 0 and v1 for phase 1 --
        # matching the document's fixed "v" convention); sign is -1
        # for phase 0 (PuO) and +1 for phase 1 (combustible). mu_m[0]
        # is this boundary cell's own local mu -- indexed explicitly
        # now that mu_m can vary with z (a bare `mu_m` here would
        # broadcast the whole field against a scalar T difference and
        # silently produce an array where a scalar boundary
        # contribution is expected).
        sign_p  = -1.0 if p == 0 else 1.0
        flux    = H_flux(float(T[p, 0]), T_f,
                          h_conv, emissivity)
        cross_0 = sign_p * ph.K * v[p_other] * mu_m[0] * (T[0, 0] - T[1, 0])
        rhs[0] += dt * (flux - cross_0) / (rCp * dz)

        # z=H: previously purely insulating (Eqs. 67-68, no H(T) term);
        # now optionally allows heat to escape to an ambient exterior,
        # using the same convective+radiative H_flux form as z=0 but
        # with T_amb in place of T_f -- H_flux is positive when its
        # first temperature argument exceeds the surface temperature,
        # so calling it as H_flux(T[-1], T_amb, ...) naturally comes
        # out NEGATIVE (heat leaving) whenever the slab is hotter than
        # ambient, with no extra sign flip needed.
        # The cross-phase coupling term's sign is NOT "-cross_H" like
        # the z=0 case: -K*T' is a flux in the +z direction, which is
        # inflow at z=0 (outward normal -z) but outflow at z=H
        # (outward normal +z), so the relation must be negated once
        # more before it represents inflow here. mu_m[-1] is this
        # boundary cell's own local mu (see mu_m[0] note above).
        cross_H  = sign_p * ph.K * v[p_other] * mu_m[-1] * (T[0, -1] - T[1, -1])
        flux_H   = (H_flux(float(T[p, -1]), T_amb, h_conv_amb, emissivity_amb)
                    if T_amb is not None else 0.0)
        rhs[-1] += dt * (flux_H + cross_H) / (rCp * dz)

        T_new[p] = solve_banded((1, 1), AB, rhs)

    return T_new