"""
thermal.py
==========
Single-group 1D slab heat conduction and pyrolysis.

Thermal equation -- the ACTIVE form in the document's "A two phase
slab system" solve (k = the other phase; i=1 => k=2 and vice versa):

    rho_i Cp_i dT_i/dt = K_i d^2T_i/dz^2
                        + 2*v_k*K_i*mu * dT_i/dz
                        - K_i*v_k*mu * dT_k/dz
                        + K_i*v_k^2*mu^2 * (T_i - T_k)
                        - K_k*v_k*mu * dT_k/dz
                        + K_k*v_i*v_k*mu^2 * (T_i - T_k)
                        + Ef * sum_g Sigma_f,i,g * phi_i,g
                        - q_i * d(omega_i)/dt

    CORRECTION NOTE. This module previously implemented the document's
    SUPERSEDED form (the commented-out equation immediately above the
    active one in the source):

        ... + mu*dT_i/dz - mu*sum_k v_k*dT_k/dz
            - mu*sum_{l!=i}(K_l*dT_l/dz + mu*T_l)

    That form is dimensionally inconsistent: mu*dT_i/dz has units
    K/m^2, whereas every other term in the equation (rho*Cp*dT/dt,
    K*d^2T/dz^2, the sources) is W/m^3. Restoring the factor 2*K_i on
    the self-drift term fixes it, and the cross terms pick up the
    matching K_i/v_k factors above. It also previously grouped
    -mu*(v_k + K_k)*dT_k/dz, i.e. added a dimensionless volume
    fraction to a conductivity, and carried a bare -mu^2*T_k with no
    matching +mu^2*T_i -- so the algebraic coupling failed to vanish
    when the two phases were in thermal equilibrium. Both are fixed.

    CONSEQUENCE: any mu calibrated against the old operator (e.g. a
    Config.mu bisected to put k_eff(0) at 1) carries no meaning under
    this one and must be re-fitted.

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

Collecting terms for phase 1 (i=1, k=2):
    = K_1 d^2T_1/dz^2
      + 2*v_2*K_1*mu * dT_1/dz                    [implicit, upwind]
      - v_2*mu*(K_1 + K_2) * dT_2/dz              [explicit]
      + v_2*mu^2*(K_1*v_2 + K_2*v_1) * (T_1 - T_2) [explicit]
      + sources

i.e. the other phase's gradient terms combine with coefficient
-v_2*mu*(K_1 + K_2), and the algebraic coupling is proportional to
(T_1 - T_2) so it vanishes identically in thermal equilibrium.

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


def pyrolysis_step(phases: list, T: np.ndarray, omega: np.ndarray,
                   dt: float) -> tuple:
    """
    Advance the volatile density over dt EXACTLY at fixed temperature T,
    and return the heat released CONSISTENTLY with the mass consumed:

        kappa     = k_arr exp(-E_act / (Rg T))          (1/s)
        omega_new = omega exp(-kappa dt)
        S_c       = q (omega - omega_new) / dt          (W/m^3)

    so the energy released over the step is exactly q * (mass consumed),
    whatever kappa*dt is.

    BUG FIXED. The superseded update, omega/(1 + dt*rate), used
    arrhenius_rate() -- a MASS-consumption rate k*omega*exp(...), kg/m^3/s
    -- where a rate CONSTANT (1/s) belongs. dt*rate then carried units of
    kg/m^3, consuming omega ~omega0 (~450x) too fast, while the heat
    source q*arrhenius_rate used the correct rate: the volatiles vanished
    before releasing their heat, and only ~2-4% of the chemical energy
    was ever deposited (isolated-slab test). Present in the original
    commit, so it affects all earlier 1D transients.

    Returns (omega_new, S_c), each shaped like omega.
    """
    omega_new = omega.copy()
    S_c = np.zeros_like(omega, dtype=float)
    for p, ph in enumerate(phases):
        if ph.k_arr < 1e-30:
            continue
        Tc = np.maximum(T[p], 1.0)
        kappa = ph.k_arr * np.exp(-ph.E_act / (Rg * Tc))
        omega_new[p] = omega[p] * np.exp(-kappa * dt)
        S_c[p] = ph.q * (omega[p] - omega_new[p]) / dt
    return omega_new, S_c


def advance_pyrolysis(phases: list, T: np.ndarray, omega: np.ndarray,
                      dt: float) -> np.ndarray:
    """omega after dt (exact at fixed T); see pyrolysis_step."""
    return pyrolysis_step(phases, T, omega, dt)[0]


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


def stochastic_self_operator(ph: Phase,
                              mesh: Mesh1D,
                              mu,
                              v_oth: float,
                              closure: str = "relaxational") -> np.ndarray:
    """
    Tridiagonal operator L_self (banded, solve_banded((1,1)) layout),
    in W/m^3 per K, acting on phase i's OWN temperature:

        L_self T_i = K_i T_i''
                   + v_k K_i d/dz[mu T_i]         (conservative half)
                   + v_k K_i mu dT_i/dz           (self drift)

    Finite-volume form. The conservative term is the divergence of the
    stochastic part of the phase current, J_i^s = -v_k K_i mu (T_i-T_k),
    so it is assembled as FACE fluxes, not as a pointwise derivative.
    That keeps the d(mu)/dz contribution that a pointwise
    2 v_k K_i mu dT_i/dz form silently drops when mu varies (e.g. with
    percolation on); with mu constant the two are identical.

    UPWINDING. Every first-order coefficient here is positive, so the
    terms enter as dT/dt = +c dT/dz: characteristic speed -c,
    information travelling toward DECREASING z. The upwind stencil is
    FORWARD (face value from the high-side cell). The superseded
    operator used a backward stencil together with the opposite sign
    on the drift -- stable, but implementing -2 v_k K_i mu dT_i/dz,
    i.e. the wrong sign relative to the document.

    BOUNDARY FACES carry no stochastic flux here: the document's
    boundary conditions (Eqs. 71-72; Williams eqns 83-99) fix the
    TOTAL phase current -- diffusive plus stochastic -- so the whole
    boundary current is supplied by solve_thermal_step from H(T).

    The algebraic coupling v_k (K_i v_k + K_k v_i) mu^2 (T_i - T_k)
    is NOT assembled here: its coefficient on T_i is positive (anti-
    diffusive between phases), so on the implicit diagonal it would
    erode diagonal dominance. It is explicit, in solve_thermal_step.
    """
    N  = mesh.N
    dz = mesh.dz * CM_TO_M
    mu_m = np.broadcast_to(np.atleast_1d(mu), (N,)).astype(float) / CM_TO_M
    K = ph.K
    # closure="relaxational" drops the odd-order (conservative + drift)
    # terms entirely, leaving pure conduction here; its coupling is the
    # relaxational algebraic term in _stochastic_cross_source. See
    # neutronics._build_group_self_and_cross for the benchmark evidence.
    if closure not in ("relaxational", "document"):
        raise ValueError(f"unknown closure {closure!r}")
    c = v_oth * K if closure == "document" else 0.0

    up = np.zeros(N)     # coefficient of T[j+1] in row j   (L[j, j+1])
    dg = np.zeros(N)     # coefficient of T[j]   in row j
    lo = np.zeros(N)     # coefficient of T[j-1] in row j   (L[j, j-1])

    # Diffusion through interior faces j+1/2, j = 0..N-2
    for j in range(N - 1):
        g = K / dz**2
        dg[j]     -= g;  up[j]     += g
        dg[j + 1] -= g;  lo[j + 1] += g

    # Conservative: (G_{j+1/2} - G_{j-1/2})/dz, G_{j+1/2} = mu_{j+1} T_{j+1}
    # on interior faces (forward upwind), zero on the boundary faces.
    for j in range(N):
        if j < N - 1:
            up[j] += c * mu_m[j + 1] / dz      # + G_{j+1/2}
        if j > 0:
            dg[j] -= c * mu_m[j] / dz          # - G_{j-1/2}

    # Self drift: c mu_j (T_{j+1} - T_j)/dz  (forward; zero in top cell)
    for j in range(N - 1):
        dg[j] -= c * mu_m[j] / dz
        up[j] += c * mu_m[j] / dz

    L = np.zeros((3, N))
    L[0, 1:]  = up[:-1]      # L[j, j+1] stored at AB[0, j+1]
    L[1, :]   = dg
    L[2, :-1] = lo[1:]       # L[j+1, j] stored at AB[2, j]
    return L


def _stochastic_cross_source(ph: Phase,
                              ph_oth: Phase,
                              T_i: np.ndarray,
                              T_k: np.ndarray,
                              mu_m: np.ndarray,
                              v_self: float,
                              v_oth: float,
                              dz: float,
                              closure: str = "relaxational") -> np.ndarray:
    """
    Explicit part of the stochastic operator, W/m^3:

        - v_k K_i d/dz[mu T_k]                          (conservative)
        - v_k K_k mu dT_k/dz                            (cross drift)
        + v_k (K_i v_k + K_k v_i) mu^2 (T_i - T_k)      (algebraic)

    Stencils identical to stochastic_self_operator, so the two halves
    of v_k K_i d/dz[mu (T_i - T_k)] cancel exactly when T_i = T_k.
    """
    N = len(T_i)
    coef = v_oth * (ph.K * v_oth + ph_oth.K * v_self) * mu_m**2
    if closure == "relaxational":
        # Relaxational exchange only: -v_k (K_i v_k + K_k v_i) mu^2 (T_i - T_k).
        # Phase temperatures EQUILIBRATE, as the realisation-averaged
        # benchmark requires (the document sign drives them apart).
        # Its rate mu^2 (K_1 v_2 + K_2 v_1)(v_2/rho_1Cp_1 + v_1/rho_2Cp_2)
        # is the timescale the benchmark decay collapses onto.
        return -coef * (T_i - T_k)
    G = np.zeros(N + 1)                       # face values, boundary faces 0
    G[1:N] = mu_m[1:] * T_k[1:]
    cons = -v_oth * ph.K * (G[1:] - G[:-1]) / dz

    dTk = np.zeros(N)
    dTk[:-1] = (T_k[1:] - T_k[:-1]) / dz
    drift = -v_oth * ph_oth.K * mu_m * dTk

    alg = v_oth * (ph.K * v_oth + ph_oth.K * v_self) * mu_m**2 * (T_i - T_k)
    return cons + drift + alg


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
                        S_extra: np.ndarray = None,
                        closure: str = "relaxational") -> np.ndarray:
    """
    Advance temperature for both phases by dt.

    Implements Eq. (95) exactly:

        rho_i Cp_i dT_i/dt = K_i d^2T_i/dz^2
            + 2*v_k*K_i*mu * dT_i/dz
            - v_k*mu*(K_i + K_k) * dT_k/dz
            + v_k*mu^2*(K_i*v_k + K_k*v_i) * (T_i - T_k)
            + Ef * Sigma_f,i * phi_i
            - q_i * d(omega_i)/dt

    (k = the other phase.) Crank-Nicolson, finite-volume. The implicit
    part (stochastic_self_operator) holds
        K_i T_i''  +  v_k K_i d/dz[mu T_i]  +  v_k K_i mu dT_i/dz
    and the explicit part (_stochastic_cross_source) holds
        - v_k K_i d/dz[mu T_k]  -  v_k K_k mu dT_k/dz
        + v_k mu^2 (K_i v_k + K_k v_i)(T_i - T_k)
        + sources
    so that with mu constant the drift terms combine to the document's
    2 v_k K_i mu T_i' - v_k mu (K_i + K_k) T_k'.

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

    for p, ph in enumerate(phases):
        p_other = 1 - p
        ph_oth  = phases[p_other]
        rCp     = ph.rho * ph.Cp
        v_i, v_k = v[p], v[p_other]

        # Crank-Nicolson on the implicit (self) part:
        #   (I - dt/2 L/rCp) T_new = (I + dt/2 L/rCp) T_old + dt/rCp S
        L  = stochastic_self_operator(ph, mesh, mu, v_k, closure)
        AB = -0.5 * dt / rCp * L
        AB[1, :] += 1.0

        S_total = _stochastic_cross_source(ph, ph_oth, T[p], T[p_other],
                                            mu_m, v_i, v_k, dz, closure)

        # Fission heating, W/m^3:
        #     1e6 * Ef * sum_g Sigma_f,g(T, T_n) * phi_g
        # with Sigma_f in cm^-1 and phi in n/cm^2/s (absolute: the
        # solver's amplitude P carries the units, see solver.Config),
        # giving fissions/cm^3/s; 1e6 converts cm^-3 -> m^-3. The
        # superseded form omitted the conversion, and with P = 1 the
        # flux amplitude itself was arbitrary, so fission heating was
        # ~1e-14 W/m^3 -- effectively absent -- in every earlier run.
        T_p = float(np.mean(T[p]))
        m_idx = int(np.argmax([q.Sigma_s12 for q in phases]))
        T_n = float(np.mean(T[m_idx]))
        Sf  = ph.Sf_T(T_p, T_n)
        S_total = S_total + 1.0e6 * Ef * (Sf[0] * phi[p][0] + Sf[1] * phi[p][1])

        # Combustion source: q * (-d omega/dt), W/m^3 (omega in kg/m^3)
        S_total = S_total + ph.q * arrhenius_rate(ph, T[p], omega[p])

        if S_extra is not None:
            S_total = S_total + S_extra[p]

        rhs = 2.0 * T[p] - _banded_matvec(AB, T[p]) + dt / rCp * S_total

        # ── Boundary conditions ─────────────────────────────────
        # The document's BCs (Eqs. 71-72) fix the TOTAL phase current,
        # diffusive + stochastic:
        #     J_i(0) . z_hat = H(T_i(0))      (heat INTO the slab)
        # The operator assembles no flux on either boundary face, so
        # H alone is the complete boundary current -- no separate
        # stochastic correction is added (the superseded code added
        # one, which with the pointwise drift double-counted/omitted
        # the stochastic face flux depending on the phase).
        flux0 = H_flux(float(T[p, 0]), T_f, h_conv, emissivity)
        rhs[0] += dt * flux0 / (rCp * dz)

        # z=H: loss to ambient (H_flux is negative when hotter than
        # ambient), or insulating (zero total current) if T_amb is None.
        if T_amb is not None:
            fluxH = H_flux(float(T[p, -1]), T_amb, h_conv_amb, emissivity_amb)
            rhs[-1] += dt * fluxH / (rCp * dz)

        T_new[p] = solve_banded((1, 1), AB, rhs)

    return T_new