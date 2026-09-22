"""
thermal.py  (2D axisymmetric r-z)
==================================
Two-phase stochastic heat conduction on a cylindrical (r,z) mesh,
built from the component-wise dichotomic-Markov closure.

Closure
-------
From <F dF/dx_i> = v_1 v_2 mu_i (Williams, Appendix B, backward-
difference convention of eqn B1), the phase currents are

    J_{1,i} = -K_1 dT_1/dx_i - K_1 v_2 mu_i (T_1 - T_2)
    J_{2,i} = -K_2 dT_2/dx_i + K_2 v_1 mu_i (T_1 - T_2)

with mu_i = 1/l_{c,i} the inverse correlation length along direction
i. Williams B9/B10 give a separate correlation length per direction,
so mu_r and mu_z are independent in general; mu_r = mu_z is the
isotropic-chunk special case, not a requirement of the formalism.

SIGN NOTE. Williams' eqn (69) carries the opposite sign on the
coupling term of J_1. Substituting the ansatz into J = -K grad T,
multiplying by F and averaging gives
    J_1 = -K_1 grad T_1 - (K_1/v_1)(T_1-T_2) <F grad F>
and his own Appendix B result <FF'> = v_1 v_2 mu then yields the sign
used here. His (69) appears to drop the minus from that division; the
form implemented here is the self-consistent one and matches the
document's Section 5 slab equation.

Governing equation
------------------
Averaging the energy equation against F and against (1-F) gives, for
phase i with k the other phase and m = (mu_r, mu_z). Because
sigma_i (T_1 - T_2) collapses to (T_i - T_k), BOTH phases share one
form:

  rho_i Cp_i dT_i/dt =
        K_i lap T_i
      + v_k K_i div[(T_i - T_k) m]              <- conservative
      + v_k K_i (m . grad) T_i                  <- self drift
      - v_k K_k (m . grad) T_k                  <- cross drift
      + v_k (K_i v_k + K_k v_i) |m|^2 (T_i-T_k) <- algebraic
      + S_i

With mu_r = 0, mu_z = mu constant and Nr = 1 this reduces EXACTLY to
the document's active slab equation

    K_i T_i'' + 2 v_k K_i mu T_i' - v_k mu (K_i + K_k) T_k'
              + v_k mu^2 (K_i v_k + K_k v_i)(T_i - T_k)

which is what reduction_test.py checks, against a reference built
independently from the written equation.

|m|^2 = mu_r^2 + mu_z^2, so at equal mu the algebraic coupling is
strictly STRONGER in 2D than in the slab -- it is not the slab
coefficient re-used.

The axis
--------
The conservative term is discretised in finite-volume form, so the
r = 0 face carries area 2*pi*r*dz = 0 and contributes nothing to the
cell balance whatever the flux there. The 1/r of the cylindrical
divergence is therefore never evaluated and the axis needs no
regularisation -- unlike a non-conservative (1/r) d(r .)/dr form,
which does.

Boundary conditions
-------------------
    r = 0   : symmetry (zero-area face, see above)
    r = R   : Dirichlet, T = T_amb (cold sidewall)
    z = 0   : heated, convective + radiative flux from furnace T_f
    z = H   : convective + radiative loss to T_amb
    At z = 0 and z = H the flux supplied is the TOTAL phase current
    (diffusive + stochastic), as in the document's slab BCs, so no
    separate stochastic face flux is assembled on those faces.

Units: mesh in cm, K in W/m/K, mu in cm^-1 (converted internally).
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from mesh import CylindricalMesh2D
from materials import Phase

CM_TO_M = 0.01
SIGMA_SB = 5.670374419e-8    # Stefan-Boltzmann, W/m^2/K^4
R_GAS = 8.314                # J/mol/K


# ─────────────────────────────────────────────────────────────────────
# Pyrolysis / sources (elementwise, dimension-agnostic)
# ─────────────────────────────────────────────────────────────────────
def arrhenius_rate(ph: Phase, T: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """d(omega)/dt = -k * omega * exp(-Ea/(Rg T)), kg/m^3/s."""
    if ph.k_arr <= 0.0:
        return np.zeros_like(omega)
    Tc = np.maximum(T, 1.0)
    return -ph.k_arr * omega * np.exp(-ph.E_act / (R_GAS * Tc))


def advance_pyrolysis(phases: list, T: np.ndarray, omega: np.ndarray,
                       dt: float) -> np.ndarray:
    out = omega.copy()
    for p, ph in enumerate(phases):
        out[p] = np.maximum(
            omega[p] + dt * arrhenius_rate(ph, T[p], omega[p]), 0.0)
    return out


def initialise_omega(phases: list, Nr: int, Nz: int) -> np.ndarray:
    return np.array([np.full((Nr, Nz), ph.omega0) for ph in phases])


def fission_source(ph: Phase, T: np.ndarray, phi: np.ndarray,
                    Ef: float, T_n: float = None) -> np.ndarray:
    """Fission heating, W/m^3: 1e6 * Ef * sum_g Sigma_f,g(T, T_n) phi_g,
    with Sigma_f in cm^-1 and phi in n/cm^2/s (1e6: cm^-3 -> m^-3).
    Thermal-group 1/v at the neutron temperature T_n (see materials.py).
    phi shape (G, Nr, Nz)."""
    Sf = ph.Sf_T(float(np.mean(T)), T_n)
    return 1.0e6 * Ef * sum(Sf[g] * phi[g] for g in range(len(Sf)))


def combustion_source(ph: Phase, T: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """-q * d(omega)/dt, W/m^3 (positive for exothermic decomposition)."""
    return -ph.q * arrhenius_rate(ph, T, omega)


# ─────────────────────────────────────────────────────────────────────
# Operator assembly
# ─────────────────────────────────────────────────────────────────────
def _as_field(x, shape) -> np.ndarray:
    return np.broadcast_to(np.atleast_1d(np.asarray(x, dtype=float)),
                           shape).astype(float)


def build_self_operator(ph: Phase,
                         mesh: CylindricalMesh2D,
                         dt: float,
                         mu_r, mu_z,
                         v_self: float,
                         v_oth: float,
                         K_oth: float,
                         T_amb,
                         closure: str = "relaxational") -> csr_matrix:
    """
    Backward-Euler matrix for every term acting on phase i's OWN
    temperature:
        K_i lap T_i + v_k K_i div(T_i m) + v_k K_i (m.grad) T_i
                    + v_k (K_i v_k + K_k v_i)|m|^2 T_i
    Cross-phase terms (everything in T_k) are explicit, in the RHS.
    Returns (I - dt/(rho Cp) L_self).
    """
    Nr, Nz, N = mesh.Nr, mesh.Nz, mesh.N
    dr = mesh.dr * CM_TO_M
    dz = mesh.dz * CM_TO_M
    r = mesh.r * CM_TO_M
    re = mesh.r_edge * CM_TO_M
    K = ph.K
    rCp = ph.rho * ph.Cp

    mr = _as_field(mu_r, (Nr, Nz)) / CM_TO_M     # cm^-1 -> m^-1
    mz = _as_field(mu_z, (Nr, Nz)) / CM_TO_M
    m2 = mr**2 + mz**2

    a = dt / rCp
    A = lil_matrix((N, N))

    for i in range(Nr):
        r_c, r_m, r_p = r[i], re[i], re[i + 1]
        for j in range(Nz):
            n = mesh.idx(i, j)

            # Dirichlet sidewall: identity row, value supplied in RHS
            if T_amb is not None and i == Nr - 1 and Nr > 1:
                A[n, n] = 1.0
                continue

            diag = 1.0

            # ---- Radial diffusion (FV; axis face has area 0) ----
            if Nr > 1:
                cm_ = a * K * r_m / (r_c * dr**2)
                cp_ = a * K * r_p / (r_c * dr**2)
                if i > 0:
                    diag += cm_
                    A[n, mesh.idx(i - 1, j)] -= cm_
                if i < Nr - 1:
                    diag += cp_
                    A[n, mesh.idx(i + 1, j)] -= cp_

            # ---- Axial diffusion ----
            cz = a * K / dz**2
            if j > 0:
                diag += cz
                A[n, mesh.idx(i, j - 1)] -= cz
            if j < Nz - 1:
                diag += cz
                A[n, mesh.idx(i, j + 1)] -= cz

            # ---- Conservative  v_k K_i div(T_i m) ----
            # UPWIND DIRECTION. These drift coefficients are POSITIVE,
            # so the term enters as dT/dt = +c dT/dx with c > 0, i.e. a
            # transport equation of characteristic speed -c:
            # information travels toward DECREASING x. The upwind
            # stencil is therefore FORWARD (value taken from the
            # high-side cell), not backward. Using a backward stencil
            # here is downwinding and is unconditionally unstable.
            # The axis face (r_m = 0) still drops out identically.
            # closure="relaxational": no odd-order terms at all (pure
            # conduction here); its coupling is the relaxational exchange
            # in solve_thermal_step. See barrel_pyrolysis1d/neutronics.py.
            cc = a * v_oth * K if closure == "document" else 0.0
            if Nr > 1:
                diag += cc * r_m * mr[i, j] / (r_c * dr)
                if i < Nr - 1:
                    A[n, mesh.idx(i + 1, j)] -= cc * r_p * mr[i + 1, j] / (r_c * dr)
            # -G_{j-1/2} face: interior only. The z=0 face carries no
            # stochastic flux here because the document's BC fixes the
            # TOTAL current there (diffusive + stochastic) = H(T),
            # supplied in full by solve_thermal_step.
            if j > 0:
                diag += cc * mz[i, j] / dz
            if j < Nz - 1:
                A[n, mesh.idx(i, j + 1)] -= cc * mz[i, j + 1] / dz

            # ---- Drift  v_k K_i (m . grad) T_i  (forward upwind) ----
            if closure == "document" and Nr > 1 and i < Nr - 1:
                sr = a * v_oth * K * mr[i, j] / dr
                diag += sr
                A[n, mesh.idx(i + 1, j)] -= sr
            if closure == "document" and j < Nz - 1:
                sz = a * v_oth * K * mz[i, j] / dz
                diag += sz
                A[n, mesh.idx(i, j + 1)] -= sz

            # ---- Algebraic coupling is NOT assembled here ----
            # v_k (K_i v_k + K_k v_i)|m|^2 (T_i - T_k) has a POSITIVE
            # coefficient on (T_i - T_k): the hotter phase gains, so
            # the term is anti-diffusive between phases. Putting its
            # T_i half on the implicit diagonal would subtract from the
            # diagonal and can destroy the M-matrix property (and with
            # it solvability) at large mu. The whole algebraic term --
            # both halves -- is therefore explicit, in solve_thermal_
            # step's S_cross. See the module docstring.

            A[n, n] = diag

    return csr_matrix(A)


def grad_upwind(T: np.ndarray, dr: float, dz: float) -> tuple:
    """Forward-difference gradients matching the implicit upwinding
    (see the upwind-direction note in build_self_operator); one-sided
    (zero) at the high-index boundaries."""
    dTdr = np.zeros_like(T)
    dTdz = np.zeros_like(T)
    dTdr[:-1, :] = (T[1:, :] - T[:-1, :]) / dr
    dTdz[:, :-1] = (T[:, 1:] - T[:, :-1]) / dz
    return dTdr, dTdz


def div_T_m(T: np.ndarray, mr: np.ndarray, mz: np.ndarray,
            mesh: CylindricalMesh2D, dr: float, dz: float) -> np.ndarray:
    """Finite-volume div(T m) with upwind face values, for the explicit
    cross-phase copy of the conservative term."""
    Nr, Nz = T.shape
    r = mesh.r * CM_TO_M
    re = mesh.r_edge * CM_TO_M
    out = np.zeros_like(T)

    if Nr > 1:
        Fr = np.zeros((Nr + 1, Nz))
        Fr[1:Nr, :] = T[1:, :] * mr[1:, :]          # forward upwind
        out = out + (re[1:, None] * Fr[1:, :]
                     - re[:-1, None] * Fr[:-1, :]) / (r[:, None] * dr)

    Fz = np.zeros((Nr, Nz + 1))
    Fz[:, 1:Nz] = T[:, 1:] * mz[:, 1:]              # forward upwind
    # Boundary faces (z=0, z=H) carry no stochastic flux: the document's
    # boundary conditions (Eqs. 71-72; Williams 83-99) fix the TOTAL
    # phase current there, supplied in full from H(T). This must match
    # build_self_operator exactly, or the two halves of
    # div[(T_i - T_k) m] fail to cancel in equilibrium.
    Fz[:, 0] = 0.0
    out = out + (Fz[:, 1:] - Fz[:, :-1]) / dz
    return out


def solve_thermal_step(phases: list,
                        mesh: CylindricalMesh2D,
                        T: np.ndarray,
                        omega: np.ndarray,
                        phi: np.ndarray,
                        v: np.ndarray,
                        mu_r, mu_z,
                        dt: float,
                        T_f: float,
                        h_conv: float,
                        emissivity: float,
                        Ef: float = 3.2e-11,
                        T_amb: float = 300.0,
                        h_conv_amb: float = None,
                        emissivity_amb: float = None,
                        S_extra: np.ndarray = None,
                        closure: str = "relaxational") -> np.ndarray:
    """
    Advance both phases' temperatures by dt.

    T          : (2, Nr, Nz)
    omega      : (2, Nr, Nz)
    phi        : (2, G, Nr, Nz)
    v          : (2,)
    mu_r, mu_z : scalars or (Nr, Nz) fields, cm^-1
    T_amb      : sidewall Dirichlet value and z=H sink; None recovers an
                 insulating sidewall and top (used by the Nr=1 slab
                 reduction test).
    S_extra    : optional explicit volumetric source, W/m^3, (2,Nr,Nz)
                 e.g. -percolation.thermal_sink(...)
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    dr = mesh.dr * CM_TO_M
    dz = mesh.dz * CM_TO_M
    if h_conv_amb is None:
        h_conv_amb = h_conv
    if emissivity_amb is None:
        emissivity_amb = emissivity

    mr = _as_field(mu_r, (Nr, Nz)) / CM_TO_M
    mz = _as_field(mu_z, (Nr, Nz)) / CM_TO_M
    m2 = mr**2 + mz**2

    T_new = T.copy()
    wall = (T_amb is not None and Nr > 1)

    for p, ph in enumerate(phases):
        k = 1 - p
        ph_oth = phases[k]
        rCp = ph.rho * ph.Cp
        v_self, v_oth = v[p], v[k]

        A = build_self_operator(ph, mesh, dt, mu_r, mu_z,
                                 v_self, v_oth, ph_oth.K, T_amb, closure)

        # ---- explicit cross-phase terms (all in T_k) ----
        coef = v_oth * (ph.K * v_oth + ph_oth.K * v_self) * m2
        dTk_dr, dTk_dz = grad_upwind(T[k], dr, dz)
        if closure == "relaxational":
            # relaxational exchange: phase temperatures equilibrate
            S_cross = -coef * (T[p] - T[k])
        else:
            S_cross = (
                -v_oth * ph.K * div_T_m(T[k], mr, mz, mesh, dr, dz)
                - v_oth * ph_oth.K * (mr * dTk_dr + mz * dTk_dz)
                # full algebraic coupling, both halves (see build_self_operator)
                + coef * (T[p] - T[k])
            )

        m_idx = int(np.argmax([q_.Sigma_s12 for q_ in phases]))
        T_n = float(np.mean(T[m_idx]))
        S = S_cross + fission_source(ph, T[p], phi[p], Ef, T_n) \
                    + combustion_source(ph, T[p], omega[p])
        if S_extra is not None:
            S = S + S_extra[p]

        rhs = (T[p] + dt / rCp * S).ravel()

        # ---- z = 0 heated face ----
        Tsurf = T[p][:, 0]
        q0 = (h_conv * (T_f - Tsurf)
              + emissivity * SIGMA_SB * (T_f**4 - Tsurf**4))
        for i in range(Nr):
            if wall and i == Nr - 1:
                continue
            rhs[mesh.idx(i, 0)] += dt / rCp * q0[i] / dz

        # ---- z = H loss to ambient ----
        if T_amb is not None:
            Ttop = T[p][:, -1]
            qH = (h_conv_amb * (T_amb - Ttop)
                  + emissivity_amb * SIGMA_SB * (T_amb**4 - Ttop**4))
            for i in range(Nr):
                if wall and i == Nr - 1:
                    continue
                rhs[mesh.idx(i, Nz - 1)] += dt / rCp * qH[i] / dz

        # ---- r = R Dirichlet ----
        if wall:
            for j in range(Nz):
                rhs[mesh.idx(Nr - 1, j)] = T_amb

        T_new[p] = spsolve(A, rhs).reshape(Nr, Nz)

    return T_new
