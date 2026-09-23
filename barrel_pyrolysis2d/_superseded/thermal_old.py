import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from materials import Material
from mesh import CylindricalMesh2D

Rg          = 8.314
SIGMA_SB    = 5.67e-8   # Stefan-Boltzmann W/m^2/K^4


def arrhenius_rate(mat: Material,
                   T: np.ndarray,
                   omega: np.ndarray) -> np.ndarray:
    """
    Pointwise Arrhenius pyrolysis rate.
    From Eqs. (33)-(34): d(omega)/dt = -k*omega*exp(-Ea/Rg/T)
    """
    exponent = np.where(
        T > 1.0,
        -mat.E_act / (Rg * T),
        -1e10
    )
    return mat.k_arr * omega * np.exp(exponent)


def advance_pyrolysis(mats: list,
                       T: np.ndarray,
                       omega: np.ndarray,
                       dt: float) -> np.ndarray:
    """
    Advance fuel fractions via implicit Euler.
    omega(r,z,0) = omega0 set at initialisation.
    """
    omega_new = omega.copy()
    for j, mat in enumerate(mats):
        if mat.k_arr < 1e-30:
            continue
        exponent  = np.where(T[j] > 1.0,
                             -mat.E_act / (Rg * T[j]),
                             -1e10)
        rate      = mat.k_arr * np.exp(exponent)
        omega_new[j] = omega[j] / (1.0 + dt * rate)
        omega_new[j] = np.clip(omega_new[j], 0.0, mat.omega0)
    return omega_new


def initialise_omega(mats: list,
                      mesh: CylindricalMesh2D) -> np.ndarray:
    """
    Apply IC omega(r,z,0) = omega0 uniformly.
    """
    omega = np.zeros((2, mesh.Nr, mesh.Nz))
    for j, mat in enumerate(mats):
        omega[j] = np.full((mesh.Nr, mesh.Nz), mat.omega0)
    return omega


def fission_source(mat: Material,
                   T: np.ndarray,
                   phi: np.ndarray,
                   Ef: float = 3.2e-11) -> np.ndarray:
    """S_fission = Ef * Sigma_f(T) * phi"""
    return Ef * mat.Sigma_f_T(float(np.mean(T))) * phi


def combustion_source(mat: Material,
                       T: np.ndarray,
                       omega: np.ndarray) -> np.ndarray:
    """
    S_combustion = q * (-d(omega)/dt)
                 = q * k * omega * exp(-Ea/Rg/T)
    """
    return mat.q * arrhenius_rate(mat, T, omega)


def H_flux(T: np.ndarray,
            T_f: float,
            h_conv: float,
            emissivity: float) -> np.ndarray:
    """
    Combined convective + radiative heat flux at z=0.
    From Eq. (66) of document:
        H(T) = h*(T_f - T) + sigma*eps*(T_f^4 - T^4)

    Parameters
    ----------
    T          : surface temperature shape (Nr,)
    T_f        : furnace temperature K
    h_conv     : convective heat transfer coeff W/m^2/K
    emissivity : surface emissivity (0-1)

    Returns
    -------
    flux : shape (Nr,) W/m^2, positive = into barrel
    """
    conv = h_conv * (T_f - T)
    rad  = SIGMA_SB * emissivity * (T_f**4 - T**4)
    return conv + rad


def build_conduction_matrix_2d(mat: Material,
                                 mesh: CylindricalMesh2D,
                                 dt: float) -> csr_matrix:
    """
    Build sparse CN matrix for 2D cylindrical heat
    conduction — interior nodes only.

    Boundary conditions are applied separately in
    solve_thermal_step via the RHS vector.

    All boundaries use Neumann conditions:
        r=0  : dT/dr = 0   (symmetry)
        r=R  : dT/dr = 0   (insulating, Eq. 67)
        z=H  : dT/dz = 0   (insulating, Eq. 67)
        z=0  : applied via RHS (Eqs. 76-77)
    """
    Nr   = mesh.Nr
    Nz   = mesh.Nz
    N    = mesh.N
    dr   = mesh.dr
    dz   = mesh.dz
    r    = mesh.r
    re   = mesh.r_edge
    K    = mat.K
    rCp  = mat.rho * mat.Cp

    A = lil_matrix((N, N))

    for i in range(Nr):
        r_c = r[i]
        r_m = re[i]
        r_p = re[i + 1]

        ar_m = K * r_m / (rCp * r_c * dr**2)
        ar_p = K * r_p / (rCp * r_c * dr**2)
        az   = K / (rCp * dz**2)

        for j in range(Nz):
            n    = mesh.idx(i, j)
            diag = 1.0

            # --- Radial ---
            if i == 0:
                # Symmetry: r_{-1/2}=0, one-sided
                diag += dt * 0.5 * ar_p
                if i + 1 < Nr:
                    A[n, mesh.idx(i+1, j)] -= dt*0.5*ar_p
            elif i == Nr - 1:
                # Insulating outer wall (Eq. 67)
                diag += dt * 0.5 * ar_m
                A[n, mesh.idx(i-1, j)] -= dt*0.5*ar_m
            else:
                diag += dt * 0.5 * (ar_m + ar_p)
                A[n, mesh.idx(i-1, j)] -= dt*0.5*ar_m
                A[n, mesh.idx(i+1, j)] -= dt*0.5*ar_p

            # --- Axial ---
            if j == 0:
                # z=0: one-sided upward only
                # BC flux applied via RHS
                diag += dt * 0.5 * az
                if j + 1 < Nz:
                    A[n, mesh.idx(i, j+1)] -= dt*0.5*az
            elif j == Nz - 1:
                # Insulating top (Eq. 67)
                diag += dt * 0.5 * az
                A[n, mesh.idx(i, j-1)] -= dt*0.5*az
            else:
                diag += dt * az
                A[n, mesh.idx(i, j-1)] -= dt*0.5*az
                A[n, mesh.idx(i, j+1)] -= dt*0.5*az

            A[n, n] = diag

    return csr_matrix(A)


def apply_thermal_bc(mat: Material,
                      mesh: CylindricalMesh2D,
                      T_j: np.ndarray,
                      T_k: np.ndarray,
                      v_j: float,
                      mu: float,
                      dt: float,
                      T_f: float,
                      h_conv: float,
                      emissivity: float,
                      phase_idx: int) -> np.ndarray:
    """
    Build RHS contribution from boundary conditions.

    From Eqs. (72)-(77) of document:

    All boundaries except z=0 (Eqs. 72-73):
        -K1*T1' - K1*(1-v)*mu*(T1-T2) = 0
        -K2*T2' + K2*v*mu*(T1-T2)     = 0

    At z=0 (Eqs. 76-77):
        -K1*T1' - K1*(1-v)*mu*(T1-T2) = H(T1)
        -K2*T2' + K2*v*mu*(T1-T2)     = H(T2)

    The stochastic coupling terms at boundaries are
    absorbed into the interior coupling in the main
    matrix. Here we add the H(T) flux at z=0.

    Parameters
    ----------
    mat        : material for this phase
    T_j        : temperature of this phase (Nr,Nz)
    T_k        : temperature of other phase (Nr,Nz)
    v_j        : volume fraction of this phase
    mu         : isotropic inverse correlation length
    dt         : timestep
    T_f        : furnace temperature K
    h_conv     : convective HTC W/m^2/K
    emissivity : surface emissivity
    phase_idx  : 0 for phase 1, 1 for phase 2

    Returns
    -------
    b_bc : RHS contribution shape (N,)
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    N      = mesh.N
    dz     = mesh.dz
    rCp    = mat.rho * mat.Cp
    b_bc   = np.zeros(N)

    # Surface temperature at z=0 for this phase
    T_surf = T_j[:, 0]   # shape (Nr,)

    # H(T_j) from Eq. (74)-(75)
    flux = H_flux(T_surf, T_f, h_conv, emissivity)
    # flux > 0 means heat into barrel

    # Add to RHS for j=0 nodes (z=0 boundary)
    for i in range(Nr):
        n = mesh.idx(i, 0)
        # +flux/(rCp*dz)*dt enters RHS
        b_bc[n] += dt * flux[i] / (rCp * dz)

    return b_bc


def solve_thermal_step(mats: list,
                        mesh: CylindricalMesh2D,
                        T: np.ndarray,
                        omega: np.ndarray,
                        phi: np.ndarray,
                        v: np.ndarray,
                        mu: float,
                        dt: float,
                        T_f: float,
                        h_conv: float,
                        emissivity: float,
                        Ef: float = 3.2e-11) -> np.ndarray:
    """
    Advance temperature for both phases by dt.

    Implements Eqs. (29)-(30) with BCs (72)-(77).

    Parameters
    ----------
    T          : temperatures (2, Nr, Nz)
    omega      : fuel fractions (2, Nr, Nz)
    phi        : fluxes (2, Nr, Nz)
    v          : volume fractions (2,)
    mu         : isotropic inverse correlation length cm^-1
    T_f        : furnace temperature K
    h_conv     : convective heat transfer coeff W/m^2/K
    emissivity : barrel surface emissivity
    """
    from stochastics import (coupling_coefficient_r,
                              coupling_coefficient_z)

    Nr, Nz = mesh.Nr, mesh.Nz
    T_new  = T.copy()

    # With mu_r = mu_z = mu (Eq. 65, document note)
    G = coupling_coefficient_r(v[0], v[1], mu)

    for j_ph, mat in enumerate(mats):
        k_other = 1 - j_ph

        A    = build_conduction_matrix_2d(mat, mesh, dt)

        # --- Interior sources ---
        S_f  = fission_source(mat, T[j_ph], phi[j_ph], Ef)
        S_c  = combustion_source(mat, T[j_ph], omega[j_ph])

        # Stochastic inter-phase coupling (interior)
        dT = T[j_ph] - T[k_other]
        if j_ph == 0:
            # Phase 1: +v2*G*K1*(T1-T2) from Eqs.(31),(72)
            coupling = v[1] * G * mat.K * dT
        else:
            # Phase 2: -v1*G*K2*(T1-T2) from Eqs.(32),(73)
            coupling = -v[0] * G * mats[k_other].K * dT

        S_total = (S_f + S_c + coupling).ravel()

        rhs = T[j_ph].ravel() + dt/(mat.rho*mat.Cp)*S_total

        # --- Boundary condition at z=0 (Eqs. 76-77) ---
        b_bc = apply_thermal_bc(
            mat, mesh,
            T[j_ph], T[k_other],
            v[j_ph], mu, dt,
            T_f, h_conv, emissivity,
            j_ph
        )
        rhs += b_bc

        T_new[j_ph] = spsolve(A, rhs).reshape(Nr, Nz)

    return T_new