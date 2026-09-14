import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigs as sp_eigs
from materials import Material, BETA, DELAYED_GROUPS
from mesh import CylindricalMesh2D
from stochastics import (coupling_coefficient_r,
                          coupling_coefficient_z)


def build_static_system(mats: list,
                          mesh: CylindricalMesh2D,
                          T: np.ndarray,
                          v: np.ndarray,
                          mu: float) -> tuple:
    """
    Build loss matrix L and fission matrix F.

    Vacuum BCs from Eqs. (83)-(84) of document:
        psi1 + d_ext1*[psi1' + (1-v)*mu*(psi1-psi2)] = 0
        psi2 + d_ext2*[psi2' - v*mu*(psi1-psi2)]     = 0

    where d_ext,j = 1/Sigma_a,j  (document Eq. 84)

    Symmetry BC at r=0:
        d(psi)/dr = 0
    """
    Nr   = mesh.Nr
    Nz   = mesh.Nz
    N    = mesh.N
    dr   = mesh.dr
    dz   = mesh.dz
    r    = mesh.r
    re   = mesh.r_edge
    size = 2 * N

    L = lil_matrix((size, size))
    F = lil_matrix((size, size))

    # mu_r = mu_z = mu as stated in document
    G = coupling_coefficient_r(v[0], v[1], mu)

    for j_ph, mat in enumerate(mats):
        offset  = j_ph * N
        T_j     = float(np.mean(T[j_ph]))
        D_j     = mat.D_T(T_j)
        Sa_j    = mat.Sigma_a_T(T_j)
        Sf_j    = mat.Sigma_f_T(T_j)

        # Phase-specific extrapolation distance
        # From document Eq. (84): d_ext,j = 1/Sigma_a,j
        d_ext = 1.0 / Sa_j if Sa_j > 1e-10 else 10.0 * dz

        for i in range(Nr):
            r_c = r[i]
            r_m = re[i]
            r_p = re[i + 1]

            ar_m = D_j * r_m / (r_c * dr**2)
            ar_p = D_j * r_p / (r_c * dr**2)
            az   = D_j / dz**2

            for j in range(Nz):
                row     = offset + mesh.idx(i, j)
                row_oth = (1 - j_ph)*N + mesh.idx(i, j)
                diag    = Sa_j

                # --- Radial diffusion ---
                if i == 0:
                    # Symmetry: d(psi)/dr = 0 at r=0
                    # One-sided: only r_p term
                    diag += ar_p
                    if i + 1 < Nr:
                        L[row, offset+mesh.idx(i+1,j)] -= ar_p

                elif i == Nr - 1:
                    # Vacuum BC at r=R (Eq. 83 or 84)
                    # psi_j + d_ext_j*[psi_j' +/- coupling] = 0
                    # Coupling term from stochastic BC:
                    #   phase 1: +(1-v)*mu*(psi1-psi2)
                    #   phase 2: -v*mu*(psi1-psi2)
                    # Robin BC: D*dpsi/dr = -(D/d_ext)*psi
                    #           + stochastic correction
                    diag += ar_m + D_j / (dr * d_ext)
                    L[row, offset+mesh.idx(i-1,j)] -= ar_m

                    # Stochastic correction to vacuum BC
                    if j_ph == 0:
                        stoch_bc = (D_j * v[1] * mu
                                    / d_ext)
                    else:
                        stoch_bc = (-D_j * v[0] * mu
                                    / d_ext)
                    diag        += stoch_bc
                    L[row, row_oth] -= stoch_bc

                else:
                    diag += ar_m + ar_p
                    L[row, offset+mesh.idx(i-1,j)] -= ar_m
                    L[row, offset+mesh.idx(i+1,j)] -= ar_p

                # --- Axial diffusion ---
                if j == 0:
                    # Vacuum BC at z=0 (Eq. 83 or 84)
                    diag += az + D_j / (dz * d_ext)
                    if j + 1 < Nz:
                        L[row, offset+mesh.idx(i,j+1)] -= az

                    # Stochastic correction to vacuum BC
                    if j_ph == 0:
                        stoch_bc = (D_j * v[1] * mu
                                    / d_ext)
                    else:
                        stoch_bc = (-D_j * v[0] * mu
                                    / d_ext)
                    diag        += stoch_bc
                    L[row, row_oth] -= stoch_bc

                elif j == Nz - 1:
                    # Vacuum BC at z=H (Eq. 83 or 84)
                    diag += az + D_j / (dz * d_ext)
                    L[row, offset+mesh.idx(i,j-1)] -= az

                    # Stochastic correction
                    if j_ph == 0:
                        stoch_bc = (D_j * v[1] * mu
                                    / d_ext)
                    else:
                        stoch_bc = (-D_j * v[0] * mu
                                    / d_ext)
                    diag        += stoch_bc
                    L[row, row_oth] -= stoch_bc

                else:
                    diag += 2.0 * az
                    L[row, offset+mesh.idx(i,j-1)] -= az
                    L[row, offset+mesh.idx(i,j+1)] -= az

                # --- Interior stochastic coupling ---
                if j_ph == 0:
                    diag        += v[1] * G
                    L[row, row_oth] -= v[1] * G
                else:
                    diag        += v[0] * G
                    L[row, row_oth] -= v[0] * G

                L[row, row] = diag

                # --- Fission ---
                if mat.Sigma_f > 0:
                    F[row, row] = mat.nu * Sf_j

    return csr_matrix(L), csr_matrix(F)


def static_shape_solve(mats: list,
                         mesh: CylindricalMesh2D,
                         T: np.ndarray,
                         v: np.ndarray,
                         mu: float) -> tuple:
    """
    Solve static stochastic eigenvalue problem (Eq. 56)
    with BCs from Eqs. (83)-(84).

    Returns
    -------
    k_eff : float
    psi   : shape (2, Nr, Nz), normalised per Eq. (52)
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    N      = mesh.N

    L, F = build_static_system(mats, mesh, T, v, mu)

    try:
        vals, vecs = sp_eigs(
            F, k=1, M=L,
            sigma=1.0, which='LM',
            tol=1e-8, maxiter=2000
        )
        k_eff    = float(np.real(vals[0]))
        psi_flat = np.real(vecs[:, 0])
    except Exception as e:
        print(f"Sparse eigensolver failed: {e}")
        from scipy.linalg import eig
        vals, vecs = eig(F.toarray(), L.toarray())
        real_mask  = np.abs(np.imag(vals)) < 1e-8
        real_vals  = np.real(vals[real_mask])
        real_vecs  = np.real(vecs[:, real_mask])
        pos_mask   = real_vals > 0
        if not np.any(pos_mask):
            return 0.0, np.zeros((2, Nr, Nz))
        idx      = np.argmax(real_vals[pos_mask])
        k_eff    = float(real_vals[pos_mask][idx])
        psi_flat = real_vecs[:, pos_mask][:, idx]

    if np.mean(psi_flat) < 0:
        psi_flat = -psi_flat

    # Reshape to (2, Nr, Nz)
    psi = np.zeros((2, Nr, Nz))
    for j_ph in range(2):
        psi[j_ph] = (
            psi_flat[j_ph*N:(j_ph+1)*N].reshape(Nr, Nz)
        )
    psi = np.maximum(psi, 0.0)

    # Normalise per Eq. (52): sum_i sum_g int psi_ig dV = 1
    norm = sum(np.sum(psi[j] * mesh.dV) for j in range(2))
    if norm > 0:
        psi /= norm

    return k_eff, psi


def compute_Lambda(mats: list,
                    mesh: CylindricalMesh2D,
                    psi: np.ndarray,
                    T: np.ndarray,
                    neutron_speed: float = 2.2e5) -> float:
    """
    Compute prompt neutron lifetime from Eq. (58):

        Lambda = sum_i int(psi_i/v) dV
               / sum_i int(nu*Sigma_fi*psi_i) dV
    """
    numerator   = 0.0
    denominator = 0.0

    for j_ph, mat in enumerate(mats):
        T_j  = float(np.mean(T[j_ph]))
        Sf_j = mat.Sigma_f_T(T_j)

        numerator   += (np.sum(psi[j_ph] * mesh.dV)
                        / neutron_speed)
        denominator += np.sum(
            mat.nu * Sf_j * psi[j_ph] * mesh.dV
        )

    if abs(denominator) < 1e-30:
        return 1e-5

    return numerator / denominator