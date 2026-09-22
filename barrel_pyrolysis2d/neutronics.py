"""
neutronics.py  (2D axisymmetric r-z)
=====================================
Two-group, two-phase stochastic neutron diffusion on a cylindrical
(r, z) mesh -- the 2D counterpart of barrel_pyrolysis1d/neutronics.py,
built on the same basis:

  * ONE eigenvalue for the coupled two-phase system (not per-phase
    k_i weighted by volume fraction; see the 1D module for why);
  * the RELAXATIONAL closure: phases exchange at the rate
        G_i = v_k (D_i v_k + D_k v_i) |m|^2,   |m|^2 = mu_r^2 + mu_z^2,
    with no odd-order drift terms;
  * Marshak vacuum boundaries, d_ext = 2.1312 D, on z = 0, z = H and the
    drum wall r = R; symmetry on the axis r = 0 (zero-area face);
  * thermal-group 1/v factor at the NEUTRON temperature (the
    moderator's), material temperature through Doppler only;
  * absolute amplitude: phi = P psi in n/cm^2/s, with the fuel phase's
    psi normalised to unit VOLUME integral (so P is in n cm/s), and the
    intrinsic source entering point kinetics via source_amplitude_rate.

Only the relaxational closure is provided in 2D. The document closure's
odd-order terms were shown against the 1D realisation benchmark to give
a negative combustible flux and no fundamental mode at moderate mu, so
there is no reason to carry them into 2D.

NOTE ON |m|^2. In 2D the exchange coefficient carries mu_r^2 + mu_z^2,
twice the slab value when mu_r = mu_z. That follows from the closure,
but it means the 1D and 2D models assign DIFFERENT exchange rates to the
same isotropic medium. The realisation benchmark exists only in 1D, so
the 2D exchange rate is unvalidated; a 2D (r-z) realisation benchmark
is the way to settle it.

Units: lengths cm, cross sections cm^-1, D cm, mu cm^-1.
"""

import warnings
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu, eigs, LinearOperator, ArpackNoConvergence

from mesh import CylindricalMesh2D
from materials import N_GROUPS, CHI

D_EXT_FACTOR = 2.1312       # Milne: 0.7104 * lambda_tr, lambda_tr = 3 D


class ClosureBreakdownError(RuntimeError):
    """Raised when there is no converged, physical fundamental mode."""


def neutron_temperature(phases: list, T: np.ndarray) -> float:
    """Mean temperature of the moderating phase (larger Sigma_s12)."""
    m = int(np.argmax([ph.Sigma_s12 for ph in phases]))
    return float(np.mean(T[m]))


def cell_volumes(mesh: CylindricalMesh2D) -> np.ndarray:
    """Annular cell volumes, cm^3, shape (Nr, Nz)."""
    re = mesh.r_edge
    ring = np.pi * (re[1:]**2 - re[:-1]**2)
    return np.outer(ring, np.full(mesh.Nz, mesh.dz))


def _group_loss(D: float, Srem: float, G_self: np.ndarray,
                mesh: CylindricalMesh2D, radial_bc: str) -> sps.csr_matrix:
    """
    Loss operator (per unit volume) for one phase and group:
        -div(D grad psi) + (Sigma_removal + G) psi
    finite volume on annular cells. Returns a sparse (N, N) matrix.
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    dr, dz = mesh.dr, mesh.dz
    r, re = mesh.r, mesh.r_edge
    d_ext = D_EXT_FACTOR * D
    rows, cols, vals = [], [], []

    def add(a, b, x):
        rows.append(a); cols.append(b); vals.append(x)

    for i in range(Nr):
        for j in range(Nz):
            n = mesh.idx(i, j)
            diag = Srem + G_self[i, j]
            # radial faces (area factor r_face / (r_c dr))
            if i > 0:
                c = D * re[i] / (r[i] * dr * dr)
                diag += c
                add(n, mesh.idx(i - 1, j), -c)
            # i == 0: axis face has r_edge = 0 -> no flux (symmetry)
            if i < Nr - 1:
                c = D * re[i + 1] / (r[i] * dr * dr)
                diag += c
                add(n, mesh.idx(i + 1, j), -c)
            elif radial_bc == "vacuum":
                diag += D * re[i + 1] / (r[i] * dr) / d_ext
            # radial_bc == "reflective": no outer-wall leakage
            # axial faces
            if j > 0:
                diag += D / dz**2
                add(n, mesh.idx(i, j - 1), -D / dz**2)
            else:
                diag += D / (dz * d_ext)
            if j < Nz - 1:
                diag += D / dz**2
                add(n, mesh.idx(i, j + 1), -D / dz**2)
            else:
                diag += D / (dz * d_ext)
            add(n, n, diag)
    return sps.csr_matrix((vals, (rows, cols)), shape=(mesh.N, mesh.N))


def _build_system(phases: list, mesh: CylindricalMesh2D, T: np.ndarray,
                  v: np.ndarray, mu_r, mu_z, radial_bc: str):
    """Joint loss A and fission F for [phase0 fast, phase0 thermal,
    phase1 fast, phase1 thermal], each block N = Nr*Nz."""
    N = mesh.N
    shape = (mesh.Nr, mesh.Nz)
    mr = np.broadcast_to(np.asarray(mu_r, dtype=float), shape)
    mz = np.broadcast_to(np.asarray(mu_z, dtype=float), shape)
    m2 = mr**2 + mz**2
    T_n = neutron_temperature(phases, T)
    I = sps.identity(N, format="csr")
    Z = sps.csr_matrix((N, N))
    A = [[None] * 4 for _ in range(4)]
    Fb = [[Z] * 4 for _ in range(4)]
    for p, ph in enumerate(phases):
        k = 1 - p
        po = phases[k]
        Sa = ph.Sa_T(float(np.mean(T[p])), T_n)
        nuSf = ph.nuSf_T(float(np.mean(T[p])), T_n)
        for g in range(N_GROUPS):
            Srem = Sa[g] + (ph.Sigma_s12 if g == 0 else 0.0)
            G = v[k] * (ph.D[g] * v[k] + po.D[g] * v[p]) * m2   # relaxational
            A[2 * p + g][2 * p + g] = _group_loss(ph.D[g], Srem, G, mesh, radial_bc)
            A[2 * p + g][2 * k + g] = sps.diags(-G.ravel())
        A[2 * p + 1][2 * p + 0] = -ph.Sigma_s12 * I          # downscatter
        for g in range(N_GROUPS):                              # chi into groups
            for gp in range(N_GROUPS):
                if CHI[g] > 0 and nuSf[gp] > 0:
                    Fb[2 * p + g][2 * p + gp] = CHI[g] * nuSf[gp] * I
    for a in range(4):
        for b in range(4):
            if A[a][b] is None:
                A[a][b] = Z
    return sps.bmat(A, format="csc"), sps.bmat(Fb, format="csr")


def static_shape_solve(phases: list, mesh: CylindricalMesh2D, T: np.ndarray,
                       v: np.ndarray, mu_r, mu_z, psi_init=None,
                       radial_bc: str = "vacuum") -> tuple:
    """
    Fundamental mode of the coupled two-phase system.

    Returns
    -------
    k_eff : float
    psi   : (2 phases, 2 groups, Nr, Nz); fuel phase normalised to unit
            volume integral (summed over groups), the other phase scaled
            by the same factor so psi_2/psi_1 is physical.
    """
    A, F = _build_system(phases, mesh, T, v, mu_r, mu_z, radial_bc)
    n = A.shape[0]
    lu = splu(A)
    op = LinearOperator((n, n), matvec=lambda x: lu.solve(F @ x), dtype=float)
    x0 = (np.asarray(psi_init, float).ravel() if psi_init is not None
          else np.ones(n))
    try:
        vals, vecs = eigs(op, k=1, which="LM", v0=x0, tol=1e-12, maxiter=20000)
    except ArpackNoConvergence as e:
        raise ClosureBreakdownError(f"eigenvalue solve did not converge: {e}")
    lam = vals[0]
    if abs(lam.imag) > 1e-9 * max(abs(lam.real), 1.0) or lam.real <= 0:
        raise ClosureBreakdownError(f"no physical fundamental mode (k = {lam!r})")
    k_eff = float(lam.real)
    psi = vecs[:, 0].real.reshape(2, N_GROUPS, mesh.Nr, mesh.Nz)
    if psi[0].sum() < 0:
        psi = -psi
    Vc = cell_volumes(mesh)
    s1 = float(np.sum(psi[0] * Vc[None, :, :]))
    psi = psi / s1
    if np.any(psi < -1e-8 * np.abs(psi).max()):
        warnings.warn("fundamental mode changes sign: not physical",
                      RuntimeWarning, stacklevel=2)
    return k_eff, psi


def compute_Lambda(phases: list, mesh: CylindricalMesh2D, psi: np.ndarray,
                   T: np.ndarray, v: np.ndarray) -> float:
    """Generation time, volume-fraction weighted, unit weight function."""
    Vc = cell_volumes(mesh)
    T_n = neutron_temperature(phases, T)
    num = den = 0.0
    for p, ph in enumerate(phases):
        nuSf = ph.nuSf_T(float(np.mean(T[p])), T_n)
        for g in range(N_GROUPS):
            num += v[p] * np.sum(psi[p, g] * Vc) / ph.v_n[g]
            den += v[p] * np.sum(nuSf[g] * psi[p, g] * Vc)
    return num / den if abs(den) > 1e-30 else 1e-5


def source_amplitude_rate(phases: list, mesh: CylindricalMesh2D,
                          psi: np.ndarray, v: np.ndarray,
                          source_density: float) -> float:
    """Intrinsic source in amplitude units/s (cf. the 1D function):
    q = (sum_p v_p Q_p V) / (sum_p v_p sum_g int psi/v_n dV)."""
    if not source_density:
        return 0.0
    Vc = cell_volumes(mesh)
    V = float(Vc.sum())
    S_tot = sum(v[p] * source_density * V
                for p, ph in enumerate(phases) if np.any(ph.nu_Sf > 0))
    N_w = sum(v[p] * np.sum(psi[p, g] * Vc) / ph.v_n[g]
              for p, ph in enumerate(phases) for g in range(N_GROUPS))
    return S_tot / N_w if N_w > 0 else 0.0
