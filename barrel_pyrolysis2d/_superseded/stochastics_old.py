"""
stochastics.py  (2D axisymmetric r-z)
======================================
The stochastic-closure spatial operator, shared by thermal.py and
neutronics.py.

The document's closure produces the SAME operator for heat and for
neutrons -- the equations for J_1,J_2 and Q_1,Q_2 are identical under
(K_i, T_i) <-> (D_i, phi_i). Building it once here means the two
physics modules cannot drift apart, and a correction to the closure
lands in both at once.

Derivation (2D axisymmetric)
-----------------------------
With F the phase indicator, <F grad F> = -v_1 v_2 g and
g = grad R|_0, the phase currents are

    J_1 = -C_1 grad T_1 + C_1 v_2 (T_1-T_2) g
    J_2 = -C_2 grad T_2 - C_2 v_1 (T_1-T_2) g

Anchoring the correlation cusp at the PHYSICAL boundaries -- the
furnace at z=0 and the cold/vacuum wall at r=r_max -- gives

    g = mu * ( f(r) r_hat - z_hat )

Note the RELATIVE SIGN. The axial component is anchored at z=0 and
points toward decreasing z (as in the slab); the radial component is
anchored at the outer wall and points toward increasing r. The
document's literal exp(-mu_r r) instead orients the radial part as
though the anchor were the axis, which is the opposite sign.

A form factor f(r) with f(0)=0 is required because r_hat is undefined
on the axis: any vector field regular at r=0 must have vanishing
radial component there. Without it the geometric term below diverges
as 1/r.

Carrying this through -div(J_i) + (coupling) gives, for phase i with
the other phase k,

  rho_i Cp_i dT_i/dt =
      C_i lap(T_i)
    + 2 v_k C_i mu (dT_i/dz - f dT_i/dr)          [self-drift]
    -   v_k mu (C_i + C_k) (dT_k/dz - f dT_k/dr)  [cross-drift]
    + [ v_k (C_i v_k + C_k v_i) mu^2 (1 + f^2)
        - C_i v_k W ] (T_i - T_k)                  [algebraic]
    + S_i

with the geometric / grad-mu field

    W = mu f / r + d(mu f)/dr - d(mu)/dz

Slab reduction (Nr=1, f=0, mu constant) recovers the document's
Section 5 slab equation term for term:

      C_i d2T_i/dz2
    + 2 v_k C_i mu dT_i/dz
    -   v_k mu (C_i + C_k) dT_k/dz
    +   v_k (C_i v_k + C_k v_i) mu^2 (T_i - T_k)

which is checked numerically in verification_tests_2d.py.

Three things are genuinely new relative to the slab:
  1. the algebraic coefficient carries (1 + f^2), i.e. |g|^2 = 2 mu^2
     away from the axis rather than mu^2 -- a factor of 2 that cannot
     be obtained by pattern-matching the 1D form;
  2. the geometric term mu f / r, with no 1D analogue;
  3. the grad-mu terms, which vanish for constant mu but NOT once
     percolation makes mu = mu(r,z,t).

Units
------
This module is unit-agnostic but NOT unit-forgiving: dr, dz, r and
mu^-1 must all be in the SAME length unit, and C_i expressed per that
unit squared. thermal.py works in metres (K is W/m/K) so it converts
the cm mesh and cm^-1 mu before calling; neutronics.py works in cm
(D is in cm) and passes them straight through.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


# ── Form factor on the axis ──────────────────────────────────────────

def form_factor(r: np.ndarray, ell0: float, mode: str = "tanh") -> np.ndarray:
    """
    f(r), the radial form factor enforcing regularity at the axis.

    mode="tanh"  : f = tanh(r/ell0). f(0)=0 as required; f -> 1 within
                   a few correlation lengths of the axis. Introduces no
                   new fitted parameter (ell0 is already in the model).
    mode="one"   : f = 1 everywhere -- the document's literal form.
                   Regularity is VIOLATED at the axis; the geometric
                   term then behaves as mu/r, finite on a cell-centred
                   mesh (r_c = dr/2) but growing without bound under
                   radial refinement. For comparison only.
    mode="zero"  : f = 0 -- no radial stochastic drift at all. The
                   operator reduces to axial drift plus ordinary
                   cylindrical conduction.

    Under the document's own scale-separation requirement (ell << dr),
    tanh saturates to 1 at every cell centre including the innermost,
    so "tanh" and "one" agree to machine precision wherever the closure
    is valid. The form factor only does work in the regime where the
    closure is already suspect -- it is insurance, not physics.
    """
    if mode == "one":
        return np.ones_like(r)
    if mode == "zero":
        return np.zeros_like(r)
    if mode == "tanh":
        return np.tanh(r / max(ell0, 1e-30))
    raise ValueError(f"unknown form-factor mode: {mode!r}")


# ── Geometric / grad-mu field ────────────────────────────────────────

def geometric_field(mu: np.ndarray, f: np.ndarray,
                     r: np.ndarray, dr: float, dz: float) -> np.ndarray:
    """
    W = mu f / r + d(mu f)/dr - d(mu)/dz,  shape (Nr, Nz).

    Every piece vanishes in the slab limit (f=0, mu constant), so W is
    exactly the part of the operator with no 1D counterpart.

    Derivatives are central where possible, one-sided at the edges.
    mu and f must already be broadcast to (Nr, Nz).
    """
    muf = mu * f
    W = muf / r[:, None]

    dmuf_dr = np.zeros_like(muf)
    if muf.shape[0] > 2:
        dmuf_dr[1:-1] = (muf[2:] - muf[:-2]) / (2.0 * dr)
    if muf.shape[0] > 1:
        dmuf_dr[0] = (muf[1] - muf[0]) / dr
        dmuf_dr[-1] = (muf[-1] - muf[-2]) / dr
    W = W + dmuf_dr

    dmu_dz = np.zeros_like(mu)
    if mu.shape[1] > 2:
        dmu_dz[:, 1:-1] = (mu[:, 2:] - mu[:, :-2]) / (2.0 * dz)
    if mu.shape[1] > 1:
        dmu_dz[:, 0] = (mu[:, 1] - mu[:, 0]) / dz
        dmu_dz[:, -1] = (mu[:, -1] - mu[:, -2]) / dz
    W = W - dmu_dz

    return W


# ── Operator assembly ────────────────────────────────────────────────

def _upwind(L, row, coeff, idx_self, idx_fwd, idx_bwd, h):
    """
    Add  coeff * d(u)/dx  to `row` using STABLE upwinding.

    For dU/dt = a dU/dx the characteristics move at -a, so information
    arrives from +x when a>0 (forward difference) and from -x when a<0
    (backward difference). Differencing from the wrong side gives the
    downwind scheme, which is unconditionally unstable -- this is the
    one discretisation choice in the operator that is not fixed by
    symmetry.

    Falls back to a one-sided difference toward whichever neighbour
    exists at a boundary. If neither exists (a degenerate single-cell
    direction, e.g. Nr=1) the term is dropped, which is correct: one
    cell has no resolvable gradient in that direction.
    """
    if coeff == 0.0:
        return
    if coeff > 0.0:
        if idx_fwd is not None:
            L[row, idx_fwd] += coeff / h
            L[row, idx_self] -= coeff / h
        elif idx_bwd is not None:
            L[row, idx_self] += coeff / h
            L[row, idx_bwd] -= coeff / h
    else:
        if idx_bwd is not None:
            L[row, idx_self] += coeff / h
            L[row, idx_bwd] -= coeff / h
        elif idx_fwd is not None:
            L[row, idx_fwd] += coeff / h
            L[row, idx_self] -= coeff / h


def algebraic_coefficient(Ci, Ck, v_self, v_oth, mu, f, W):
    """
    The coefficient A multiplying (T_i - T_k) in the algebraic coupling:

        A = v_k (C_i v_k + C_k v_i) mu^2 (1 + f^2)  -  C_i v_k W

    Factored out so it can be unit-tested directly. In the assembled
    operator this lands on BOTH the diagonal (+A) and the same-cell
    off-diagonal block (-A), where it is superposed with the cross-drift
    term -- so reading it back off the matrix does not isolate it.

    The (1 + f^2) is |g|^2 / mu^2: away from the axis f -> 1 and the
    coupling is TWICE the slab value, since g has two components of
    equal magnitude. The -C_i v_k W piece has no slab analogue at all.
    """
    return (v_oth * (Ci * v_oth + Ck * v_self) * mu * mu * (1.0 + f * f)
            - Ci * v_oth * W)


def build_coupled_operator(C1, C2, v, mu, f, r, r_edge,
                            dr: float, dz: float,
                            Nr: int, Nz: int,
                            outer_r: str = "neumann") -> csr_matrix:
    """
    Assemble the (2N, 2N) spatial operator L for the coupled two-phase
    system, N = Nr*Nz, unknowns ordered [phase1 block; phase2 block]
    with flat index i*Nz + j inside each block.

    Parameters
    ----------
    C1, C2   : transport coefficients (K for heat, D for neutrons),
               scalar or (Nr, Nz)
    v        : (2,) volume fractions [v1, v2]
    mu       : inverse correlation length, scalar or (Nr, Nz)
    f        : radial form factor, (Nr,) or (Nr, Nz)
    r        : cell-centre radii, (Nr,)
    r_edge   : cell-edge radii, (Nr+1,)
    dr, dz   : cell sizes, same length unit as r and 1/mu
    outer_r  : "neumann" (zero flux at r_max) or "open" (no ghost
               coupling added; the caller imposes Dirichlet/vacuum by
               overwriting the row or adding a Robin term)

    Returns
    -------
    L : csr_matrix (2N, 2N)

    Returned WITHOUT any time-stepping factor: callers form (I - dt*L)
    or an eigenvalue problem from it as needed. Axial boundaries are
    left natural (zero flux) here; thermal.py applies the furnace flux
    at z=0 through the RHS and neutronics.py imposes its extrapolated
    vacuum conditions on the boundary rows.
    """
    N = Nr * Nz
    C1 = np.broadcast_to(np.asarray(C1, dtype=float), (Nr, Nz))
    C2 = np.broadcast_to(np.asarray(C2, dtype=float), (Nr, Nz))
    mu = np.broadcast_to(np.asarray(mu, dtype=float), (Nr, Nz))
    f = np.asarray(f, dtype=float)
    if f.ndim == 1:
        f = np.broadcast_to(f[:, None], (Nr, Nz))
    f = np.broadcast_to(f, (Nr, Nz))

    W = geometric_field(mu, f, r, dr, dz)

    L = lil_matrix((2 * N, 2 * N))

    def idx(ph, i, j):
        return ph * N + i * Nz + j

    C = (C1, C2)
    for ph in range(2):
        oth = 1 - ph
        Ci, Ck = C[ph], C[oth]
        v_self, v_oth = float(v[ph]), float(v[oth])

        for i in range(Nr):
            r_c = r[i]
            r_m = r_edge[i]
            r_p = r_edge[i + 1]

            for j in range(Nz):
                row = idx(ph, i, j)
                cs = Ci[i, j]
                ck = Ck[i, j]
                m = mu[i, j]
                ff = f[i, j]

                # ---- Radial diffusion (finite volume, cylindrical) ----
                # r=0: r_edge[0] = 0, so the inner face area vanishes
                # identically -- symmetry is enforced by the geometry,
                # no special case needed.
                if i > 0:
                    a_m = cs * r_m / (r_c * dr * dr)
                    L[row, idx(ph, i - 1, j)] += a_m
                    L[row, row] -= a_m
                if i < Nr - 1:
                    a_p = cs * r_p / (r_c * dr * dr)
                    L[row, idx(ph, i + 1, j)] += a_p
                    L[row, row] -= a_p
                # i == Nr-1: "neumann" -> zero flux, nothing added.
                # "open" -> also nothing; caller supplies the closure.

                # ---- Axial diffusion ----
                if j > 0:
                    a = cs / (dz * dz)
                    L[row, idx(ph, i, j - 1)] += a
                    L[row, row] -= a
                if j < Nz - 1:
                    a = cs / (dz * dz)
                    L[row, idx(ph, i, j + 1)] += a
                    L[row, row] -= a

                # ---- Neighbour indices for drift ----
                i_f = idx(ph, i + 1, j) if i < Nr - 1 else None
                i_b = idx(ph, i - 1, j) if i > 0 else None
                j_f = idx(ph, i, j + 1) if j < Nz - 1 else None
                j_b = idx(ph, i, j - 1) if j > 0 else None
                oi_f = idx(oth, i + 1, j) if i < Nr - 1 else None
                oi_b = idx(oth, i - 1, j) if i > 0 else None
                oj_f = idx(oth, i, j + 1) if j < Nz - 1 else None
                oj_b = idx(oth, i, j - 1) if j > 0 else None

                # ---- Self-drift: +2 v_k C_i mu (dT_i/dz - f dT_i/dr)
                b = 2.0 * v_oth * cs * m
                _upwind(L, row, b, row, j_f, j_b, dz)
                _upwind(L, row, -b * ff, row, i_f, i_b, dr)

                # ---- Cross-drift: -v_k mu (C_i+C_k)(dT_k/dz - f dT_k/dr)
                bx = -v_oth * m * (cs + ck)
                _upwind(L, row, bx, idx(oth, i, j), oj_f, oj_b, dz)
                _upwind(L, row, -bx * ff, idx(oth, i, j), oi_f, oi_b, dr)

                # ---- Algebraic coupling ----
                A_alg = algebraic_coefficient(cs, ck, v_self, v_oth,
                                               m, ff, W[i, j])
                L[row, row] += A_alg
                L[row, idx(oth, i, j)] -= A_alg

    return csr_matrix(L)
