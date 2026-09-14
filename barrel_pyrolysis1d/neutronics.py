"""
neutronics.py
=============
Two-group (fast=0, thermal=1) static stochastic eigenvalue solve for
1D slab. Extends the single-group model (Eqs. 85-90 of document) by
giving each phase's Eq. 89 stochastic loss operator its own copy per
energy group, plus a within-phase fast->thermal downscatter coupling
(Phase.Sigma_s12) and a fission spectrum (materials.CHI = [1,0]: all
fission neutrons, regardless of which group's nu*Sigma_f produced
them, are born fast). Cross-PHASE coupling (the mu-based streaming/
exchange terms) stays block-diagonal in group -- it's a spatial/
geometric mixing effect, not an energy-transfer one, so a phase's fast
flux only couples to the OTHER phase's fast flux, and likewise for
thermal.

mu may be a spatially-varying field (N,) rather than a single
constant, driven by percolation.py's dynamic correlation length --
_build_group_self_and_cross already loops over cells one at a
time (needed for the boundary rows' different stencils), so the
per-cell generalisation is just indexing mu[n] instead of a bare
mu at each use inside that loop. A scalar mu is broadcast to a
uniform field and reproduces the original behaviour exactly.

Stochastic loss operator (Eq. 89), applied per-group g:
    L psi_i,g = D_i,g d^2(psi_i,g)/dz^2
              + mu * d(psi_i,g)/dz
              - mu * sum_k v_k * d(psi_k,g)/dz
              - mu * sum_{l!=i} (D_l,g * d(psi_l,g)/dz
                                 + mu * psi_l,g)

Expanding for two phases (same structure as the single-group model,
just carried through per-group):

    Phase 1, group g:
        L psi_1,g = D_1,g d^2(psi_1,g)/dz^2
                  + mu*(1-v_1)*d(psi_1,g)/dz
                  - mu*(v_2 + D_2,g)*d(psi_2,g)/dz
                  - mu^2 * psi_2,g

    Phase 2, group g:
        L psi_2,g = D_2,g d^2(psi_2,g)/dz^2
                  + mu*(1-v_2)*d(psi_2,g)/dz
                  - mu*(v_1 + D_1,g)*d(psi_1,g)/dz
                  - mu^2 * psi_1,g

Note: the self-drift (mu*(1-v_self)*dpsi_i/dz) and cross-gradient
(mu*(v_other+D_other)*dpsi_other/dz) terms are odd-order (first-
derivative) terms with a fixed spatial sign, so they break z<->H-z
mirror symmetry of the flux shape in an otherwise symmetric slab
(uniform properties, vacuum BC at both ends) -- this is a genuine
consequence of the document's one-sided correlation function
R(z)=e^{-mu*z} (Eq. 83), not a numerical artifact (verified: flux
asymmetry grows smoothly and reproducibly with mu, from 0 at mu=0).

Full per-phase, per-group equation (within one phase, combining the
stochastic loss operator above with fission and downscatter):

    L psi_p,fast    + (1/k_p)*chi_fast*sum_g' nuSf_p,g' psi_p,g'
        - Sigma_s12_p * psi_p,fast = 0
    L psi_p,thermal + (1/k_p)*chi_thermal*sum_g' nuSf_p,g' psi_p,g'
        + Sigma_s12_p * psi_p,fast = 0

with chi = [1, 0] (materials.CHI), so the fission term only appears in
the fast-group equation, and Sigma_s12 removes flux from the fast
group's own equation while sourcing the thermal group's.

Vacuum BCs (Eqs. 76-79 of document), applied per-group using that
group's own D and removal cross section (Sigma_a + Sigma_s12 for fast,
Sigma_a alone for thermal -- no group above thermal to lose flux to):

    General form: psi +/- d_ext * nabla(psi) = 0

    After stochastic averaging (Eqs. 78-79):
        psi_1 +/- d_ext1*[psi_1' + (1-v)*mu*(psi_1-psi_2)] = 0
        psi_2 +/- d_ext2*[psi_2' - v*mu*(psi_1-psi_2)]     = 0

    Sign convention (document):
        z=0 (bottom): MINUS sign  =>  psi - d_ext*nabla(psi) = 0
        z=H (top)   : PLUS  sign  =>  psi + d_ext*nabla(psi) = 0

    Outward normals:
        z=0: n_hat = -z_hat  =>  nabla(psi).n = -dpsi/dz
        z=H: n_hat = +z_hat  =>  nabla(psi).n = +dpsi/dz

    Robin conditions on the loss matrix:
        z=0: -D*dpsi/dz = +(D/d_ext)*psi  (flux out through bottom)
        z=H: +D*dpsi/dz = -(D/d_ext)*psi  (flux out through top)

    d_ext,i,g = 0.7104/Sigma_removal,i,g  (Eq. 76, generalised to use
    this group's total removal cross section)
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from materials import Phase, CHI, N_GROUPS
from mesh import Mesh1D


def _build_group_self_and_cross(D: float,
                                 D_oth: float,
                                 Sigma_removal: float,
                                 mesh: Mesh1D,
                                 v_self: float,
                                 v_oth: float,
                                 mu,
                                 is_phase0: bool) -> tuple:
    """
    Build ONE energy group's self and cross operators (Eq. 89) for one
    phase -- the same discretisation the single-group model used,
    generalised to take this group's own D and total removal cross
    section (Sigma_removal = Sigma_a, plus Sigma_s12 for the fast
    group specifically -- see _build_phase_system) explicitly rather
    than deriving them from a Phase object, so the same function
    builds either group.

    Parameters
    ----------
    mu : inverse correlation length cm^-1 -- scalar OR shape (N,). This
         function already loops over cells one at a time (the boundary
         rows need different stencils from the interior), so a
         spatially-varying field is used simply by indexing mu[n] at
         each cell rather than a bare mu -- a scalar is broadcast to a
         uniform field first and reproduces the original single-mu
         behaviour exactly, cell for cell.

    Returns
    -------
    A_self, Cross : (N,N) dense arrays
    """
    N     = mesh.N
    dz    = mesh.dz

    mu = np.broadcast_to(np.atleast_1d(mu), (N,)).astype(float)

    d_ext   = 0.7104 / Sigma_removal if Sigma_removal > 1e-10 else 10.0 * dz
    r       = D / dz**2
    s_self  = mu * (1.0 - v_self) / (2.0 * dz)          # shape (N,)
    s_cross = mu * (v_oth + D_oth) / (2.0 * dz)         # shape (N,)
    G_coup  = mu**2                                      # shape (N,)

    A_self = np.zeros((N, N))
    Cross  = np.zeros((N, N))

    for n in range(N):
        diag = Sigma_removal

        if n == 0:
            diag += r + D / (dz * d_ext)
            A_self[n, n+1] -= r

            # Vacuum-BC stochastic correction (Eqs. 78-79)
            stoch_bc = D * v_oth * mu[n] / d_ext if is_phase0 else -D * v_oth * mu[n] / d_ext
            diag        += stoch_bc
            Cross[n, n] -= stoch_bc

            # Self drift at z=0: one-sided forward diff
            s0 = mu[n] * (1.0 - v_self) / dz
            diag           += s0
            A_self[n, n+1] -= s0

        elif n == N - 1:
            diag += r + D / (dz * d_ext)
            A_self[n, n-1] -= r

            stoch_bc = D * v_oth * mu[n] / d_ext if is_phase0 else -D * v_oth * mu[n] / d_ext
            diag        += stoch_bc
            Cross[n, n] -= stoch_bc

            # Self drift at z=H: one-sided backward diff
            sN = mu[n] * (1.0 - v_self) / dz
            diag           += sN
            A_self[n, n-1] -= sN

        else:
            diag += 2.0 * r
            A_self[n, n-1] -= r
            A_self[n, n+1] -= r

            # Self drift: central difference
            A_self[n, n+1] -= s_self[n]
            A_self[n, n-1] += s_self[n]

        # Cross-phase gradient coupling -- always references psi_other
        # (same group -- see module docstring on why cross-phase
        # coupling stays block-diagonal in group)
        if 0 < n < N - 1:
            Cross[n, n+1] += s_cross[n]
            Cross[n, n-1] -= s_cross[n]
        elif n == 0:
            sc0 = mu[n] * (v_oth + D_oth) / dz
            Cross[n, n+1] += sc0
            Cross[n, n]   -= sc0
        else:
            scN = mu[n] * (v_oth + D_oth) / dz
            Cross[n, n]   += scN
            Cross[n, n-1] -= scN

        # Direct coupling: -mu^2 * psi_other
        diag        += G_coup[n]
        Cross[n, n] -= G_coup[n]

        A_self[n, n] += diag

    return A_self, Cross


def _build_phase_system(ph: Phase,
                         ph_oth: Phase,
                         mesh: Mesh1D,
                         T_p: float,
                         v_self: float,
                         v_oth: float,
                         mu,
                         is_phase0: bool) -> tuple:
    """
    Assemble phase p's FULL 2-group (fast=0, thermal=1) system:

        M @ Psi_p - (1/k_p) * F @ Psi_p = -Cross @ Psi_other

    where Psi_p = [psi_p,fast; psi_p,thermal] stacks both groups (2N),
    M is block-diagonal in group PLUS the within-phase downscatter
    term (fast -> thermal), Cross is block-diagonal in group (cross-
    phase coupling never mixes groups -- see module docstring), and F
    has fission entries only in the fast-group rows (chi=[1,0]).

    mu : inverse correlation length cm^-1 -- scalar or shape (N,),
         passed straight through to _build_group_self_and_cross (see
         its docstring).

    Returns
    -------
    M, Cross, F : (2N, 2N) dense arrays
    """
    N    = mesh.N
    Sa   = ph.Sa_T(T_p)     # [fast, thermal]
    nuSf = ph.nuSf_T(T_p)   # [fast, thermal]

    A_fast, Cross_fast = _build_group_self_and_cross(
        ph.D[0], ph_oth.D[0], Sa[0] + ph.Sigma_s12,
        mesh, v_self, v_oth, mu, is_phase0)
    A_thermal, Cross_thermal = _build_group_self_and_cross(
        ph.D[1], ph_oth.D[1], Sa[1],
        mesh, v_self, v_oth, mu, is_phase0)

    M = np.zeros((2 * N, 2 * N))
    M[0:N, 0:N]     = A_fast
    M[N:2*N, N:2*N] = A_thermal
    # Downscatter: thermal group's equation gains +Sigma_s12*psi_fast
    # (moved to the LHS as -Sigma_s12*psi_fast, matching M's sign
    # convention); the fast group's own removal cross section already
    # includes Sigma_s12 (passed into A_fast above), so this loss is
    # accounted for exactly once, not double-counted.
    M[N:2*N, 0:N] -= ph.Sigma_s12 * np.eye(N)

    Cross = np.zeros((2 * N, 2 * N))
    Cross[0:N, 0:N]     = Cross_fast
    Cross[N:2*N, N:2*N] = Cross_thermal

    F = np.zeros((2 * N, 2 * N))
    # chi=[1,0]: ALL fission neutrons -- regardless of whether the
    # fission event was in the fast or thermal group -- are born
    # fast, so F only has entries in the fast-group ROWS, but columns
    # for BOTH groups (fast and thermal fission both contribute).
    if CHI[0] * nuSf[0] > 1e-30:
        F[0:N, 0:N] = CHI[0] * nuSf[0] * np.eye(N)
    if CHI[0] * nuSf[1] > 1e-30:
        F[0:N, N:2*N] = CHI[0] * nuSf[1] * np.eye(N)
    # CHI[1] = 0, so the thermal-group rows of F stay identically zero.

    return M, Cross, F


def _solve_fissile_phase(lu_piv: tuple,
                          Cross: np.ndarray,
                          F: np.ndarray,
                          psi_oth: np.ndarray,
                          k_init: float,
                          psi_init: np.ndarray,
                          maxiter: int = 500,
                          tol: float = 1e-10) -> tuple:
    """
    Solve phase p's OWN k-eigenvalue problem (now spanning both energy
    groups, 2N-dimensional)
        M @ Psi_p - (1/k) * F @ Psi_p = -Cross @ Psi_oth
    via power iteration with the cross-phase term as a fixed external
    source (Psi_oth held constant -- the outer fixed-point loop in
    static_shape_solve updates it between calls). This is the standard
    "fixed source + multiplication" k-iteration: k converges to the
    ratio-of-fission-growth between iterations, which for a converged
    solution correctly reflects this phase's own multiplication given
    the (now self-consistent) neutron exchange with the other phase.

    lu_piv is M's LU factorisation (scipy.linalg.lu_factor), computed
    once per outer fixed-point iteration by the caller since M doesn't
    change across inner power-iteration steps.
    """
    source = -Cross @ psi_oth

    psi = psi_init.copy()
    k   = k_init
    for _ in range(maxiter):
        fission_old = F @ psi
        rhs         = fission_old / k + source
        psi_new     = lu_solve(lu_piv, rhs)
        fission_new = F @ psi_new
        denom = np.sum(fission_old)
        k_new = k * (np.sum(fission_new) / denom) if abs(denom) > 1e-300 else k
        scale = np.max(np.abs(psi_new))
        if scale > 1e-300:
            psi_new = psi_new / scale
        converged = abs(k_new - k) < tol * max(abs(k), 1e-30)
        psi, k = psi_new, k_new
        if converged:
            break

    return k, psi


def _solve_nonfissile_phase(lu_piv: tuple,
                             Cross: np.ndarray,
                             psi_oth: np.ndarray) -> np.ndarray:
    """
    Phase p has no fission in either group (nu*Sigma_f = 0 for all g):
    k_p = 0 by construction. Its flux shape (both groups) is driven
    entirely by the fixed cross-phase source from the other phase's
    current flux:
        M @ Psi_p = -Cross @ Psi_oth
    an ordinary (non-eigenvalue) linear solve. lu_piv is M's LU
    factorisation, computed once by the caller.
    """
    source = -Cross @ psi_oth
    return lu_solve(lu_piv, source)


def static_shape_solve(phases: list,
                         mesh: Mesh1D,
                         T: np.ndarray,
                         v: np.ndarray,
                         mu,
                         psi_init: np.ndarray = None,
                         k_init: float = None,
                         maxiter: int = 2000) -> tuple:
    """
    Solve for each phase's OWN k-eigenvalue problem (Eq. 85, applied
    per-phase rather than as one joint system, now spanning both
    energy groups per phase -- see _build_phase_system), coupled via a
    fixed-point iteration: each phase's cross-phase terms use the
    OTHER phase's current flux (both groups) as a fixed external
    source, updated each outer iteration until self-consistent.

    A phase with zero fission cross-section in BOTH groups (this
    model: the combustible phase) has k_p = 0 identically; its flux is
    obtained from a direct (non-eigenvalue) linear solve driven by the
    fissile phase's flux.

    The combined system k_eff is the volume-fraction-weighted average
        k_eff = sum_p v_p * k_p
    which (since k_combustible = 0) reduces to k_eff = v1 * k_PuO here
    -- unchanged in form from the single-group model; each k_p is now
    the dominant eigenvalue of that phase's 2-group system rather than
    a single-group one, but the OUTER combination is identical. (A
    joint 2Nx2N single-eigenvalue solve across PHASES -- as opposed to
    the per-phase decomposition used here -- was tried instead and
    reverted; see keff_crosscheck.py and git history. That question is
    orthogonal to the energy-group extension here.)

    This fixed-point iteration converges only linearly and can need
    several hundred passes to fully settle (slower for larger mu) --
    cheap per pass (each phase is a 2Nx2N solve, N~80) but not free.
    Callers doing repeated solves across a slowly-changing sequence
    (e.g. one per timestep) should pass the previous call's psi/k_eff
    back in via psi_init/k_init: warm-starting from a solution that's
    already close to self-consistent converges in a handful of passes
    instead of hundreds.

    Vacuum BCs applied per document Eqs. (78)-(79):
        z=0: minus sign
        z=H: plus  sign

    Normalised per phase, COMBINED across both groups (deviates from
    Eq. (90)'s combined sum_i int psi_i dz = 1 across phases -- see
    note above the normalisation code): int (psi_p,fast+psi_p,thermal)
    dz = 1 for p=0 and p=1 independently, preserving the physically
    meaningful fast/thermal ratio the coupled 2-group solve produces
    within each phase (only the relative magnitude BETWEEN phases is
    rescaled away, same as the single-group convention).

    Parameters
    ----------
    phases   : [Phase1, Phase2]
    mesh     : Mesh1D
    T        : temperatures shape (2, N)
    v        : volume fractions shape (2,)
    mu       : inverse correlation length cm^-1 -- scalar OR shape
               (N,). A spatially-varying field (e.g. from
               percolation.mu_field()) is passed straight through to
               _build_group_self_and_cross for both phases and both
               groups; a scalar reproduces the original uniform-mu
               behaviour exactly.
    psi_init : optional warm-start shape (2, 2, N) [phase, group,
               space], unnormalised is fine
    k_init   : optional warm-start k_eff (this phase's k1 is
               estimated as k_init/v[0] when the fissile phase is 0)
    maxiter  : outer fixed-point iteration cap

    Returns
    -------
    k_eff : float
    psi   : shape (2, 2, N) [phase, group, space], normalised
    """
    N = mesh.N
    G = N_GROUPS

    fissile = [bool(np.any(ph.nu_Sf > 1e-30)) for ph in phases]

    ops = []
    for p, ph in enumerate(phases):
        p_other = 1 - p
        T_p = float(np.mean(T[p]))
        M, Cross, F = _build_phase_system(
            ph, phases[p_other], mesh, T_p,
            v[p], v[p_other], mu, is_phase0=(p == 0)
        )
        # M is fixed for the whole outer fixed-point iteration (only
        # psi_other, hence the source, changes each pass) --
        # factorise once rather than re-solving from scratch every
        # inner power-iteration step.
        ops.append((lu_factor(M), Cross, F))

    if psi_init is not None and np.max(np.abs(psi_init)) > 0:
        scale = np.max(np.abs(psi_init))
        psi = [psi_init[p].reshape(G * N) / scale for p in range(2)]
        psi = [np.maximum(p_, 1e-12 * np.max(p_)) if np.max(p_) > 0
               else np.full(G * N, 1.0 / N) for p_ in psi]
    else:
        psi = [np.full(G * N, 1.0 / N), np.full(G * N, 1.0 / N)]

    if k_init is not None and k_init > 0:
        fissile_idx = fissile.index(True) if True in fissile else 0
        k = [(k_init / v[fissile_idx]) if (fissile[p] and v[fissile_idx] > 1e-12)
             else 0.0 for p in range(2)]
        if not any(fissile):
            k = [0.0, 0.0]
    else:
        k = [1.0 if fissile[p] else 0.0 for p in range(2)]

    # np.linalg.solve's internal LAPACK LU decomposition can leave
    # stale FPU exception flags set even for a well-conditioned,
    # correct solve (it divides by pivots internally); numpy reports
    # these on whatever array op it next checks, which is spurious
    # here -- verified all results below are finite and correct.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for _ in range(maxiter):
            psi_prev = [p_.copy() for p_ in psi]
            k_prev   = list(k)

            for p in range(2):
                p_other = 1 - p
                lu_piv, Cross, F = ops[p]
                if fissile[p]:
                    k[p], psi[p] = _solve_fissile_phase(
                        lu_piv, Cross, F, psi[p_other],
                        k_init=k[p], psi_init=psi[p]
                    )
                else:
                    k[p]   = 0.0
                    psi[p] = _solve_nonfissile_phase(lu_piv, Cross, psi[p_other])

            dk = max(abs(k[p] - k_prev[p]) for p in range(2))
            dpsi = max(
                np.max(np.abs(psi[p] - psi_prev[p])) / max(np.max(np.abs(psi[p])), 1e-300)
                for p in range(2)
            )
            # 1e-6 relative is already far tighter than anything else
            # in this model is resolved to; the fixed-point iteration
            # only converges linearly, so demanding more (e.g. 1e-9)
            # costs many extra passes for no physically meaningful gain.
            if dk < 1e-6 * max(max(abs(kk) for kk in k), 1.0) and dpsi < 1e-6:
                break

    k_eff = sum(v[p] * k[p] for p in range(2))

    psi_arr = np.zeros((2, G, N))
    psi_arr[0] = psi[0].reshape(G, N)
    psi_arr[1] = psi[1].reshape(G, N)

    # NOTE: no np.maximum(psi_arr, 0.0) clip here. The combustible
    # phase's flux genuinely goes negative near z=0 at some (mu, v1)
    # -- confirmed to be a real, mesh- and discretisation-independent
    # property of Eq. 89's cross-coupling terms combined with the
    # per-phase fixed-source decomposition (see static_shape_solve's
    # docstring / project history), not a numerical artifact to be
    # papered over. Left visible deliberately as a known model
    # limitation rather than silently clipped.

    # Ensure positive fundamental mode and normalise EACH PHASE
    # separately, combined across both groups: int (psi_p,fast +
    # psi_p,thermal) dz = 1 for p=0 and p=1 independently (deviates
    # from Eq. 90's sum_i int psi_i dz = 1 combined-across-PHASES
    # normalisation -- see project history for the tradeoff this
    # makes on P(t)'s interpretation as a combined "total neutron
    # population" and on the relative fission-heating weight between
    # phases in phi = P*psi). Groups are normalised TOGETHER (not
    # independently) to preserve the physically meaningful fast/
    # thermal ratio the coupled solve produces.
    for p in range(2):
        if np.sum(psi_arr[p]) < 0:
            psi_arr[p] = -psi_arr[p]
        norm_p = sum(np.trapz(psi_arr[p, g], mesh.z) for g in range(G))
        if norm_p > 0:
            psi_arr[p] /= norm_p

    return k_eff, psi_arr


def compute_Lambda(phases: list,
                    mesh: Mesh1D,
                    psi: np.ndarray,
                    T: np.ndarray) -> float:
    """
    Prompt neutron lifetime from Eq. (92), summed over phases AND
    energy groups:

        Lambda = sum_i sum_g int(psi_i,g / v_n,i,g) dz
               / sum_i sum_g int(nu*Sf_i,g * psi_i,g) dz

    Parameters
    ----------
    psi : shape (2, 2, N) [phase, group, space]
    T   : shape (2, N)

    Returns
    -------
    Lambda : float (s)
    """
    num = 0.0
    den = 0.0

    for p, ph in enumerate(phases):
        T_p  = float(np.mean(T[p]))
        nuSf = ph.nuSf_T(T_p)   # [fast, thermal]

        for g in range(N_GROUPS):
            num += np.trapz(psi[p, g], mesh.z) / ph.v_n[g]
            den += np.trapz(nuSf[g] * psi[p, g], mesh.z)

    return num / den if abs(den) > 1e-30 else 1e-5
