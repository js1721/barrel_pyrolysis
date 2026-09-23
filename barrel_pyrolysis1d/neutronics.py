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
              + 2*v_k*D_i,g * mu * d(psi_i,g)/dz
              - v_k*mu*(D_i,g + D_k,g) * d(psi_k,g)/dz
              + v_k*mu^2*(D_i,g*v_k + D_k,g*v_i)
                        * (psi_i,g - psi_k,g)

Expanding for two phases (same structure as the single-group model,
just carried through per-group):

    Phase 1, group g:
        L psi_1,g = D_1,g d^2(psi_1,g)/dz^2
                  + 2*v_2*D_1,g*mu*d(psi_1,g)/dz
                  - v_2*mu*(D_1,g + D_2,g)*d(psi_2,g)/dz
                  + v_2*mu^2*(D_1,g*v_2 + D_2,g*v_1)
                            * (psi_1,g - psi_2,g)

    Phase 2, group g:
        L psi_2,g = D_2,g d^2(psi_2,g)/dz^2
                  + 2*v_1*D_2,g*mu*d(psi_2,g)/dz
                  - v_1*mu*(D_2,g + D_1,g)*d(psi_1,g)/dz
                  + v_1*mu^2*(D_2,g*v_1 + D_1,g*v_2)
                            * (psi_2,g - psi_1,g)

Note: the self-drift (2*v_k*D_i*mu*dpsi_i/dz) and cross-gradient
(v_k*mu*(D_i+D_k)*dpsi_k/dz) terms are odd-order (first-
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
import warnings
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
                                 is_phase0: bool,
                                 closure: str = "relaxational") -> tuple:
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
    N  = mesh.N
    dz = mesh.dz
    mu = np.broadcast_to(np.atleast_1d(mu), (N,)).astype(float)

    if closure not in ("relaxational", "document"):
        raise ValueError(f"unknown closure {closure!r}")

    # Extrapolation distance for the vacuum boundary: the Milne-problem
    # value 0.7104 * lambda_tr with lambda_tr = 3 D, i.e. 2.131 D.
    # (The superseded 0.7104 / Sigma_removal used the REMOVAL cross
    # section in place of the transport one. For weak absorbers that
    # overstates d_ext enormously -- 75.7 cm instead of 0.28 cm for the
    # combustible thermal group, on a 40 cm slab -- making the "vacuum"
    # boundary nearly reflecting and inflating k.)
    d_ext = 2.1312 * D

    # ── Operator (document's active form, k = other phase) ─────────
    #   L psi_i = D_i psi_i''
    #           + v_k D_i d/dz[mu (psi_i - psi_k)]       (conservative)
    #           + v_k D_i mu psi_i'                      (self drift)
    #           - v_k D_k mu psi_k'                      (cross drift)
    #           + v_k (D_i v_k + D_k v_i) mu^2 (psi_i - psi_k)  (algebraic)
    #
    # A_self and Cross are LOSS operators (-L), matching the removal
    # term's sign: the phase system reads
    #     A_self psi_i + Cross psi_k = (fission source).
    #
    # Finite-volume assembly. The first two terms are -d/dz of the
    # TOTAL phase current
    #     J_i = -D_i psi_i' - v_k D_i mu (psi_i - psi_k),
    # assembled as face currents, so d(mu)/dz is retained when mu
    # varies (percolation) and the vacuum BC can be imposed on the
    # total current, as the physics requires. Central differencing
    # (a static eigenproblem has no stability constraint on the drift).
    #
    # CORRECTIONS relative to the superseded builder:
    #  * algebraic coupling sign: it implemented -G(psi_i - psi_k);
    #    the document has +G. Note +G is anti-diffusive -- it REDUCES
    #    the loss-operator diagonal, so at large mu the loss operator
    #    can become indefinite. That is a property of the closure.
    #  * top-boundary self drift: its one-sided stencil carried the
    #    opposite sign to the interior (-2 v_k D mu psi' at z=H).
    #  * the pointwise drift dropped the d(mu)/dz term.
    #  * the separate "stoch_bc" boundary correction is gone: with the
    #    stochastic current assembled on interior faces and the vacuum
    #    condition imposed on the TOTAL boundary current, it is not
    #    needed (and would double-count).
    #  is_phase0 is no longer needed: the phase-dependent sign combines
    #  with (psi_1 - psi_2) into (psi_i - psi_k), one form for both.
    G = mu**2 * v_oth * (D * v_oth + D_oth * v_self)       # (N,)
    c = v_oth * D

    # CLOSURE SWITCH.
    #  "document":     the operator written above (the document's active
    #                  slab equation): odd-order drift/conservative terms
    #                  and the algebraic coupling with the anti-diffusive
    #                  sign +G (psi_i - psi_k).
    #  "relaxational": no odd-order terms, algebraic coupling with the
    #                  relaxational sign -G (psi_i - psi_k).
    # The realisation-averaged benchmark (benchmark_markov.py) shows the
    # document form gives a negative combustible-phase flux and loses
    # its fundamental mode at moderate mu, while the relaxational form
    # stays positive. It is benchmarked, NOT derived: it keeps the
    # document's mu^2 exchange coefficient with the sign that the
    # benchmark requires. With the joint eigenvalue and the Marshak BC
    # it matches the benchmark k to ~1% at mu = 0.12 cm^-1 but is ~6%
    # LOW (non-conservative) at mu = 0.5-2 cm^-1: the mu^2 exchange
    # over-couples the phases at fine mixing.
    relax = (closure == "relaxational")
    if relax:
        c = 0.0

    A_self = np.zeros((N, N))
    Cross  = np.zeros((N, N))

    def add_face_current(f, coeff_i, coeff_k):
        """Face f lies between cells f-1 and f. coeff_* map cell index ->
        coefficient of the +z face current J_f. Loss in cell j is
        (J_{j+1} - J_j)/dz."""
        for cell, w in coeff_i.items():
            if f - 1 >= 0: A_self[f - 1, cell] += w / dz
            if f <= N - 1: A_self[f, cell]     -= w / dz
        for cell, w in coeff_k.items():
            if f - 1 >= 0: Cross[f - 1, cell] += w / dz
            if f <= N - 1: Cross[f, cell]     -= w / dz

    # Interior faces: diffusive + stochastic current, central average
    for f in range(1, N):
        jm, jp = f - 1, f
        ci = {jm: D / dz - 0.5 * c * mu[jm],
              jp: -D / dz - 0.5 * c * mu[jp]}
        ck = {jm: 0.5 * c * mu[jm],
              jp: 0.5 * c * mu[jp]}
        add_face_current(f, ci, ck)

    # Boundary faces: vacuum condition on the TOTAL outward current,
    #   J_out = (D_i / d_ext) psi_i(boundary cell)
    add_face_current(0, {0: -D / d_ext}, {})          # +z current at z=0
    add_face_current(N, {N - 1: D / d_ext}, {})       # +z current at z=H

    # Removal
    for n in range(N):
        A_self[n, n] += Sigma_removal

    # Volumetric terms (enter the loss with a minus sign)
    for n in range(N):
        if n == 0:
            dm, dp, h = n, n + 1, dz
        elif n == N - 1:
            dm, dp, h = n - 1, n, dz
        else:
            dm, dp, h = n - 1, n + 1, 2.0 * dz
        if not relax:
            # - v_k D_i mu psi_i'
            A_self[n, dp] -= c * mu[n] / h
            A_self[n, dm] += c * mu[n] / h
            # + v_k D_k mu psi_k'   (negated cross drift)
            Cross[n, dp] += v_oth * D_oth * mu[n] / h
            Cross[n, dm] -= v_oth * D_oth * mu[n] / h
            # - G (psi_i - psi_k)      [document: anti-diffusive]
            A_self[n, n] -= G[n]
            Cross[n, n]  += G[n]
        else:
            # + G (psi_i - psi_k) in the loss  [relaxational]
            A_self[n, n] += G[n]
            Cross[n, n]  -= G[n]

    return A_self, Cross


def neutron_temperature(phases: list, T: np.ndarray) -> float:
    """
    Temperature of the thermal neutron spectrum: the mean temperature of
    the MODERATING phase, identified as the phase with the larger
    fast->thermal downscatter cross section. See materials.py for why
    the thermal-group 1/v factor of every phase uses this, not the
    phase's own temperature.
    """
    m = int(np.argmax([ph.Sigma_s12 for ph in phases]))
    return float(np.mean(T[m]))


def effective_D(phases: list, v: np.ndarray, mu) -> list:
    """
    Scale-dependent effective diffusion coefficients D_i*, per phase and
    group, for the conditional-average (two-phase) operator.

    WHY. In a layered medium both the flux and the CURRENT are continuous
    at an interface, so in the fine-mixing limit the conditional currents
    are equal -- they are not -D_i grad(psi_i) with the bulk D_i. A
    two-equation model without cross-diffusion terms therefore mixes D
    arithmetically: summing the volume-weighted phase equations at
    psi_1 = psi_2 leaves sum_i v_i D_i. The correct homogeneous limit is
    the ATOMIC MIX, in which Sigma_tr adds linearly and D combines
    HARMONICALLY, D_harm = 1/(v_1/D_1 + v_2/D_2). With D differing ~6x
    between these phases the model otherwise leaks far too much and
    under-predicts k by ~6% -- non-conservatively (benchmark_markov.py).

    The two limits are fixed by the chunk size relative to the diffusion
    length L_i,g = sqrt(D_i,g / Sigma_rem,i,g), with mean chord
    lambda_i = 1/(mu v_k):

        chunks >> L : the flux diffuses inside a chunk with its own D_i
        chunks << L : the medium acts as an atomic mix, D -> D_harm

    interpolated with w = 1/(1 + (lambda_i/L_i)^2),

        D_i* = (1 - w) D_i + w D_harm,

    which has NO fitted parameter. Against the realisation benchmark this
    reduces the error in k from -1.1/-6.5/-6.1% to -0.2/-2.1/+1.0% at
    mu = 0.12/0.5/2.0 cm^-1. The residual is not a D effect: the
    correction the benchmark demands is non-monotonic in lambda/L, which
    points to flux depression inside fuel chunks, largest where
    chunk ~ diffusion length, and outside the reach of a model carrying
    one flux per phase.

    A field mu (percolation) is reduced to its mean here, since the
    operator takes one D per phase and group.
    """
    mu_s = float(np.mean(np.atleast_1d(mu)))
    D = [np.asarray(ph.D, dtype=float) for ph in phases]
    if mu_s <= 0.0:
        return D
    D_harm = 1.0 / (v[0] / D[0] + v[1] / D[1])
    out = []
    for p, ph in enumerate(phases):
        Srem = np.array([ph.Sigma_a[0] + ph.Sigma_s12, ph.Sigma_a[1]])
        L = np.sqrt(D[p] / np.maximum(Srem, 1e-30))
        lam = 1.0 / (mu_s * max(v[1 - p], 1e-30))
        w = 1.0 / (1.0 + (lam / L)**2)
        out.append((1.0 - w) * D[p] + w * D_harm)
    return out


def _build_phase_system(ph: Phase,
                         ph_oth: Phase,
                         mesh: Mesh1D,
                         T_p: float,
                         v_self: float,
                         v_oth: float,
                         mu,
                         is_phase0: bool,
                         closure: str = "relaxational",
                         T_n: float = None) -> tuple:
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
    Sa   = ph.Sa_T(T_p, T_n)     # [fast, thermal]
    nuSf = ph.nuSf_T(T_p, T_n)   # [fast, thermal]

    A_fast, Cross_fast = _build_group_self_and_cross(
        ph.D[0], ph_oth.D[0], Sa[0] + ph.Sigma_s12,
        mesh, v_self, v_oth, mu, is_phase0, closure)
    A_thermal, Cross_thermal = _build_group_self_and_cross(
        ph.D[1], ph_oth.D[1], Sa[1],
        mesh, v_self, v_oth, mu, is_phase0, closure)

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



class ClosureBreakdownError(RuntimeError):
    """Raised when the stochastic closure has no fundamental mode."""


def closure_margin(phases: list, T: np.ndarray, v: np.ndarray, mu) -> list:
    """
    For each phase and group, the ratio

        G_max / Sigma_removal,   G = v_k (D_i v_k + D_k v_i) mu^2

    where G is the algebraic stochastic coupling. G enters the loss
    operator with a NEGATIVE sign (it is anti-diffusive), so once it
    reaches the removal cross section the loss operator can no longer
    be guaranteed positive and the fundamental mode may cease to exist.
    Ratios approaching 1 mean the closure is near breakdown -- a
    property of the model, not of the discretisation.

    Returns a list of (phase, group, ratio) tuples.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    out = []
    for p, ph in enumerate(phases):
        k = 1 - p
        po = phases[k]
        Sa = ph.Sa_T(float(np.mean(T[p])), neutron_temperature(phases, T))
        for g in range(N_GROUPS):
            Sr = Sa[g] + (ph.Sigma_s12 if g == 0 else 0.0)
            Gmax = float(np.max(mu**2)) * v[k] * (ph.D[g] * v[k] + po.D[g] * v[p])
            out.append((p, g, Gmax / Sr if Sr > 0 else np.inf))
    return out


def _check_closure(phases, T, v, mu, k_eff, psi_arr, margin_limit=1.0,
                   closure="document"):
    """Raise ClosureBreakdownError rather than return garbage. The
    coupling-vs-removal margin applies only to the document closure,
    whose algebraic term subtracts from the loss diagonal; the
    relaxational term adds to it."""
    mx = float(np.max(np.atleast_1d(mu)))
    for p, g, ratio in (closure_margin(phases, T, v, mu)
                        if closure == "document" else []):
        if ratio >= margin_limit:
            raise ClosureBreakdownError(
                f"stochastic coupling exceeds removal in phase {p+1}, "
                f"group {g} (G/Sigma_r = {ratio:.3f}) at max mu = {mx:.4g} "
                f"cm^-1: the loss operator is no longer positive and the "
                f"model has no reliable fundamental mode.")
    if not np.isfinite(k_eff) or k_eff <= 0.0:
        raise ClosureBreakdownError(
            f"eigenvalue solve returned k_eff = {k_eff!r} at max mu = "
            f"{mx:.4g} cm^-1: the operator is indefinite or singular "
            f"for this configuration.")
    for p in range(len(phases)):
        if not np.all(np.isfinite(psi_arr[p])):
            raise ClosureBreakdownError(
                f"non-finite flux in phase {p+1} at max mu = {mx:.4g} cm^-1.")


def static_shape_solve(phases: list,
                         mesh: Mesh1D,
                         T: np.ndarray,
                         v: np.ndarray,
                         mu,
                         psi_init: np.ndarray = None,
                         k_init: float = None,
                         maxiter: int = 2000,
                         closure: str = "relaxational",
                         effective_D_on: bool = True) -> tuple:
    """
    Fundamental mode of the COUPLED two-phase system, one eigenvalue:

        [ M_1  C_1 ] [psi_1]   1  [ F_1  0  ] [psi_1]
        [ C_2  M_2 ] [psi_2] = -  [ 0   F_2 ] [psi_2]
                               k

    where M_i, C_i, F_i are phase i's conditional-average loss, coupling
    and fission operators (both energy groups).

    WHY ONE EIGENVALUE. The superseded formulation solved a separate
    eigenproblem per phase and reported k_eff = v_1 k_1 + v_2 k_2. That
    is exact only for infinitely coarse mixing, where each realisation is
    all one phase (a fraction v_1 of them pure fuel, the rest pure
    moderator). At any finite chunk size moderator interleaved with fuel
    raises k, which v_1 k_1 cannot see: against the realisation-averaged
    benchmark it put criticality at v_1 = 0.67 where the random medium
    is critical near 0.13 -- a large NON-conservative error. The joint
    eigenvalue with the relaxational closure removes most of it, but a
    bias remains: with the Marshak BC the model k is ~1% low at
    mu = 0.12 cm^-1 and ~6% low at mu = 0.5-2 cm^-1 relative to the
    benchmark (benchmark_markov.py). In the coarse limit (mu -> 0) the
    joint k tends to the fuel phase's own eigenvalue -- the k of a drum
    that CONTAINS fuel, near the benchmark median -- rather than the
    ensemble mean, which also averages in drums with no fuel.

    NORMALISATION. psi_1 is scaled to unit integrated flux (summed over
    groups), exactly as before, so the fission heating P*psi keeps its
    meaning. psi_2 is scaled by the SAME factor, so the ratio
    psi_2/psi_1 is physical (the old per-phase normalisation discarded
    it). The eigenvector's overall sign is fixed jointly; the phases are
    never flipped independently. A fundamental mode with a sign change
    in either phase triggers a warning -- expected with
    closure="document", which produces a negative combustible flux.

    Returns
    -------
    k_eff : float
    psi   : (2, N_GROUPS, N) [phase, group, space]
    """
    N = mesh.N
    G = N_GROUPS
    n = G * N
    import dataclasses
    Ds = effective_D(phases, v, mu) if effective_D_on else [ph.D for ph in phases]
    phases = [dataclasses.replace(ph, D=np.asarray(Ds[i], dtype=float))
              for i, ph in enumerate(phases)]
    Ms, Cs, Fs = [], [], []
    for p, ph in enumerate(phases):
        q = 1 - p
        M, C, F = _build_phase_system(ph, phases[q], mesh,
                                      float(np.mean(T[p])), v[p], v[q],
                                      mu, p == 0, closure,
                                      T_n=neutron_temperature(phases, T))
        Ms.append(M); Cs.append(C); Fs.append(F)
    A = np.block([[Ms[0], Cs[0]], [Cs[1], Ms[1]]])
    Z = np.zeros((n, n))
    Fj = np.block([[Fs[0], Z], [Z, Fs[1]]])

    lu_piv = lu_factor(A)
    if psi_init is not None and np.all(np.isfinite(psi_init)):
        x0 = np.asarray(psi_init, dtype=float).reshape(-1).copy()
    else:
        x0 = np.ones(2 * n)

    # Arnoldi (ARPACK) on A^-1 F rather than plain power iteration: in
    # large or loosely coupled systems the fundamental and first
    # harmonic are nearly degenerate (dominance ratio -> 1) and power
    # iteration crawls. The warm start x0 is used in transients.
    from scipy.sparse.linalg import eigs, LinearOperator, ArpackNoConvergence
    op = LinearOperator((2 * n, 2 * n), dtype=float,
                        matvec=lambda y: lu_solve(lu_piv, Fj @ y))
    converged = True
    try:
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            vals, vecs = eigs(op, k=1, which="LM", v0=x0, tol=1e-12,
                              maxiter=max(maxiter, 1000))
    except ArpackNoConvergence:
        converged = False
        vals, vecs = np.array([np.nan + 0j]), np.full((2 * n, 1), np.nan)
    lam = vals[0]
    if converged and abs(lam.imag) > 1e-9 * max(abs(lam.real), 1.0):
        raise ClosureBreakdownError(
            f"dominant eigenvalue is complex ({lam!r}) (closure={closure!r}, "
            f"max mu = {float(np.max(np.atleast_1d(mu))):.4g} cm^-1): "
            f"no physical fundamental mode.")
    k_eff = float(lam.real)
    x = vecs[:, 0].real

    psi = x.reshape(2, G, N)
    if psi[0].sum() < 0:                  # one joint sign, never per phase
        psi = -psi
    s1 = sum(np.sum(psi[0, g]) * mesh.dz for g in range(G))
    if s1 > 0:
        psi = psi / s1
    for p in range(2):
        if v[p] > 0 and np.any(psi[p] < -1e-8 * np.abs(psi).max()):
            warnings.warn(
                f"fundamental mode changes sign in phase {p+1} "
                f"(closure={closure!r}, max mu = "
                f"{float(np.max(np.atleast_1d(mu))):.4g} cm^-1): the "
                f"conditional flux is not physical.", RuntimeWarning,
                stacklevel=2)

    _check_closure(phases, T, v, mu, k_eff, psi, closure=closure)
    if not converged:
        raise ClosureBreakdownError(
            f"coupled eigenvalue iteration did not converge in {maxiter} "
            f"iterations (closure={closure!r}, max mu = "
            f"{float(np.max(np.atleast_1d(mu))):.4g} cm^-1, last k_eff = "
            f"{k_eff:.6g}).")
    return k_eff, psi

def source_amplitude_rate(phases: list, mesh: Mesh1D, psi: np.ndarray,
                          v: np.ndarray, source_density: float) -> float:
    """
    External source expressed in amplitude units per second, the q in
    kinetics.advance_kinetics. With phi = P psi (phi in n/cm^2/s) and
    the same unit weight function as compute_Lambda,

        q = S_tot / N_w,
        S_tot = sum_p v_p * int Q_p dz         (source neutrons / cm^2 / s)
        N_w   = sum_p v_p sum_g int psi_p,g / v_n,p,g dz,

    so that Lambda = N_w / F_w and q enter the point-kinetics balance
    consistently. Q is source_density (n/s/cm^3 of fuel phase) in the
    fissile phase(s), zero elsewhere; the unit weight counts source
    neutrons regardless of the group they are born in.
    """
    if not source_density:
        return 0.0
    S_tot = sum(v[p] * source_density * mesh.H
                for p, ph in enumerate(phases) if np.any(ph.nu_Sf > 0))
    N_w = sum(v[p] * np.sum(psi[p, g]) * mesh.dz / ph.v_n[g]
              for p, ph in enumerate(phases) for g in range(N_GROUPS))
    return S_tot / N_w if N_w > 0 else 0.0


# Integrals over the slab are finite-volume CELL SUMS, sum(f) * dz,
# consistent with the finite-volume discretisation and with the source
# integral (which uses the full height H). np.trapezoid over cell centres
# omits the half-cells at both ends -- an O(dz) inconsistency (~1e-3 in
# Lambda and the source-driven flux at N = 80) exposed by the 2D
# reduction test.
def compute_Lambda(phases: list,
                    mesh: Mesh1D,
                    psi: np.ndarray,
                    T: np.ndarray,
                    v: np.ndarray = None) -> float:
    """
    Prompt neutron generation time, summed over phases and groups with
    each phase weighted by its volume fraction (the ensemble average of
    a conditional quantity is sum_p v_p * <.>_p):

        Lambda = sum_p v_p sum_g int(psi_p,g / v_n,p,g) dz
               / sum_p v_p sum_g int(nu*Sf_p,g * psi_p,g) dz

    With psi from the joint eigenproblem the phases carry their physical
    relative amplitudes, so the v_p weights are required; the superseded
    unweighted sum was only harmless while each phase was normalised
    separately. v=None reproduces the unweighted sum.

    psi : (2, 2, N) [phase, group, space];  T : (2, N)
    """
    w = np.ones(len(phases)) if v is None else np.asarray(v, dtype=float)
    num = 0.0
    den = 0.0
    for p, ph in enumerate(phases):
        nuSf = ph.nuSf_T(float(np.mean(T[p])), neutron_temperature(phases, T))
        for g in range(N_GROUPS):
            num += w[p] * np.sum(psi[p, g]) * mesh.dz / ph.v_n[g]
            den += w[p] * np.sum(nuSf[g] * psi[p, g]) * mesh.dz
    return num / den if abs(den) > 1e-30 else 1e-5
