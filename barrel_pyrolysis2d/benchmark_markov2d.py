"""
benchmark_markov2d.py
======================
Realisation-ensemble safety check for the 2D (r,z) drum -- the 2D
counterpart of barrel_pyrolysis1d/benchmark_markov.py's neutronics
benchmark, wired into a P(k_eff > 1) report the same way
barrel_pyrolysis1d/main.py now reports one.

WHY THIS IS APPROXIMATE (read before trusting the numbers)
------------------------------------------------------------
barrel_pyrolysis1d's benchmark samples an EXACT realisation of the
medium: a stationary alternating Markov process along z, with the same
(mu, v1) the closed-form model uses, so the two are directly
comparable -- no extra modelling choice is smuggled in.

There is no known closed-form isotropic 2D generalisation of that
process for v1 != 1/2 (the natural isotropic constructions -- e.g. a
Poisson line tessellation -- either force v1 = 1/2 by symmetry, or, if
cells are coloured independently, do not reproduce the SAME chord-length
statistics as the 1D process along any single line). barrel_pyrolysis2d's
own neutronics.py already flags this as open ("2D exchange rate is
unvalidated... a 2D (r-z) realisation benchmark is the way to settle
it") -- this file is a first, explicitly partial, attempt at that
benchmark, not the settled answer.

What's implemented here: each radial ring (fixed r) gets its own
INDEPENDENT realisation of the exact 1D alternating-Markov process along
z, using the same (mu, v1) and the same sampler as the 1D benchmark. So:
  * along z, at any fixed r, the statistics are EXACTLY the validated 1D
    process;
  * v1 is matched exactly (every ring is drawn from the same v1);
  * there is NO radial correlation at all -- adjacent rings are drawn
    independently, so this medium contains no radially-elongated fuel
    chunks, only axially-elongated ones.

Net effect: this is likely a LOWER bound on the true clumping risk, not
an upper one. The model's own stated justification for stochastic mixing
is that "large chunks of fissile material lumped together is where the
risk of criticality is highest" (paper/secondary.tex) -- a fully
isotropic medium would allow chunks elongated in ANY direction,
including radially, which this construction cannot produce. Treat the
P(k>1) below as a floor, not a ceiling.
"""

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu, eigs, LinearOperator, ArpackNoConvergence

from mesh import CylindricalMesh2D
from materials import make_PuO, make_combustible


def chord_lengths(mu: float, v1: float) -> tuple:
    """(lambda_1, lambda_2) in cm, same convention as the 1D benchmark."""
    return 1.0 / (mu * (1.0 - v1)), 1.0 / (mu * v1)


def _sample_line(z: np.ndarray, H: float, lam1: float, lam2: float,
                 v1: float, rng) -> np.ndarray:
    """Phase index (0/1) at each z cell centre for one 1D realisation --
    identical to barrel_pyrolysis1d/benchmark_markov.py's sample_phase_map."""
    phase = 0 if rng.random() < v1 else 1
    edges, phases = [0.0], []
    zc = 0.0
    while zc < H:
        zc += rng.exponential(lam1 if phase == 0 else lam2)
        edges.append(min(zc, H))
        phases.append(phase)
        phase = 1 - phase
    idx = np.searchsorted(np.asarray(edges), z, side="right") - 1
    return np.asarray(phases)[np.clip(idx, 0, len(phases) - 1)]


def sample_phase_map_2d(mesh: CylindricalMesh2D, mu: float, v1: float,
                        rng) -> np.ndarray:
    """Phase index (0/1), shape (Nr, Nz): each ring independently sampled
    -- see module docstring for what this does and doesn't capture."""
    lam1, lam2 = chord_lengths(mu, v1)
    pm = np.empty((mesh.Nr, mesh.Nz), dtype=int)
    for i in range(mesh.Nr):
        pm[i, :] = _sample_line(mesh.z, mesh.H, lam1, lam2, v1, rng)
    return pm


def _group_loss_realisation(D_cell: np.ndarray, Srem_cell: np.ndarray,
                            mesh: CylindricalMesh2D) -> sps.csr_matrix:
    """
    Loss operator for one group of ONE realisation: per-cell D and
    removal cross section (no stochastic coupling term -- the medium is
    already fully resolved cell by cell). Interior faces use the
    harmonic mean of the two cells' D (exact for a material interface
    on the face, matching barrel_pyrolysis1d/benchmark_markov.py's
    realisation_eigen). Boundary faces use the LOCAL boundary cell's own
    D for the Marshak vacuum condition -- same geometry as
    neutronics.py's _group_loss, specialised to a single phase per cell.
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    dr, dz = mesh.dr, mesh.dz
    r, re = mesh.r, mesh.r_edge
    rows, cols, vals = [], [], []

    def add(a, b, x):
        rows.append(a); cols.append(b); vals.append(x)

    def hmean(a, b):
        return 2.0 * a * b / (a + b)

    for i in range(Nr):
        for j in range(Nz):
            n = mesh.idx(i, j)
            Di = D_cell[i, j]
            d_ext = 2.1312 * Di
            diag = Srem_cell[i, j]
            if i > 0:
                Dh = hmean(Di, D_cell[i - 1, j])
                c = Dh * re[i] / (r[i] * dr * dr)
                diag += c
                add(n, mesh.idx(i - 1, j), -c)
            if i < Nr - 1:
                Dh = hmean(Di, D_cell[i + 1, j])
                c = Dh * re[i + 1] / (r[i] * dr * dr)
                diag += c
                add(n, mesh.idx(i + 1, j), -c)
            else:
                diag += Di * re[i + 1] / (r[i] * dr) / d_ext   # vacuum wall
            if j > 0:
                Dh = hmean(Di, D_cell[i, j - 1])
                diag += Dh / dz**2
                add(n, mesh.idx(i, j - 1), -Dh / dz**2)
            else:
                diag += Di / (dz * d_ext)
            if j < Nz - 1:
                Dh = hmean(Di, D_cell[i, j + 1])
                diag += Dh / dz**2
                add(n, mesh.idx(i, j + 1), -Dh / dz**2)
            else:
                diag += Di / (dz * d_ext)
            add(n, n, diag)
    return sps.csr_matrix((vals, (rows, cols)), shape=(mesh.N, mesh.N))


def realisation_eigen_2d(pm: np.ndarray, mesh: CylindricalMesh2D,
                         T: float = 300.0) -> tuple:
    """Fundamental (k, phi, negfrac) for one realisation. phi shape
    (2 groups, Nr, Nz). Returns (0.0, None, 0.0) if the realisation
    holds no fissile material (phase 0 = PuO absent everywhere)."""
    if not np.any(pm == 0):
        return 0.0, None, 0.0

    ph = [make_PuO(), make_combustible()]
    N = mesh.N
    D    = np.zeros((2, mesh.Nr, mesh.Nz))
    Sa   = np.zeros((2, mesh.Nr, mesh.Nz))
    nuSf = np.zeros((2, mesh.Nr, mesh.Nz))
    s12  = np.zeros((mesh.Nr, mesh.Nz))
    for p in (0, 1):
        mask = (pm == p)
        Sa_p, nuSf_p = ph[p].Sa_T(T, T), ph[p].nuSf_T(T, T)
        for g in range(2):
            D[g][mask]    = ph[p].D[g]
            Sa[g][mask]   = Sa_p[g]
            nuSf[g][mask] = nuSf_p[g]
        s12[mask] = ph[p].Sigma_s12

    Z = sps.csr_matrix((N, N))
    A_fast = _group_loss_realisation(D[0], Sa[0] + s12, mesh)
    A_th   = _group_loss_realisation(D[1], Sa[1],       mesh)
    A = sps.bmat([[A_fast, Z], [-sps.diags(s12.ravel()), A_th]], format="csc")
    F = sps.bmat([[sps.diags(nuSf[0].ravel()), sps.diags(nuSf[1].ravel())],
                 [Z, Z]], format="csr")           # chi = [1, 0]

    lu = splu(A)
    op = LinearOperator((2 * N, 2 * N), matvec=lambda x: lu.solve(F @ x),
                        dtype=float)
    try:
        vals, vecs = eigs(op, k=1, which="LM", tol=1e-10, maxiter=20000)
    except ArpackNoConvergence:
        return np.nan, None, 0.0
    k = float(vals[0].real)
    phi = vecs[:, 0].real
    if phi.sum() < 0:
        phi = -phi
    scale = np.abs(phi).max()
    negfrac = float(np.mean(phi < -1e-9 * scale)) if scale > 0 else 0.0
    return k, phi.reshape(2, mesh.Nr, mesh.Nz), negfrac


def neutronics_benchmark_2d(mu: float, v1: float, mesh: CylindricalMesh2D,
                            n_real: int = 300, seed: int = 2024) -> dict:
    rng = np.random.default_rng(seed)
    ks, negs = [], []
    n_nofuel = n_nan = 0
    for _ in range(n_real):
        pm = sample_phase_map_2d(mesh, mu, v1, rng)
        k, phi, neg = realisation_eigen_2d(pm, mesh)
        if phi is None:
            if np.isnan(k):
                n_nan += 1
            else:
                n_nofuel += 1
            continue
        ks.append(k)
        negs.append(neg)
    return dict(mu=mu, v1=v1, R=mesh.R, H=mesh.H, k=np.array(ks),
                nofuel=n_nofuel / n_real, nan=n_nan / n_real,
                max_negfrac=(max(negs) if negs else 0.0), n_real=n_real)


def safety_report_2d(mu: float, v1: float, mesh: CylindricalMesh2D,
                     n_real: int = 300, seed: int = 2024,
                     model_k: float = None) -> dict:
    """Same statistics as barrel_pyrolysis1d's safety_report, over the
    per-ring-independent 2D realisation ensemble above."""
    b = neutronics_benchmark_2d(mu, v1, mesh, n_real=n_real, seed=seed)
    k = b["k"]
    se = float(k.std() / np.sqrt(len(k))) if len(k) else float("nan")
    return dict(mu=mu, v1=v1, R=mesh.R, H=mesh.H, n_real=n_real,
                model_k=model_k, mean=float(k.mean()) if len(k) else float("nan"),
                median=float(np.median(k)) if len(k) else float("nan"),
                std=float(k.std()) if len(k) else float("nan"), se=se,
                p_gt1=float(np.mean(k > 1)) if len(k) else float("nan"),
                nofuel=b["nofuel"], nan=b["nan"], max_negfrac=b["max_negfrac"])


def print_safety_report_2d(rep: dict) -> None:
    print("\n" + "=" * 50)
    print("2D REALISATION-ENSEMBLE SAFETY CHECK (Monte Carlo)")
    print("=" * 50)
    print(f"  mu = {rep['mu']} cm^-1, v1 = {rep['v1']:.4f}, "
          f"R = {rep['R']} cm, H = {rep['H']} cm, {rep['n_real']} realisations")
    print("  NOTE: axially-exact, radially-INDEPENDENT rings -- likely a")
    print("  lower bound on true clumping risk. See module docstring.")
    if rep["model_k"] is not None:
        print(f"  closed-form model k_eff        : {rep['model_k']:.4f}")
    print(f"  realisation mean k_eff          : {rep['mean']:.4f} "
          f"+/- {rep['se']:.4f} (SE)")
    print(f"  realisation median k_eff        : {rep['median']:.4f}")
    print(f"  realisation std(k_eff)          : {rep['std']:.4f}")
    print(f"  P(k_eff > 1) over realisations  : {100 * rep['p_gt1']:.1f}%")
    if rep["nofuel"] > 0:
        print(f"  ({100 * rep['nofuel']:.1f}% of realisations contained no "
              f"fissile material at all)")
    if rep["nan"] > 0:
        print(f"  ({100 * rep['nan']:.1f}% of realisations failed to "
              f"converge to a fundamental mode)")
    if rep["max_negfrac"] > 1e-6:
        print(f"  (up to {100 * rep['max_negfrac']:.2f}% negative flux in "
              f"the worst realisation -- some individual realisations may")
        print(f"  contain isolated single-cell fuel specks with a poorly "
              f"resolved local mode; treat k for those with caution)")
    print("=" * 50)


if __name__ == "__main__":
    import time
    mesh = CylindricalMesh2D(R=28.6, H=88.0, Nr=22, Nz=64)
    t0 = time.time()
    rep = safety_report_2d(0.5, 0.15988287397030493, mesh, n_real=300,
                           model_k=0.98)
    print_safety_report_2d(rep)
    print(f"\n({time.time() - t0:.1f} s for {rep['n_real']} realisations)")
