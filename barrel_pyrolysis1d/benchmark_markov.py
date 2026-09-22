"""
benchmark_markov.py
===================
Closure-independent benchmark for the two-phase stochastic model.

Instead of closing the ensemble-averaged equations, sample many
REALISATIONS of the binary Markov mixture, solve the ordinary
deterministic problem on each (mesh much finer than the chord lengths),
and form the conditional averages directly:

    T_i(z,t) = < T(z,t) 1{phase i at z} > / < 1{phase i at z} >

These are the quantities the closed equations claim to predict, so any
closure can be tested against them without relying on how it was
derived.

The medium
----------
Stationary dichotomic Markov process on [0, H]: alternating segments,
phase i segments exponentially distributed with mean chord length
lambda_i. Its volume fraction and correlation decay rate are

    v_1 = lambda_1 / (lambda_1 + lambda_2),
    mu  = 1/lambda_1 + 1/lambda_2,

i.e. lambda_1 = 1/(mu v_2), lambda_2 = 1/(mu v_1) -- the same (mu, v)
the closed model uses. Exponential chords are memoryless, so starting
in phase i with probability v_i at z = 0 gives an exactly stationary
process with no burn-in.

Thermal test (this file, part 1)
--------------------------------
Isolated slab (insulating at both ends), no sources, each realisation
started at T = T1_0 in phase-1 material and T2_0 in phase-2 material.
The physics here is unambiguous: conduction equalises temperature, so
T_1 - T_2 must DECAY to zero. For these initial conditions the closed
model's fields stay uniform in z and reduce exactly to an ODE,

    d(T_1 - T_2)/dt = s * lam_c * (T_1 - T_2),
    lam_c = mu^2 (K_1 v_2 + K_2 v_1)(v_2/(rho_1 Cp_1) + v_1/(rho_2 Cp_2)),

with s = +1 for the document's closure (growth) and s = -1 for a
relaxational one. The benchmark decides the sign and measures the rate.
"""

import os
import numpy as np
from scipy.linalg import solve_banded

from materials import make_PuO, make_combustible

CM_TO_M = 1.0e-2
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ─────────────────────────────────────────────────────────────────────
# Medium sampling
# ─────────────────────────────────────────────────────────────────────
def chord_lengths(mu: float, v1: float) -> tuple:
    """(lambda_1, lambda_2) in cm for inverse correlation length mu
    (cm^-1) and phase-1 volume fraction v1."""
    return 1.0 / (mu * (1.0 - v1)), 1.0 / (mu * v1)


def sample_phase_map(z: np.ndarray, H: float, lam1: float, lam2: float,
                     v1: float, rng) -> np.ndarray:
    """Phase index (0 or 1) at each cell centre z for one realisation."""
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


# ─────────────────────────────────────────────────────────────────────
# Deterministic heterogeneous conduction
# ─────────────────────────────────────────────────────────────────────
def conduction_matrix(K: np.ndarray, rCp: np.ndarray, dz: float,
                      dt: float) -> np.ndarray:
    """Backward-Euler banded matrix for rho Cp dT/dt = d/dz(K dT/dz),
    insulating ends. Face conductivity is the harmonic mean, which is
    exact for a material interface located on the face."""
    N = len(K)
    Kf = 2.0 * K[:-1] * K[1:] / (K[:-1] + K[1:])       # interior faces
    a = dt / (rCp * dz**2)
    AB = np.zeros((3, N))
    AB[1, :] = 1.0
    AB[1, :-1] += a[:-1] * Kf
    AB[1, 1:] += a[1:] * Kf
    AB[0, 1:] = -a[:-1] * Kf          # row j, col j+1
    AB[2, :-1] = -a[1:] * Kf          # row j+1, col j
    return AB


def thermal_benchmark(mu: float, v1: float, H: float = 40.0,
                      dz: float = 0.02, n_real: int = 300,
                      t_end: float = None, n_steps: int = 400,
                      T10: float = 310.0, T20: float = 300.0,
                      seed: int = 12345) -> dict:
    """Realisation-averaged isolated-slab equilibration test."""
    ph = [make_PuO(), make_combustible()]
    K_ph = np.array([ph[0].K, ph[1].K])
    rCp_ph = np.array([ph[0].rho * ph[0].Cp, ph[1].rho * ph[1].Cp])
    v2 = 1.0 - v1
    mu_m = mu / CM_TO_M
    lam_c = mu_m**2 * (K_ph[0] * v2 + K_ph[1] * v1) * (v2 / rCp_ph[0] + v1 / rCp_ph[1])
    if t_end is None:
        t_end = 4.0 / lam_c
    dt = t_end / n_steps

    N = int(round(H / dz))
    z = (np.arange(N) + 0.5) * dz
    dz_m = dz * CM_TO_M
    lam1, lam2 = chord_lengths(mu, v1)
    rng = np.random.default_rng(seed)

    t = np.linspace(0.0, t_end, n_steps + 1)
    sumT = np.zeros((2, n_steps + 1))
    sumW = np.zeros(2)
    frac1 = []

    for _ in range(n_real):
        pm = sample_phase_map(z, H, lam1, lam2, v1, rng)
        K = K_ph[pm]
        rCp = rCp_ph[pm]
        AB = conduction_matrix(K, rCp, dz_m, dt)
        T = np.where(pm == 0, T10, T20).astype(float)
        m1 = (pm == 0)
        sumW += [m1.sum(), (~m1).sum()]
        frac1.append(m1.mean())
        sumT[0, 0] += T[m1].sum()
        sumT[1, 0] += T[~m1].sum()
        for n in range(1, n_steps + 1):
            T = solve_banded((1, 1), AB, T)
            sumT[0, n] += T[m1].sum()
            sumT[1, n] += T[~m1].sum()

    Tbar = sumT / sumW[:, None]
    return dict(mu=mu, v1=v1, lam1=lam1, lam2=lam2, lam_c=lam_c, t=t,
                T1=Tbar[0], T2=Tbar[1], dT=Tbar[0] - Tbar[1],
                v1_sampled=float(np.mean(frac1)), n_real=n_real, dz=dz)


def report_thermal(res: dict) -> None:
    t, dT = res["t"], res["dT"]
    r = dT / dT[0]
    lc = res["lam_c"]
    # effective decay rate from the early-time log slope (first 5% of run)
    k5 = max(2, len(t) // 20)
    lam_eff = -np.polyfit(t[:k5], np.log(r[:k5]), 1)[0]
    print(f"\nmu = {res['mu']} cm^-1   (lambda_1 = {res['lam1']:.3g} cm, "
          f"lambda_2 = {res['lam2']:.3g} cm, v1 sampled = {res['v1_sampled']:.4f} "
          f"vs {res['v1']:.4f}; {res['n_real']} realisations, dz = {res['dz']} cm)")
    print(f"  closure rate lam_c = {lc:.4g} 1/s   (1/lam_c = {1/lc:.3g} s)")
    print(f"  {'t (s)':>8}  {'benchmark':>10}  {'document e^+':>12}  {'relax e^-':>10}")
    for frac in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
        n = int(round(frac * (len(t) - 1)))
        print(f"  {t[n]:8.2f}  {r[n]:10.4f}  {np.exp(lc*t[n]):12.4f}  {np.exp(-lc*t[n]):10.4f}")
    print(f"  benchmark: monotone decay = {bool(np.all(np.diff(r) <= 1e-12))},  "
          f"early-time rate = {lam_eff:.4g} 1/s  = {lam_eff/lc:.3f} x lam_c")


def plot_thermal(results: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.8),
                             sharey=True)
    for ax, res in zip(np.atleast_1d(axes), results):
        t, r, lc = res["t"], res["dT"] / res["dT"][0], res["lam_c"]
        ax.plot(t, r, "k-", lw=2, label="realisation benchmark")
        ax.plot(t, np.exp(-lc * t), "C0--", label="relaxational, rate $\\lambda_c$")
        ax.plot(t, np.exp(np.minimum(lc * t, np.log(3))), "C3:",
                label="document closure (growth)")
        ax.set(xlabel="t (s)", title=f"$\\mu$ = {res['mu']} cm$^{{-1}}$", ylim=(0, 3))
    np.atleast_1d(axes)[0].set_ylabel(r"$(T_1-T_2)/(T_1-T_2)_0$")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "fig_benchmark_thermal.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved: {path}")



# ─────────────────────────────────────────────────────────────────────
# Part 2: two-group neutronics
# ─────────────────────────────────────────────────────────────────────
#
# Each realisation is an ordinary heterogeneous two-group diffusion
# eigenproblem  M phi = (1/k) F phi  with piecewise-constant data:
#   - cross sections from materials.py at T (Sa_T, nuSf_T), exactly as
#     neutronics._build_phase_system uses them;
#   - face diffusion coefficient = harmonic mean (exact for an interface
#     on the face);
#   - Marshak vacuum BC, d_ext = 2.1312 D of the boundary cell's
#     material -- the SAME convention as the closed model, so
#     differences are attributable to the closure alone.
# Realisations containing no fissile material have no fundamental mode:
# they contribute k = 0 to the k statistics and are excluded from the
# flux averages (their fraction is reported).

import scipy.sparse as sps
from scipy.sparse.linalg import splu, eigs, LinearOperator


def realisation_eigen(pm: np.ndarray, dz: float, T: float = 300.0):
    """Fundamental (k, phi) for one realisation. phi shape (2, N),
    normalised to unit integrated total flux (cm units). Returns
    (0.0, None, 0.0) if the realisation holds no fissile material."""
    ph = [make_PuO(), make_combustible()]
    N = len(pm)
    Sa = np.array([ph[q].Sa_T(T) for q in (0, 1)])       # (phase, group)
    nuSf = np.array([ph[q].nuSf_T(T) for q in (0, 1)])
    D = np.array([ph[q].D for q in (0, 1)])
    s12 = np.array([ph[q].Sigma_s12 for q in (0, 1)])
    if not np.any(nuSf[pm] > 0):
        return 0.0, None, 0.0

    blocks = []
    for g in range(2):
        Dg = D[pm, g]
        Srem = Sa[pm, g] + (s12[pm] if g == 0 else 0.0)
        Df = 2.0 * Dg[:-1] * Dg[1:] / (Dg[:-1] + Dg[1:])
        diag = Srem.copy()
        diag[:-1] += Df / dz**2
        diag[1:] += Df / dz**2
        diag[0] += Dg[0] / ((2.1312 * Dg[0]) * dz)
        diag[-1] += Dg[-1] / ((2.1312 * Dg[-1]) * dz)
        blocks.append(sps.diags([-Df / dz**2, diag, -Df / dz**2], [-1, 0, 1]))
    Z = sps.csr_matrix((N, N))
    M = sps.bmat([[blocks[0], Z], [sps.diags(-s12[pm]), blocks[1]]], format="csc")
    F = sps.bmat([[sps.diags(nuSf[pm, 0]), sps.diags(nuSf[pm, 1])], [Z, Z]],
                 format="csr")                                # chi = (1, 0)

    lu = splu(M)
    op = LinearOperator((2 * N, 2 * N), matvec=lambda x: lu.solve(F @ x),
                        dtype=float)
    vals, vecs = eigs(op, k=1, which="LM", tol=1e-11, maxiter=20000)
    k = float(vals[0].real)
    phi = vecs[:, 0].real
    if phi.sum() < 0:
        phi = -phi
    negfrac = float(np.mean(phi < -1e-9 * np.abs(phi).max()))
    phi = phi / (phi.sum() * dz)
    return k, phi.reshape(2, N), negfrac


def model_fluxes(mu: float, v1: float, closure: str, H: float = 40.0,
                 N: int = 80):
    """Closed-model k_eff and (psi_1, psi_2), each (2 groups, N), from
    neutronics.static_shape_solve with the given closure, re-normalised
    jointly (unit integrated ensemble flux v_1 psi_1 + v_2 psi_2) to
    match the benchmark's per-realisation normalisation. Returns None if
    the model has no converged fundamental mode."""
    import warnings
    import neutronics as nt
    from mesh import Mesh1D
    ph = [make_PuO(), make_combustible()]
    mesh = Mesh1D(H=H, N=N)
    T = np.full((2, N), 300.0)
    v = np.array([v1, 1.0 - v1])
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            k, psi = nt.static_shape_solve(ph, mesh, T, v, mu=mu,
                                           maxiter=20000, closure=closure)
    except Exception:
        return None
    norm = (v[0] * psi[0] + v[1] * psi[1]).sum() * mesh.dz
    return dict(k=k, psi1=psi[0] / norm, psi2=psi[1] / norm, z=mesh.z)


def neutronics_benchmark(mu: float, v1: float, H: float = 40.0,
                         dz: float = 0.05, n_real: int = 300,
                         seed: int = 2024) -> dict:
    N = int(round(H / dz))
    z = (np.arange(N) + 0.5) * dz
    lam1, lam2 = chord_lengths(mu, v1)
    rng = np.random.default_rng(seed)
    ks, negs = [], []
    sums = np.zeros((2, 2, N))        # (phase, group, z)
    cnt = np.zeros((2, N))
    n_nofuel = 0
    for _ in range(n_real):
        pm = sample_phase_map(z, H, lam1, lam2, v1, rng)
        k, phi, neg = realisation_eigen(pm, dz)
        ks.append(k)
        if phi is None:
            n_nofuel += 1
            continue
        negs.append(neg)
        for q in (0, 1):
            m = (pm == q)
            sums[q][:, m] += phi[:, m]
            cnt[q] += m
    with np.errstate(invalid="ignore", divide="ignore"):
        cond = sums / cnt[:, None, :]
    return dict(mu=mu, v1=v1, lam1=lam1, lam2=lam2, z=z, k=np.array(ks),
                psi=cond, max_negfrac=max(negs) if negs else 0.0,
                nofuel=n_nofuel / n_real, n_real=n_real, dz=dz)


def safety_report(mu: float, v1: float, H: float = 40.0, n_real: int = 300,
                  seed: int = 2024, model_k: float = None) -> dict:
    """
    Realisation-ensemble safety check for a given composition: P(k_eff > 1)
    and the mean/median/std of k_eff sampled EXACTLY over independent
    realisations of the binary Markov mixture (one heterogeneous
    eigenproblem per realisation via neutronics_benchmark), independent of
    the closed-form (relaxational) closure used for the dynamic solve.

    This exists because the closed-form closure is a biased point estimate
    of this distribution, not a substitute for it -- see HANDOFF.md "Open"
    items 1-2. At the closed model's own nominal critical point for
    mu = 0.5 cm^-1 (v1 = 0.1817), the true realisation mean k is 1.04 (not
    1.0) and 67% of individual realisations are already supercritical; the
    closed model has no mechanism to report this on its own.

    model_k, if given, is the closed-form model's k_eff for the same
    (mu, v1, H) -- printed alongside for comparison, not used in the
    Monte Carlo itself.
    """
    b = neutronics_benchmark(mu, v1, H=H, n_real=n_real, seed=seed)
    k = b["k"]
    se = float(k.std() / np.sqrt(len(k)))
    return dict(mu=mu, v1=v1, H=H, n_real=n_real, model_k=model_k,
                mean=float(k.mean()), median=float(np.median(k)),
                std=float(k.std()), se=se, p_gt1=float(np.mean(k > 1)),
                nofuel=float(b["nofuel"]))


def print_safety_report(rep: dict) -> None:
    print("\n" + "=" * 50)
    print("REALISATION-ENSEMBLE SAFETY CHECK (Monte Carlo)")
    print("=" * 50)
    print(f"  mu = {rep['mu']} cm^-1, v1 = {rep['v1']:.4f}, H = {rep['H']} cm, "
          f"{rep['n_real']} realisations")
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
    print("-" * 50)
    print("  Independent of the closed-form (relaxational) closure above --")
    print("  see HANDOFF.md 'Open' items 1-2 for why, and by how much, the")
    print("  two can disagree in this mu regime.")
    print("=" * 50)


def _central(z, f, H, frac=0.2):
    m = np.abs(z - H / 2) <= frac * H / 2
    return float(np.nanmean(f[..., m]))


def report_neutronics(b: dict, models: dict, H: float = 40.0) -> None:
    k = b["k"]
    print(f"\nmu = {b['mu']} cm^-1  (lambda_1 = {b['lam1']:.3g} cm, "
          f"lambda_2 = {b['lam2']:.3g} cm; {b['n_real']} realisations, "
          f"dz = {b['dz']} cm; {100*b['nofuel']:.1f}% with no fuel)")
    print(f"  benchmark k:  mean {k.mean():.4f}  median {np.median(k):.4f}  "
          f"std {k.std():.4f}  (fundamental modes single-signed: "
          f"{b['max_negfrac'] == 0.0})")
    th = 1
    r_b = _central(b["z"], b["psi"][1, th], H) / _central(b["z"], b["psi"][0, th], H)
    print(f"  {'':24s} {'k_eff':>8s}  {'psi2/psi1 thermal, mid-slab':>28s}  {'min psi2':>9s}")
    print(f"  {'benchmark':24s} {'(above)':>8s}  {r_b:28.4f}  {np.nanmin(b['psi'][1]):9.2e}")
    for name, mres in models.items():
        if mres is None:
            print(f"  {name:24s} {'--':>8s}  {'no converged fundamental mode':>28s}")
            continue
        r_m = _central(mres["z"], mres["psi2"][th], H) / _central(mres["z"], mres["psi1"][th], H)
        print(f"  {name:24s} {mres['k']:8.4f}  {r_m:28.4f}  {mres['psi2'].min():9.2e}")


def plot_neutronics(runs: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.8))
    for ax, (b, models) in zip(np.atleast_1d(axes), runs):
        ax.plot(b["z"], b["psi"][0, 1], "C0-", lw=2, label=r"$\psi_1$ benchmark")
        ax.plot(b["z"], b["psi"][1, 1], "C1-", lw=2, label=r"$\psi_2$ benchmark")
        styles = {"document closure": "--", "relaxational (joint)": ":"}
        for name, mres in models.items():
            if mres is None:
                continue
            ls = styles.get(name, "-.")
            ax.plot(mres["z"], mres["psi1"][1], "C0" + ls, lw=1.2, label=rf"$\psi_1$ {name}")
            ax.plot(mres["z"], mres["psi2"][1], "C1" + ls, lw=1.2, label=rf"$\psi_2$ {name}")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set(xlabel="z (cm)", title=f"thermal group, $\\mu$ = {b['mu']} cm$^{{-1}}$")
    np.atleast_1d(axes)[0].set_ylabel("conditional flux (joint unit norm)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "fig_benchmark_neutronics.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    import sys
    V1 = 0.6744613390173657          # critical composition, see main.py
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("thermal", "all"):
        results = []
        for mu in (0.5, 1.0, 2.0):
            res = thermal_benchmark(mu, V1)
            report_thermal(res)
            results.append(res)
        plot_thermal(results)
    if part in ("neutronics", "all"):
        runs = []
        for mu in (0.05, 0.12, 0.5, 2.0):
            b = neutronics_benchmark(mu, V1)
            models = {"document closure": model_fluxes(mu, V1, "document"),
                      "relaxational (joint)": model_fluxes(mu, V1, "relaxational")}
            report_neutronics(b, models)
            runs.append((b, models))
        plot_neutronics(runs)
    if part in ("critical", "all"):
        # Is the model's critical composition critical for the random medium?
        MU, VC = 0.5, 0.1816804741043431
        b = neutronics_benchmark(MU, VC, n_real=400, seed=31)
        se = b["k"].std() / np.sqrt(len(b["k"]))
        print(f"\nmodel critical point mu={MU}, v1={VC:.4f}: benchmark mean k = "
              f"{b['k'].mean():.4f} +/- {se:.4f}  (median {np.median(b['k']):.4f}, "
              f"std {b['k'].std():.3f}, P(k>1) = {np.mean(b['k'] > 1):.2f})")
