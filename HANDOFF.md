# Handoff — percolation submodel

Written 2026-09-14, at the end of the session that implemented the
moderator-percolation submodel. Purpose: let a new session (on any
device) pick up without re-deriving the context.

## What this project is

Coupled stochastic-mixture heat conduction + two-group neutron
diffusion / point kinetics for a PuO2 + combustible-waste storage
drum. Python solver in `barrel_pyrolysis1d/`; the paper lives in
`paper/`.

**On `paper/`:** it is simultaneously a clone of an Overleaf repo *and*
vendored into this repo, so the `.tex` is readable and editable without
Overleaf credentials. That means two copies exist between syncs. Treat
this repo as the working copy; Overleaf gets updated by pushing from
inside `paper/` (which has its own `origin`) from the machine that holds
those credentials. Don't edit both in the same stretch of work.

## What changed this session

**New:** `barrel_pyrolysis1d/percolation.py`

Mobile liquid-moderator saturation `theta(z,t)`: melting, gravity
drainage, flash-off, and a saturating-sigmoid map from `theta` to a
per-cell correlation length `ell`, hence `mu = 1/ell`.

**Modified:**

| File | Change |
|---|---|
| `thermal.py` | `mu` accepts scalar **or** `(N,)` array; new optional `S_extra` volumetric source |
| `neutronics.py` | same scalar→array generalisation of `mu` |
| `solver.py` | `theta` added to `State`; `PercolationConfig` field on `Config`; percolation wired into the operator split; percolation CFL added to `_adaptive_dt` |

`materials.py`, `kinetics.py`, `mesh.py`, `main.py` untouched.

## Design decisions worth not re-litigating

- **Percolation params live in their own `PercolationConfig`**, not on
  `materials.Phase`. `mu`/`ell` is a mixture/geometry property shared
  between both phases, not a single phase's material data.
- **`z=0` is the heated bottom**, so gravity drains toward *decreasing*
  z. The document's draft text didn't pin a direction; this module
  makes the choice explicit in its docstring.
- **Transport uses a Cell Transmission Model flux limiter**
  (supply/demand), not plain upwind. Plain upwind either broke mass
  conservation when clipped, or collapsed the timestep forever once a
  cell saturated — saturation is a *persistent* state, unlike the
  transient reaction singularities `solver._adaptive_dt` handles.
- **`percolation.enabled=False` is an exact no-op.**
  `Solver._mu_field()` short-circuits to the scalar `Config.mu` without
  consulting `theta` at all. Verified bit-identical to pre-percolation
  behaviour; please keep this property.
- **The `AB[2,:-1]` indexing fix in `_build_conduction_matrix`** now
  uses `s[1:]`, not `s[:-1]`. `AB[2,j]` is the coefficient in row
  *j+1*, so the drift term must be evaluated at the row's own cell.
  Invisible with a uniform mu; real once mu varies.

## Verification status

Tested in a sandbox **without scipy** (no network to install it), using
a pure-numpy stand-in for `solve_banded` / `lu_factor` / `lu_solve` /
`expm` injected via `sys.modules`. The module source under test was
byte-identical to what's committed here — only the dependency was
faked.

Passing:

- scalar-mu vs uniform-array-mu: **exactly 0.0 difference** in both
  `thermal.py` and `solve_thermal_step`, and in `neutronics.static_shape_solve`
- spatially-varying mu runs clean, no NaNs, sane `k_eff` and flux
- full solver with percolation disabled reproduces baseline **bit-for-bit**,
  including when handed nonsense percolation parameters
- percolation enabled runs end-to-end; `theta` genuinely evolves
- `mu=0` analytical limits unperturbed (bare-slab `k_eff` ~0.03% at the
  meshes tested; semi-infinite conduction profile)
- `percolation.py`'s own self-test: mass balance residual ~9.4e-14

## Outstanding

1. **Run the real test suite on a scipy-equipped machine** —
   `verification_tests.py` and `main.py`. My checks used a substitute
   linear-algebra backend, so LAPACK-specific behaviour is unconfirmed.
2. **`main.py` doesn't expose percolation.** It still constructs a
   plain `Config`, so percolation is off by default there. Needs a
   `PercolationConfig(...)` passed in, with parameters chosen for the
   real PVC-dominated waste case (the values in `percolation.py`'s
   self-test are illustrative, not fitted).
3. **`paper/secondary.tex` not yet updated** to match the final code —
   deliberately deferred. Also still contains a stray empty
   `\subsubsection{Percolation}` stub, and a large commented-out
   "Numerical Results" section referencing seven figures that do exist
   in `paper/`.
4. **Doppler section nitpicks** in `secondary.tex`: the "vanish"
   wording, and `T_ref=293K` in prose vs the table's 300K row.

## Repo state

Local repo on `main`, one commit, remote `origin` set to
`git@github.com:js1721/barrel_pyrolysis.git`. **Not yet pushed** as of
writing — run `git push -u origin main`.

---

## Session 2026-09-21 — closure correction, benchmark, source-driven kinetics, 2D port

### 1D operator corrections (both `thermal.py` and `neutronics.py`)
- Code implemented the **superseded, commented-out** slab equation (dimensionally
  inconsistent: `mu dT/dz` is K/m^2 vs W/m^3). Now the active equation, finite-volume
  (conservative term as face current; `d mu/dz` retained when mu varies).
- Sign errors found by term-by-term tests: thermal self-drift (with a backward stencil —
  correct upwind is FORWARD), neutronic algebraic coupling, neutronic top-boundary drift.
- BCs fix the TOTAL phase current (diffusive + stochastic); `stoch_bc` removed.
- Integrals are finite-volume cell sums (trapezoid over cell centres dropped end half-cells).

### The document closure is unphysical — realisation benchmark (`benchmark_markov.py`)
Closure-independent: sample Markov-mixture realisations, solve exactly, average conditionally.
- Document closure: phase temperatures DIVERGE in an isolated slab (2nd-law violation);
  combustible flux NEGATIVE; no fundamental mode for mu >~ 0.2. The mu^2 term's sign is even
  in the orientation convention (Williams' equations share it). Williams eqn (69) has a sign
  slip relative to his own B8 — the document's sign is the self-consistent one.
- Benchmark: exchange must be RELAXATIONAL; mu^2 rate scaling confirmed (thermal decay is a
  universal function of lambda_c t) but not single-exponential (long tail).
- Per-phase `k_eff = v1 k1` is exact only for infinitely coarse mixing: put criticality at
  v1 = 0.67 where the random medium is critical near 0.13 (~5x non-conservative).
- Relaxational closure + ONE eigenvalue for the coupled system: k ~1% low at mu = 0.12,
  ~6% LOW (non-conservative) at mu = 0.5–2 with Marshak BC; over-couples at fine mixing.
- Ensemble spread matters: at the model's k=1 point, 67% of realisations are supercritical.

### Now in the 1D code
- `Config.closure = "relaxational"` (default) | `"document"` (kept for comparison).
- Joint eigenvalue (ARPACK) in `static_shape_solve`; physical psi_2/psi_1; no per-phase sign
  flip (sign change -> warning); `ClosureBreakdownError` instead of silent garbage;
  `compute_Lambda` volume-fraction weighted.
- Marshak `d_ext = 2.1312 D` (was 0.7104/Sigma_removal: 75.7 cm vs 0.28 cm for combustible
  thermal — near-reflecting "vacuum").
- Thermal-group 1/v at the NEUTRON (moderator) temperature for all phases; material T via
  Doppler only. Removes a spurious ~ -16% fuel coefficient. Uniform-T results unchanged.
- ABSOLUTE power: phi = P psi in n/cm^2/s; fission heating x1e6 to W/m^3 (was ~1e-14 W/m^3,
  i.e. fission heating absent in all earlier runs). `sources.py`: Pu intrinsic source from
  isotopics (PANDA yields; (a,n) values approximate — CHECK against your isotopics).
  `P0 = None` -> source-driven steady state P0 = -q Lambda / rho0 (needs k0 < 1).
- Criticality by composition exists only for mu > ~0.071 cm^-1 (bigger fuel chords are
  supercritical alone). Scripts now start SUBCRITICAL at k0 = 0.98 (a choice).
- `homogeneous_comparison.py` ported to two groups (harmonic D for atomic mix).
- `operator_tests.py` (Test 3 of `verification_tests.py`) exercises mu != 0.

### 2D (`barrel_pyrolysis2d/`, old modules in `_superseded/`)
- `neutronics.py`: two-group, relaxational, joint eigenvalue, Marshak on z=0, z=H, r=R.
  `neutronics_reduction_test.py`: Nr=1 reflective reproduces 1D k, psi ratio, Lambda and
  absolute flux to ~1e-15.
- `thermal.py`: closure switch, T_n, W/m^3 heating; `reduction_test.py` (document operator).
- `solver.py`/`main.py`: 200 L drum (R 28.6, H 88), k0 = 0.98 at v1 = 0.160 (lean branch).
  Energy-exact pyrolysis heat release. A 20x40 cm drum cannot reach k 0.98 at any v1.
- The 2D exchange uses |m|^2 = mu_r^2 + mu_z^2 (2x slab at mu_r = mu_z): UNVALIDATED —
  needs an r-z realisation benchmark.

### Open
1. Derive the exchange from the Markov transition rates (not a benchmarked sign flip);
   address the ~6% k bias and the non-exponential thermal tail.
2. P(k > 1) over realisations, not just the ensemble mean, for safety claims.
3. Positive uniform-T coefficient in moderator-rich mixtures rests on crude 1/v thermal group.
4. Paper: document-closure results in Section 5 are superseded; LaTeX segment
   `cylindrical_section.tex` needs revisiting for the relaxational closure.
5. Test 1 flat convergence (~0.028% floor) — shared-buckling analytic reference, unrelated.

### Pyrolysis energy (follow-up)
- Combustible `q` was 1.6e7 J/kg (≈ PVC heat of COMBUSTION): adiabatic rise ~1e4 K. Now
  3.6e5 J/kg (Bamford/Williams heat of pyrolysis). Real PVC dehydrochlorination is net
  endothermic — replace with a PVC-specific value if available.
- `advance_pyrolysis` bug (original commit): used the MASS rate k*omega*exp(..) as a rate
  constant in omega/(1+dt*rate) → volatiles consumed ~omega0 (~450x) too fast, only 2–4% of
  their heat deposited. Replaced by `thermal.pyrolysis_step` (exact at fixed T, heat =
  q*(omega_old − omega_new)/dt, energy-exact); used by 1D solver, homogeneous model, 2D solver.
  Operator Test 8 checks conservation (~1e-13). The two errors nearly cancelled in 1D
  (effective q ~3–6e5), which is why 1D looked sane while 2D ignited.

### Exchange coefficient: derived, and the 2D doubling removed
- Geometric derivation (series resistance across the interface, a_v = 4 v1 v2 mu from the
  chord lengths): h_geo/h_doc = 0.9*beta, mu-INDEPENDENT — same functional scaling as the
  document's mu^2 coefficient, agreeing within 10% for shape factor beta = 1. So h now rests
  on a derivation as well as the benchmark. (The two combine K_1,K_2 differently —
  series/harmonic vs arithmetic — which coincide numerically near this composition only.)
- 2D now uses the scalar mu_ex = 1/l_c (arg `mu_ex`, default max(mu_r,mu_z)) instead of
  mu_r^2+mu_z^2, so the same medium gets the same h in 1D and 2D. Drum composition re-bisected:
  k0 = 0.98 at v1 = 0.2027 (was 0.1599); drum k peaks ~1.17 near v1 ~ 0.5.

### The ~6% k bias is NOT the exchange — it is the diffusion coefficients
- No alpha_h reproduces the benchmark k (needs ~20 at mu=0.5; no solution at 0.12 or 2.0).
- The model's h -> infinity limit is k = 1.3684, exactly the ARITHMETIC-D atomic mix, whereas
  the correct atomic mix (harmonic D, since Sigma_tr mixes linearly) gives 1.4859. Summing the
  volume-weighted phase equations at psi_1 = psi_2 leaves sum_i v_i D_i — arithmetic. With D
  differing ~6x between phases this is a large, NON-conservative leakage error.
- D_i* = f D_i with f = D_harm/D_arith fixes the limit exactly (1.4859) and the fine-mixing
  bias (mu=0.5: -6.5% -> -1.7%; mu=2: -6.1% -> +1.4%) but BREAKS the coarse limit
  (mu=0.12: -1.1% -> +11.5%), as it must: a big chunk leaks with its own D.
- The f the benchmark requires is NON-MONOTONIC in chunk/diffusion-length
  (0.95 at 9.9, 0.375 at 2.4, 0.60 at 0.59), so no bulk<->harmonic interpolation can work:
  an extra intermediate-scale effect (flux depression inside fuel chunks) acts where
  chunk ~ diffusion length and a single-flux-per-phase model cannot represent it.
- NOT implemented: derive D_i* from the two-equation cell problem, with
  sum_i v_i D_i* = D_harm as the test; expect a residual intermediate-scale discrepancy.

### The distribution of k (benchmark_markov.py distribution)
`k_statistics`, `composition_study`, `plot_distribution`; run `python benchmark_markov.py
distribution`. At mu = 0.5, 250 realisations per point:

  v1      bench mean  median  P(k>1)  95th pct   closed model k
  0.020     0.195     0.000    0.04     0.973        0.265
  0.040     0.432     0.229    0.20     1.141        0.434
  0.100     0.794     0.902    0.40     1.290        0.764
  0.160     0.979     1.075    0.59     1.367        0.952
  0.182     1.052     1.142    0.70     1.367        1.000   <- model "critical"
  0.220     1.111     1.170    0.77     1.394        1.069

- The distribution is BIMODAL: a dense reactive cluster near k ~ 1.1-1.3 (well-mixed drums)
  and a tail of fuel-poor realisations down to k = 0. At v1 = 0.16 the ensemble mean falls in
  the TROUGH between the modes -- no drum has k near the mean, so the mean is not a
  description of any realisation.
- At the closed model's critical composition, 70% of drums are supercritical and the 95th
  percentile is 1.37.
- A criterion of P(k>1) <= 5% needs v1 ~ 0.02, where the closed model reports k = 0.27: an
  ORDER OF MAGNITUDE less fuel than a mean-based criterion allows.
- A mean-field closure cannot produce a percentile at all, however good its closure. For a
  per-drum safety statement the realisation benchmark (or a variance closure, as Williams
  does for temperature/volatiles) is the method, not a check on it.
- Caveats: 1D slab; 250 realisations (P(k>1) +/- ~0.03); chord statistics of a real 3D drum
  differ from the 1D layered medium.

### Effective diffusion coefficients D_i* (fixes most of the 6% bias)
`neutronics.effective_D` (1D and 2D). In a layered medium flux AND current are continuous, so
at fine mixing the conditional currents are equal -- not -D_i grad(psi_i) with bulk D_i. A
two-equation model without cross-diffusion therefore mixes D arithmetically; the atomic mix
needs it HARMONIC. Limits are set by chunk vs diffusion length, lambda_i = 1/(mu v_k),
L = sqrt(D/Sigma_rem):
    w = 1/(1 + (lambda_i/L_i)^2),   D_i* = (1-w) D_i + w D_harm       (no fitted parameter)
Benchmark error in k: -1.1/-6.5/-6.1%  ->  -0.2/-2.1/+1.0%  at mu = 0.12/0.5/2.0.
Operator Test 9 checks both limits (bulk D; and fine-mixing k = harmonic atomic mix, 1.4859).
The residual is NOT a D effect: the correction the benchmark requires is non-monotonic in
lambda/L (0.95, 0.375, 0.60 at lambda/L = 9.9, 2.4, 0.59), pointing to flux depression inside
fuel chunks where chunk ~ diffusion length -- beyond a model with one flux per phase.
Compositions re-bisected: 1D k0=0.98 at v1 = 0.1707 (mu=0.5), 0.2166 (mu=0.2);
2D drum v1 = 0.1875. The thermal analogue (K_i*, tortuosity) is NOT implemented.

### Thermal analogue K_i* — analysed, NOT wired in
`thermal.effective_K` (documented, exercised by operator Test 10; the conduction operator
still uses bulk K_i).
- The defect is the same as the neutronic one: at T_1 = T_2 the phase equations leave
  sum_i v_i K_i (arithmetic), while a layered medium conducts with the HARMONIC mean
  (heat flux, not gradient, is continuous at an interface). K_1/K_2 = 4 here.
- It CANNOT be fixed from the microstructure alone. The neutronic interpolation works because
  sqrt(D/Sigma_rem) is a material length independent of the mixing scale. Conduction has no
  such length: using the model's intrinsic timescale 1/lambda_c (lambda_c ~ mu^2) gives a
  penetration depth ~1/mu, scaling exactly like the mean chord, so the ratio is
  SCALE-INVARIANT -> a fixed w (~0.35 here) that never reaches either limit.
- The comparison length must be macroscopic: in a transient, delta_i = sqrt(alpha_i t).
  effective_K returns that form when given t (bulk K at t->0, harmonic as t grows), but the
  INTERPOLATION IS UNVALIDATED and it brings history dependence into a per-step coefficient.
- To settle it: a heated-slab realisation benchmark comparing conditional T_i(z,t) against
  realisations (the existing thermal benchmark is an isolated slab with no macroscopic
  gradient, so it tests the exchange only).

### Heated-slab thermal benchmark (settles K_i*)
`benchmark_markov.heated_slab_benchmark` / `model_heated_slab` / `report_heated_slab`;
run `python benchmark_markov.py heated`. Slab heated at z=0 (h = 1e4, effectively Dirichlet
at T_f = 1000 K), insulated at z=H, no sources, 150-200 realisations; conditional T_i(z,t)
compared with the closed model using bulk K and using effective_K's K*(t).
`thermal.solve_thermal_step` gained an optional `K_eff` (conduction only; the exchange keeps
bulk K).

  t (s)  phase   RMS err bulk K   RMS err K*(t)   mean bias bulk / K*   bench rise
   10      1         26.5             26.4            +4.5 / +4.4        up to 494 K
   30      1         22.7             21.8            +5.4 / +5.0        up to 588 K
   60      1         21.4             18.8            +6.5 / +5.5        up to 625 K
   60      2         15.8             15.6            +4.7 / +4.6        up to 614 K

- The model runs ~5 K HOT on average (<1% of a ~600 K rise): it over-conducts, the sign the
  arithmetic-K defect predicts. K*(t) removes part of it (fuel phase at 60 s: 6.5 -> 5.5 K).
- But the effect is sub-1%, and RMS error (~3-4% of the local rise, dominated by the steep
  front) is barely changed. CONCLUSION: K_i* stays UNWIRED -- the correction does not justify
  putting history dependence into a per-step coefficient. Revisit for larger K contrast,
  longer times, or finer mixing.
- The residual RMS is not a conductivity error; the likely candidate is the exchange's
  non-exponential behaviour (the long tail seen in the isolated-slab benchmark).
