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
