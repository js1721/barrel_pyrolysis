# Handoff — percolation submodel

Written 2026-09-14, at the end of the session that implemented the
moderator-percolation submodel. Purpose: let a new session (on any
device) pick up without re-deriving the context.

## What this project is

Coupled stochastic-mixture heat conduction + two-group neutron
diffusion / point kinetics for a PuO2 + combustible-waste storage
drum. Python solver in `barrel_pyrolysis1d/`; the paper is a separate
Overleaf-backed git repo under `paper/` (gitignored here — Overleaf is
the single source of truth for the `.tex`).

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
