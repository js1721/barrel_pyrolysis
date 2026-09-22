"""
Self-test for the 2D percolation module.

The decisive check is EQUIVALENCE: because transport is axial-only and
carries no radial metric factor, a 2D run whose rings all see the same
temperature column must reproduce the 1D solver's theta exactly (to
machine precision), and rings with different temperature columns must
each independently reproduce the 1D result for their own column.

If that holds, the 2D generalization has introduced no transport error
-- any difference from 1D in the full solver then comes from the
thermal/neutronic coupling, not from this module.

Pure numpy, no scipy. Run directly: `python percolation_selftest.py`
"""

import sys
import os
import numpy as np

import percolation as p2

# Import the 1D module under a DISTINCT module name. A plain
# `sys.path.insert` + `import percolation` would silently re-bind the
# already-imported 2D module (same module name, cached in sys.modules),
# making the equivalence check below compare the 2D module against
# itself and pass vacuously.
import importlib.util             # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
_one_d_path = os.path.join(
    os.path.dirname(_here), "barrel_pyrolysis1d", "percolation.py"
)
_spec = importlib.util.spec_from_file_location("percolation_1d", _one_d_path)
p1 = importlib.util.module_from_spec(_spec)
sys.modules["percolation_1d"] = p1
_spec.loader.exec_module(p1)
assert p1 is not p2, "1D module failed to load separately from the 2D one"


def make_cfg(mod):
    return mod.PercolationConfig(
        enabled=True,
        theta_r=0.02, theta_s=0.30,
        T_melt=400.0, T_boil=900.0,
        k_melt=0.05, k_flash=0.02,
        L_fus=2.0e5, L_vap=8.0e5, rho_liquid=900.0,
        K_sat=0.05, n_perc=3.0,
        ell0=1.0/0.3, ell_max=15.0, theta_c=0.25, s0_perc=1.0, p_perc=4.0,
    )


np.set_printoptions(precision=4, suppress=True)

Nr, Nz = 6, 40
H, R = 40.0, 20.0
dz = H / Nz
z = np.linspace(dz / 2, H - dz / 2, Nz)
r = np.linspace(R / Nr / 2, R - R / Nr / 2, Nr)

cfg2 = make_cfg(p2)
cfg1 = make_cfg(p1)

# Radially varying temperature: outer rings heat less (a crude stand-in
# for radial heat loss), so the rings are genuinely different columns
# and the equivalence check is not trivially testing identical copies.
ring_factor = np.linspace(1.0, 0.55, Nr)


def T2_2d(t):
    base = 300.0 + 700.0 * np.exp(-z / 12.0) * np.tanh(t / 40.0)
    return 300.0 + ring_factor[:, None] * (base - 300.0)[None, :]


# ── Run the 2D solver ────────────────────────────────────────────────
theta2 = p2.initialise_theta(cfg2, Nr, Nz)
t, t_end = 0.0, 300.0
melted = np.zeros(Nr)
flashed = np.zeros(Nr)
dt_list = []
theta_t40 = None

while t < t_end:
    T2 = T2_2d(t)
    dt = min(p2.adaptive_dt(theta2, T2, dz, cfg2), 0.5, t_end - t)
    dt_list.append(dt)
    theta2, Gm, Gf = p2.advance_percolation(theta2, T2, dz, dt, cfg2)
    melted += dz * np.sum(Gm, axis=1) * dt
    flashed += dz * np.sum(Gf, axis=1) * dt
    t += dt
    if theta_t40 is None and t >= 40.0:
        theta_t40 = theta2.copy()

print(f"2D run: {len(dt_list)} steps, dt in "
      f"[{min(dt_list):.4g}, {max(dt_list):.4g}] s")
print(f"theta: min={theta2.min():.4f} max={theta2.max():.4f}")

# ── Per-ring mass balance ────────────────────────────────────────────
mass0 = dz * Nz * cfg2.theta_r
resid = np.array([
    dz * np.sum(theta2[i]) - mass0 - (melted[i] - flashed[i])
    for i in range(Nr)
])
print(f"\nPer-ring mass balance residual: max|r| = {np.abs(resid).max():.3e}")
assert np.abs(resid).max() < 1e-8 * max(dz * np.sum(theta2), 1.0), \
    "per-ring mass balance violated -- transport is not conservative"

# ── Drainage direction, per ring ─────────────────────────────────────
peak_z = np.array([z[np.argmax(theta_t40[i])] for i in range(Nr)])
print(f"t=40s peak z per ring (cm): {peak_z}")
assert np.all(peak_z < H / 4), \
    "expected early drainage to concentrate theta toward z=0 in every ring"

# ── Equivalence to 1D, ring by ring ──────────────────────────────────
# Each ring is re-run through the *1D* module with that ring's own
# temperature column and the 2D run's exact dt sequence.
max_dev = 0.0
for i in range(Nr):
    th1 = p1.initialise_theta(cfg1, Nz)
    tt = 0.0
    for dt in dt_list:
        T2col = 300.0 + ring_factor[i] * (
            300.0 + 700.0 * np.exp(-z / 12.0) * np.tanh(tt / 40.0) - 300.0
        )
        th1, _, _ = p1.advance_percolation(th1, T2col, dz, dt, cfg1)
        tt += dt
    dev = np.abs(th1 - theta2[i]).max()
    max_dev = max(max_dev, dev)

print(f"\nMax |theta_2D - theta_1D| over all rings: {max_dev:.3e}")
assert max_dev < 1e-14, (
    "2D transport does not reproduce 1D per-column -- the axial-only "
    "generalization has introduced an error"
)

# ── mu field ─────────────────────────────────────────────────────────
mu = p2.mu_field(theta2, cfg2)
print(f"mu: shape={mu.shape} min={mu.min():.4f} max={mu.max():.4f} cm^-1")
assert mu.shape == (Nr, Nz)

mu_dis = p2.mu_field(theta2, p2.PercolationConfig(enabled=False, ell0=cfg2.ell0))
assert np.allclose(mu_dis, 1.0 / cfg2.ell0), \
    "disabled config must return uniform dry mu0"

print("\nAll 2D self-checks passed.")
