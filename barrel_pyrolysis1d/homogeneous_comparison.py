"""
homogeneous_comparison.py
==========================
Compares the stochastic two-phase (segregated, mu-correlated) model
against a naive "atomic mix" homogenisation: PuO and combustible
collapsed into ONE effective medium via standard volume-weighted
mixing rules, with no spatial correlation length at all (mu doesn't
apply -- homogenisation is the well-mixed limit).

Mixing rules (all volume-fraction weighted, v1*X1 + v2*X2):
    Sigma_a, Sigma_f, nu*Sigma_f : each phase's own T-dependent value
    D                            : diffusion coefficient
    rho*Cp                       : volumetric heat capacity
    K                            : thermal conductivity

Pyrolysis is tracked per-constituent (omega1, omega2) against the
SHARED temperature field, each releasing its own combustion heat
weighted by its volume fraction -- reusing advance_pyrolysis()
directly by feeding it a (2,N) T array with both rows equal.

This directly tests the document's own justification for the
stochastic model (p.5): that segregating fissile material into large
correlated "chunks" is more dangerous than a naive homogeneous-mix
estimate would suggest. If that's right, the homogenised k_eff here
should sit measurably BELOW the stochastic model's k_eff at the same
(H, v1, T) -- less "risk" from a naive averaging viewpoint.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

from mesh import Mesh1D
from materials import make_PuO, make_combustible, BETA, I_GRP
from thermal import (H_flux, advance_pyrolysis, initialise_omega,
                      _banded_matvec, CM_TO_M)
from kinetics import advance_kinetics, initialise_precursors
from neutronics import static_shape_solve, compute_Lambda
from solver import Solver, Config

BLUE   = "#2a78d6"
ORANGE = "#eb6834"
GRAY   = "#8a8a86"
INK    = "#0b0b0b"

plt.rcParams.update({
    "figure.figsize":  (6.0, 4.0),
    "figure.dpi":       200,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "font.size":        12,
    "axes.labelsize":   12,
    "axes.edgecolor":   INK,
    "axes.linewidth":   0.8,
    "axes.grid":        True,
    "grid.color":       "#d8d8d4",
    "grid.linewidth":   0.6,
    "legend.fontsize":  10,
    "legend.frameon":   False,
    "lines.linewidth":  2.0,
})


# ── Homogenised material properties (volume-weighted) ────────────────

def _mix(v, x1, x2):
    return v[0] * x1 + v[1] * x2


def Sa_hom(phases, T, v):
    return _mix(v, phases[0].Sa_T(T), phases[1].Sa_T(T))


def Sf_hom(phases, T, v):
    return _mix(v, phases[0].Sf_T(T), phases[1].Sf_T(T))


def nuSf_hom(phases, T, v):
    return _mix(v, phases[0].nuSf_T(T), phases[1].nuSf_T(T))


def D_hom(phases, v):
    return _mix(v, phases[0].D, phases[1].D)


def rCp_hom(phases, v):
    return _mix(v, phases[0].rho * phases[0].Cp, phases[1].rho * phases[1].Cp)


def K_hom(phases, v):
    return _mix(v, phases[0].K, phases[1].K)


# ── Homogenised neutronics: single-region eigenvalue problem ─────────

def static_shape_solve_hom(phases, mesh, T, v):
    N, dz = mesh.N, mesh.dz
    T_mean = float(np.mean(T))
    Sa   = Sa_hom(phases, T_mean, v)
    nuSf = nuSf_hom(phases, T_mean, v)
    D    = D_hom(phases, v)
    d_ext = 0.7104 / Sa if Sa > 1e-10 else 10.0 * dz
    r = D / dz**2

    L = np.zeros((N, N))
    F = np.zeros((N, N))
    for n in range(N):
        diag = Sa
        if n == 0:
            diag += r + D / (dz * d_ext)
            L[n, n+1] -= r
        elif n == N - 1:
            diag += r + D / (dz * d_ext)
            L[n, n-1] -= r
        else:
            diag += 2.0 * r
            L[n, n-1] -= r
            L[n, n+1] -= r
        L[n, n] = diag
        if nuSf > 1e-30:
            F[n, n] = nuSf

    # plain power iteration -- single region, no cross-phase source,
    # no eigenvalue-selection pathology to worry about
    Linv = np.linalg.inv(L)
    psi = np.full(N, 1.0 / N)
    k = 1.0
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for _ in range(500):
            src = F @ psi
            # rhs divides by the CURRENT k estimate (matching neutronics.py's
            # joint power iteration convention), so the fission-source ratio
            # below converges to k_true/k_old and must be multiplied back by
            # k_old -- NOT applied to a solve that already omits the 1/k
            # (that mismatch is what caused the earlier exponential blowup).
            psi_new = Linv @ (src / k)
            src_new = F @ psi_new
            k_new = k * (np.sum(src_new) / np.sum(src)) if np.sum(src) > 0 else k
            scale = np.max(np.abs(psi_new))
            if scale > 0:
                psi_new /= scale
            if abs(k_new - k) < 1e-9 * max(abs(k), 1.0):
                psi, k = psi_new, k_new
                break
            psi, k = psi_new, k_new

    norm = np.trapz(psi, mesh.z)
    if norm > 0:
        psi /= norm
    return k, psi


def compute_Lambda_hom(phases, mesh, psi, T, v):
    T_mean = float(np.mean(T))
    nuSf = nuSf_hom(phases, T_mean, v)
    v_n  = phases[0].v_n   # identical for both phases in this model
    num = np.trapz(psi, mesh.z) / v_n
    den = np.trapz(nuSf * psi, mesh.z)
    return num / den if abs(den) > 1e-30 else 1e-5


# ── Homogenised thermal: single-region Crank-Nicolson conduction ─────

def _build_conduction_matrix_hom(phases, mesh, dt, v):
    N, dz = mesh.N, mesh.dz * CM_TO_M
    K = K_hom(phases, v)
    rCp = rCp_hom(phases, v)
    r = K * dt / (rCp * dz**2)

    AB = np.zeros((3, N))
    AB[1, :]  = 1.0 + r
    AB[0, 1:] = -0.5 * r
    AB[2, :-1] = -0.5 * r
    AB[1, 0]  = 1.0 + 0.5 * r
    AB[1, -1] = 1.0 + 0.5 * r
    return AB


def solve_thermal_step_hom(phases, mesh, T, omega, phi, v, dt,
                            T_f, h_conv, emissivity, Ef,
                            T_amb=None, h_conv_amb=None, emissivity_amb=None):
    N, dz = mesh.N, mesh.dz * CM_TO_M
    rCp = rCp_hom(phases, v)
    AB = _build_conduction_matrix_hom(phases, mesh, dt, v)
    if h_conv_amb is None:
        h_conv_amb = h_conv
    if emissivity_amb is None:
        emissivity_amb = emissivity

    T_mean = float(np.mean(T[0]))
    S_f = Ef * Sf_hom(phases, T_mean, v) * phi[0]
    # q_p * rate_p is already volumetric (W/m^3): omega is the density
    # (kg/m^3) of combustible content, not a dimensionless fraction --
    # see materials.py's omega0 comment and thermal.py's
    # solve_thermal_step for the matching convention -- so rate_p is a
    # volumetric mass-consumption rate (kg/m^3/s) with no separate rho
    # factor needed. v-weighted into the mixed medium as usual.
    S_c = (v[0] * phases[0].q * _arrhenius(phases[0], T[0], omega[0])
           + v[1] * phases[1].q * _arrhenius(phases[1], T[1], omega[1]))
    S_total = S_f + S_c

    rhs = 2.0 * T[0] - _banded_matvec(AB, T[0]) + dt / rCp * S_total

    flux = H_flux(float(T[0, 0]), T_f, h_conv, emissivity)
    rhs[0] += dt * flux / (rCp * dz)

    # z=H: optional heat loss to an ambient exterior (see thermal.py's
    # solve_thermal_step -- same H_flux form, naturally negative/cooling
    # when the slab is hotter than T_amb). None keeps it insulating.
    if T_amb is not None:
        flux_H = H_flux(float(T[0, -1]), T_amb, h_conv_amb, emissivity_amb)
        rhs[-1] += dt * flux_H / (rCp * dz)

    T_new = solve_banded((1, 1), AB, rhs)
    return np.array([T_new, T_new])


def _arrhenius(ph, T, omega):
    Rg = 8.314
    exponent = np.where(T > 1.0, -ph.E_act / (Rg * T), -1e10)
    return ph.k_arr * omega * np.exp(exponent)


# ── Full transient run, homogenised model ─────────────────────────────

def run_homogeneous(cfg: Config):
    phases = [make_PuO(), make_combustible()]
    v = np.array([cfg.v1, 1.0 - cfg.v1])
    mesh = Mesh1D(cfg.H, cfg.N)

    T = np.full((2, mesh.N), cfg.T0)
    omega = initialise_omega(phases, mesh)

    k_eff, psi = static_shape_solve_hom(phases, mesh, T, v)
    Lambda = compute_Lambda_hom(phases, mesh, psi, T, v)
    P = cfg.P0
    C = initialise_precursors(P, Lambda)
    phi = np.array([P * psi, P * psi])
    rho = (k_eff - 1.0) / k_eff

    history = [{"t": 0.0, "k_eff": k_eff, "rho": rho, "P": P,
                "Lambda": Lambda, "T_max": float(T[0].max()),
                "T_base": float(T[0, 0])}]

    t = 0.0
    while t < cfg.t_end:
        # Same adaptive-dt safety refinement as Solver._adaptive_dt:
        # tighten dt whenever the prompt-kinetics growth rate is fast,
        # whether approaching prompt critical or already past it.
        # Without this a single fixed-size step near/above prompt
        # critical overflows before the safety check below ever runs.
        margin = BETA - rho
        dt = cfg.dt
        if margin < 0.1 * BETA:
            scale = max(abs(margin), 1e-8)
            dt = min(dt, 0.01 * abs(Lambda) / scale)
        dt = max(dt, 1e-8)

        P_new, C_new = advance_kinetics(P, C, rho, Lambda, dt)
        phi_new = np.array([P_new * psi, P_new * psi])
        T = solve_thermal_step_hom(phases, mesh, T, omega, phi_new, v, dt,
                                    cfg.T_f, cfg.h_conv, cfg.emissivity, cfg.Ef,
                                    T_amb=cfg.T_amb, h_conv_amb=cfg.h_conv_amb,
                                    emissivity_amb=cfg.emissivity_amb)
        omega = advance_pyrolysis(phases, T, omega, dt)
        k_eff, psi = static_shape_solve_hom(phases, mesh, T, v)
        Lambda = compute_Lambda_hom(phases, mesh, psi, T, v)
        rho = (k_eff - 1.0) / k_eff
        P, C = P_new, C_new
        t += dt

        history.append({"t": t, "k_eff": k_eff, "rho": rho, "P": P,
                         "Lambda": Lambda, "T_max": float(T[0].max()),
                         "T_base": float(T[0, 0])})

        if rho >= BETA:
            print(f"*** HOMOGENEOUS MODEL: PROMPT CRITICAL at t={t:.4f}s ***")
            break

    return history, T, psi, mesh


FIGDIR = "figures"


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.tight_layout()
    path = f"{FIGDIR}/{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    cfg_stoch = Config(
        H=40.0, N=80, mu=0.05, v1=0.545,
        t_end=300.0, dt=0.05,
        T0=300.0, P0=1.0,
        T_f=1200.0, h_conv=50.0, emissivity=0.3,
        Ef=3.2e-11,
    )

    # ── Finding 1: SAME composition (v1=0.545), different model ──────
    # The stochastic model's k_eff = v1*k1 combines each phase's own
    # (leakage-inclusive) eigenvalue with a simple linear v-weighting
    # chosen earlier in this project -- it is NOT the same mathematical
    # operation as diluting cross-sections before solving the diffusion
    # equation (what "homogeneous" does below). So this is a genuine,
    # honestly-computed difference between the two models as built, not
    # a claim about which is "more correct" -- see the printed caveat.
    print("=== Finding 1: same v1=0.545 in both models ===")
    phases = [make_PuO(), make_combustible()]
    mesh_chk = Mesh1D(cfg_stoch.H, cfg_stoch.N)
    T_chk = np.full((2, cfg_stoch.N), cfg_stoch.T0)
    v_chk = np.array([cfg_stoch.v1, 1.0 - cfg_stoch.v1])
    k_hom_same, _ = static_shape_solve_hom(phases, mesh_chk, T_chk, v_chk)
    k_stoch_same, _ = static_shape_solve(phases, mesh_chk, T_chk, v_chk, cfg_stoch.mu)
    print(f"  Stochastic k_eff(0)   = {k_stoch_same:.6f}  "
          f"(rho={(k_stoch_same-1)/k_stoch_same:.5f})")
    print(f"  Homogeneous k_eff(0)  = {k_hom_same:.6f}  "
          f"(rho={(k_hom_same-1)/k_hom_same:.5f})")
    print()

    # ── Finding 2: each model at its OWN near-critical v1, full 300s ──
    # v1=0.081 found by bisection for k_eff(0)~1 in the homogeneous
    # model specifically (very different from the stochastic model's
    # v1=0.545 -- expected, given Finding 1).
    cfg_hom = Config(
        H=40.0, N=80, mu=0.05, v1=0.08096321032164894,
        t_end=300.0, dt=0.05,
        T0=300.0, P0=1.0,
        T_f=1200.0, h_conv=50.0, emissivity=0.3,
        Ef=3.2e-11,
    )

    print("=== Finding 2: each model at its own near-critical v1 ===")
    print(f"=== Homogeneous (atomic mix), v1={cfg_hom.v1} ===")
    hist_hom, T_hom, psi_hom, mesh = run_homogeneous(cfg_hom)
    print(f"  k_eff(0)={hist_hom[0]['k_eff']:.6f}  "
          f"k_eff(end)={hist_hom[-1]['k_eff']:.6f}  "
          f"t_end={hist_hom[-1]['t']:.2f}s")

    print(f"=== Stochastic two-phase, v1={cfg_stoch.v1} ===")
    solver = Solver(cfg_stoch)
    hist_stoch = solver.run()
    print(f"  k_eff(0)={hist_stoch[0]['k_eff']:.6f}  "
          f"k_eff(end)={hist_stoch[-1]['k_eff']:.6f}")

    t_hom   = np.array([h["t"] for h in hist_hom])
    k_hom   = np.array([h["k_eff"] for h in hist_hom])
    rho_hom = np.array([h["rho"] for h in hist_hom])
    P_hom   = np.array([h["P"] for h in hist_hom])

    t_st    = np.array([h["t"] for h in hist_stoch])
    k_st    = np.array([h["k_eff"] for h in hist_stoch])
    rho_st  = np.array([h["rho"] for h in hist_stoch])
    P_st    = np.array([h["P"] for h in hist_stoch])

    lbl_st  = f"Stochastic ($v_1$={cfg_stoch.v1})"
    lbl_hom = f"Homogeneous ($v_1$={cfg_hom.v1})"

    fig, ax = plt.subplots()
    ax.plot(t_st, k_st, color=BLUE, label=lbl_st)
    ax.plot(t_hom, k_hom, color=ORANGE, label=lbl_hom)
    ax.axhline(1.0, color=GRAY, ls="--", lw=1.2, label="Critical")
    ax.set(xlabel="$t$ (s)", ylabel=r"$k_\mathrm{eff}$")
    ax.legend()
    _save(fig, "fig_compare_k_eff")

    fig, ax = plt.subplots()
    ax.plot(t_st, rho_st * 1e5, color=BLUE, label=lbl_st)
    ax.plot(t_hom, rho_hom * 1e5, color=ORANGE, label=lbl_hom)
    ax.axhline(BETA * 1e5, color=GRAY, ls="--", lw=1.2, label=r"$\beta$")
    ax.set(xlabel="$t$ (s)", ylabel=r"Reactivity ($\times 10^{-5}$)")
    ax.legend()
    _save(fig, "fig_compare_reactivity")

    fig, ax = plt.subplots()
    ax.semilogy(t_st, P_st, color=BLUE, label=lbl_st)
    ax.semilogy(t_hom, P_hom, color=ORANGE, label=lbl_hom)
    ax.set(xlabel="$t$ (s)", ylabel="$P(t)$")
    ax.legend()
    _save(fig, "fig_compare_power")
