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
from thermal import (H_flux, advance_pyrolysis, pyrolysis_step, initialise_omega,
                      _banded_matvec, CM_TO_M)
from kinetics import source_driven_steady_state
from sources import source_density as _pu_source_density
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
    # Atomic mix: MACROSCOPIC cross sections combine linearly in volume
    # fraction, and D = 1/(3 Sigma_tr), so D combines HARMONICALLY. (The
    # superseded arithmetic average of D overstates leakage whenever the
    # two phases' D differ -- here by factors of ~6.)
    return 1.0 / _mix(v, 1.0 / phases[0].D, 1.0 / phases[1].D)


def Ss12_hom(phases, v):
    return _mix(v, phases[0].Sigma_s12, phases[1].Sigma_s12)


def rCp_hom(phases, v):
    return _mix(v, phases[0].rho * phases[0].Cp, phases[1].rho * phases[1].Cp)


def K_hom(phases, v):
    return _mix(v, phases[0].K, phases[1].K)


# ── Homogenised neutronics: single-region eigenvalue problem ─────────

def static_shape_solve_hom(phases, mesh, T, v):
    """
    Two-group fundamental mode of the ATOMIC-MIX (homogenised) slab,
    built exactly like one realisation of benchmark_markov.py but with
    uniform mixed data: Marshak vacuum BC (d_ext = 2.1312 D), downscatter
    Sigma_s12, chi = (1, 0). The model has one temperature, which is
    therefore also the neutron temperature. Returns (k, psi) with psi
    shape (2 groups, N), normalised to unit integrated total flux.
    """
    import scipy.sparse as sps
    from scipy.sparse.linalg import splu, eigs, LinearOperator
    N, dz = mesh.N, mesh.dz
    T_mean = float(np.mean(T))
    Sa, nuSf = Sa_hom(phases, T_mean, v), nuSf_hom(phases, T_mean, v)
    D, s12 = D_hom(phases, v), Ss12_hom(phases, v)
    blocks = []
    for g in range(2):
        Srem = Sa[g] + (s12 if g == 0 else 0.0)
        diag = np.full(N, Srem + 2.0 * D[g] / dz**2)
        diag[[0, -1]] = Srem + D[g] / dz**2 + D[g] / ((2.1312 * D[g]) * dz)
        off = np.full(N - 1, -D[g] / dz**2)
        blocks.append(sps.diags([off, diag, off], [-1, 0, 1]))
    Z = sps.csr_matrix((N, N))
    M = sps.bmat([[blocks[0], Z], [sps.identity(N) * -s12, blocks[1]]], format="csc")
    F = sps.bmat([[sps.identity(N) * nuSf[0], sps.identity(N) * nuSf[1]], [Z, Z]],
                 format="csr")
    lu = splu(M)
    op = LinearOperator((2 * N, 2 * N), matvec=lambda x: lu.solve(F @ x), dtype=float)
    vals, vecs = eigs(op, k=1, which="LM", tol=1e-12, maxiter=20000)
    k = float(vals[0].real)
    psi = vecs[:, 0].real
    if psi.sum() < 0:
        psi = -psi
    psi = psi.reshape(2, N)
    psi /= sum(np.trapezoid(psi[g], mesh.z) for g in range(2))
    return k, psi


def compute_Lambda_hom(phases, mesh, psi, T, v):
    """Two-group generation time of the homogenised slab (unit weight)."""
    nuSf = nuSf_hom(phases, float(np.mean(T)), v)
    v_n = phases[0].v_n            # identical for both phases in this model
    num = sum(np.trapezoid(psi[g], mesh.z) / v_n[g] for g in range(2))
    den = sum(np.trapezoid(nuSf[g] * psi[g], mesh.z) for g in range(2))
    return num / den if abs(den) > 1e-30 else 1e-5


def source_rate_hom(phases, mesh, psi, v, source_density):
    """Intrinsic source in amplitude units/s (cf. neutronics.
    source_amplitude_rate): the fuel's source density diluted by its
    volume fraction, spread uniformly through the mixture."""
    if not source_density:
        return 0.0
    v_n = phases[0].v_n
    N_w = sum(np.trapezoid(psi[g], mesh.z) / v_n[g] for g in range(2))
    return v[0] * source_density * mesh.H / N_w


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
                            T_amb=None, h_conv_amb=None, emissivity_amb=None,
                            S_comb=None):
    N, dz = mesh.N, mesh.dz * CM_TO_M
    rCp = rCp_hom(phases, v)
    AB = _build_conduction_matrix_hom(phases, mesh, dt, v)
    if h_conv_amb is None:
        h_conv_amb = h_conv
    if emissivity_amb is None:
        emissivity_amb = emissivity

    T_mean = float(np.mean(T[0]))
    Sf = Sf_hom(phases, T_mean, v)
    # W/m^3: Sigma_f cm^-1 x phi n/cm^2/s = fissions/cm^3/s; x1e6 -> m^-3
    S_f = 1.0e6 * Ef * (Sf[0] * phi[0][0] + Sf[1] * phi[0][1])
    # q_p * rate_p is already volumetric (W/m^3): omega is the density
    # (kg/m^3) of combustible content, not a dimensionless fraction --
    # see materials.py's omega0 comment and thermal.py's
    # solve_thermal_step for the matching convention -- so rate_p is a
    # volumetric mass-consumption rate (kg/m^3/s) with no separate rho
    # factor needed. v-weighted into the mixed medium as usual.
    # Combustion heat: supplied energy-exactly by the caller (S_comb, the
    # mixture's volume-weighted q*(omega_old - omega_new)/dt from
    # thermal.pyrolysis_step); the explicit form is kept only as fallback.
    if S_comb is not None:
        S_c = S_comb
    else:
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
    rho = (k_eff - 1.0) / k_eff
    if cfg.source_density is not None:
        q_dens = float(cfg.source_density)
    else:
        q_dens = _pu_source_density(cfg.isotopics, phases[0].rho * 1.0e-3)
    q = source_rate_hom(phases, mesh, psi, v, q_dens)
    P = source_driven_steady_state(q, rho, Lambda) if cfg.P0 is None else cfg.P0
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

        q = source_rate_hom(phases, mesh, psi, v, q_dens)
        P_new, C_new = advance_kinetics(P, C, rho, Lambda, dt, q)
        phi_new = np.array([P_new * psi, P_new * psi])
        omega_new, S_cp = pyrolysis_step(phases, T, omega, dt)
        T = solve_thermal_step_hom(phases, mesh, T, omega, phi_new, v, dt,
                                    cfg.T_f, cfg.h_conv, cfg.emissivity, cfg.Ef,
                                    T_amb=cfg.T_amb, h_conv_amb=cfg.h_conv_amb,
                                    emissivity_amb=cfg.emissivity_amb,
                                    S_comb=v[0] * S_cp[0] + v[1] * S_cp[1])
        omega = omega_new
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
        # v1 bisected for k_eff(0) = 1 under the relaxational closure with
        # joint eigenvalue and Marshak BC (was 0.545 under the superseded
        # operator). NOTE: this script's homogeneous model is still
        # single-group and cannot run against the two-group materials.
        H=40.0, N=80, mu=0.5, v1=0.17068927633801423,   # stochastic k0 = 0.98
        t_end=300.0, dt=0.05,
        T0=300.0,
        T_f=1200.0, h_conv=50.0, emissivity=0.3,
        Ef=3.2e-11,
    )

    # ── Finding 1: SAME composition (cfg_stoch.v1), different model ──
    # The stochastic model's k_eff = v1*k1 combines each phase's own
    # (leakage-inclusive) eigenvalue with a simple linear v-weighting
    # chosen earlier in this project -- it is NOT the same mathematical
    # operation as diluting cross-sections before solving the diffusion
    # equation (what "homogeneous" does below). So this is a genuine,
    # honestly-computed difference between the two models as built, not
    # a claim about which is "more correct" -- see the printed caveat.
    print(f"=== Finding 1: same v1={cfg_stoch.v1:.4f} in both models ===")
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
    # v1 found by bisection for k_eff(0) = 0.98 in the homogeneous
    # model specifically (very different from the stochastic model's
    # own critical v1 -- expected, given Finding 1).
    cfg_hom = Config(
        H=40.0, N=80, mu=0.05, v1=0.09894598494912485,  # atomic-mix k0 = 0.98 (mu unused)
        t_end=300.0, dt=0.05,
        T0=300.0,
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
