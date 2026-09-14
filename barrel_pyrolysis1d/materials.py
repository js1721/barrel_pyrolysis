"""
materials.py
============
Material properties and delayed neutron data.
Two energy groups: index 0 = fast, index 1 = thermal.
"""

import numpy as np
from dataclasses import dataclass, field

# ── Delayed neutron data (6-group Pu-239) ─────────────────────
BETA_I   = np.array([0.0002, 0.0022, 0.0060,
                      0.0026, 0.0014, 0.0006])
LAMBDA_I = np.array([0.0124, 0.0305, 0.111,
                      0.301,  1.14,   3.01 ])
BETA     = float(BETA_I.sum())
I_GRP    = len(BETA_I)

# ── Energy group structure ──────────────────────────────────────
N_GROUPS = 2   # 0 = fast, 1 = thermal

# Fission spectrum: fraction of fission neutrons born into each group.
# Fission neutrons are emitted at ~MeV energies regardless of which
# isotope/group caused the fission -- none are born thermal -- so this
# is a property of fission itself, not a per-material quantity.
CHI = np.array([1.0, 0.0])


@dataclass
class Phase:
    """
    All properties for one phase, two energy groups (index 0=fast,
    1=thermal). Cross sections in cm^-1, lengths in cm.
    """
    name:  str

    # --- Thermal (heat, not neutron-thermal) ---
    rho:   float    # density          kg/m^3
    Cp:    float    # specific heat    J/kg/K
    K:     float    # conductivity     W/m/K

    # --- Nuclear, per group: [fast, thermal] ---
    D:       np.ndarray  # diffusion coeff  cm
    Sigma_a: np.ndarray  # absorption       cm^-1
    Sigma_f: np.ndarray  # fission          cm^-1
    nu_Sf:   np.ndarray  # nu * Sigma_f     cm^-1
    v_n:     np.ndarray  # neutron speed    cm/s

    # Downscatter removal cross section, fast -> thermal, cm^-1.
    # Upscatter (thermal -> fast) is neglected -- standard for a
    # coarse 2-group split, since thermal neutrons essentially never
    # regain enough energy to re-enter the fast group.
    Sigma_s12: float = 0.0

    # --- Pyrolysis ---
    q:      float   = 0.0   # heat of combustion  J/kg
    k_arr:  float   = 0.0   # Arrhenius pre-exp   s^-1
    E_act:  float   = 1.0   # activation energy   J/mol
    omega0: float   = 0.0   # initial density of the COMBUSTIBLE fraction of
                             # this phase's material, kg/m^3 (distinct from
                             # rho, the whole material's bulk density -- e.g.
                             # combustible waste is only partly volatile
                             # organic content, the rest inert filler/residue
                             # that never pyrolyses). omega decays via the same
                             # Arrhenius kinetics regardless of scale (the ODE
                             # is homogeneous in omega), so no other equation
                             # needs to change form -- only omega0's value and
                             # the derived quantities' unit interpretation.

    T_ref:  float = 293.0

    # Doppler (resonance-broadening) coefficient, K^-1/2. Applied only
    # to THERMAL-group absorption (see Sa_T): captures the standard
    # reactor-physics picture that resonance broadening in the fuel
    # predominantly increases parasitic capture, not fission, as fuel
    # temperature rises. alpha_D=0 (the default) recovers plain 1/v
    # scaling with no Doppler feedback -- combustible keeps this
    # default since it has no significant absorption resonance
    # structure here (see make_combustible's comment).
    alpha_D: float = 0.0

    def Sa_T(self, T: float) -> np.ndarray:
        """
        Absorption cm^-1, per group. FAST-group absorption (index 0)
        is treated as T-independent: the 1/v Maxwellian-averaging
        picture specifically describes a population in near-thermal
        equilibrium with the material, which fast neutrons (freshly
        born at ~MeV energies, far from equilibrium) are not -- at
        this coarse 2-group resolution there's no physical basis for
        giving the fast group its own T-dependence. THERMAL-group
        absorption (index 1) keeps the original 1/v thermal-averaging
        (sqrt(T_ref/T)) plus a Doppler resonance-broadening correction
        (the alpha_D term): thermal motion of the absorber nucleus
        broadens resonances as T rises, increasing effective
        absorption -- a negative reactivity feedback. Sa_T(T_ref)[1]
        = Sigma_a[1] exactly (both terms are referenced to zero at
        T=T_ref).
        """
        Tc = max(T, 1.0)
        thermal_factor = (
            np.sqrt(self.T_ref / Tc)
            + self.alpha_D * (np.sqrt(Tc) - np.sqrt(self.T_ref))
        )
        return np.array([self.Sigma_a[0], self.Sigma_a[1] * thermal_factor])

    def Sf_T(self, T: float) -> np.ndarray:
        """Fission cm^-1, per group. Thermal-averaging only in the
        thermal group (see Sa_T note); fast group unchanged."""
        Tc = max(T, 1.0)
        thermal_factor = np.sqrt(self.T_ref / Tc)
        return np.array([self.Sigma_f[0], self.Sigma_f[1] * thermal_factor])

    def nuSf_T(self, T: float) -> np.ndarray:
        """nu*Sigma_f cm^-1, per group. Thermal-averaging only in the
        thermal group (see Sa_T note); fast group unchanged."""
        Tc = max(T, 1.0)
        thermal_factor = np.sqrt(self.T_ref / Tc)
        return np.array([self.nu_Sf[0], self.nu_Sf[1] * thermal_factor])


def make_PuO() -> Phase:
    """
    PuO phase — two-group thermal/fast cross sections.
    Thermal-group (index 1) data from JENDL-4.0 Pu-239 at 0.0253 eV
    (unchanged from the single-group model this was extended from).
    Fast-group (index 0) data is illustrative -- order-of-magnitude
    reasonable for a heavy fissile oxide, not fit to a library: Pu-239
    has a huge thermal fission resonance, so fast Sigma_f/Sigma_a are
    taken much smaller than thermal (~1/50, ~1/20); fast D is taken
    somewhat larger than thermal (less scattering at higher energy is
    typical); Sigma_s12 (downscatter) is modest, reflecting that a
    dense, heavy-nucleus oxide is a comparatively weak moderator
    (elastic scatter off heavy nuclei loses little energy per
    collision, unlike light moderators).
    """
    Sigma_a_th = 0.12
    Sigma_f_th = 0.08
    nu = 2.88   # nu = 2.88 for Pu-239
    return Phase(
        name    = "PuO",
        # Thermal (heat)
        rho     = 11460.0,
        Cp      = 230.0,
        K       = 8.0,
        # Nuclear: [fast, thermal]
        D       = np.array([1.5, 0.8]),
        Sigma_a = np.array([Sigma_a_th / 20.0, Sigma_a_th]),
        Sigma_f = np.array([Sigma_f_th / 50.0, Sigma_f_th]),
        nu_Sf   = np.array([Sigma_f_th / 50.0, Sigma_f_th]) * nu,
        v_n     = np.array([3.0e7, 2.2e5]),
        Sigma_s12 = 0.02,
        # Illustrative Doppler coefficient (not fit to JENDL resonance
        # data -- this is a coarse 2-group model with no explicit
        # resonance treatment, applied only to the thermal group's
        # absorption -- see Phase.Sa_T). Chosen to give ~5% k_inf
        # reduction over 293K to 1200K, a physically reasonable order
        # of magnitude for a resonance absorber; see materials.py
        # header note and the barrel pyrolysis document's Doppler
        # section.
        alpha_D = 0.0015,
        # No pyrolysis -- omega0's value is inert here (k_arr=0 makes
        # the Arrhenius rate identically 0 regardless of omega), kept
        # at half of rho only for unit-convention consistency with
        # make_combustible().
        q       = 0.0,
        k_arr   = 0.0,
        E_act   = 1.0,
        omega0  = 0.5 * 11460.0,
    )


def make_combustible() -> Phase:
    """
    Averaged combustible phase (PVC-dominated mix), two-group.

    Thermal properties represent an effective medium
    average over the heterogeneous waste matrix contents
    (plastics, metals, concrete, paper).

    Effective conductivity K=2.0 W/m/K is justified by
    the presence of metallic components in the PCM waste
    stream, which significantly enhance bulk conductivity
    relative to pure PVC (K=0.19 W/m/K).

    Bulk density 900 kg/m^3 accounts for void fraction
    and heterogeneous packing of the waste matrix.

    Arrhenius parameters represent effective single-step
    decomposition of the PVC-dominated organic fraction.

    Fast-group (index 0) nuclear data is illustrative, matching
    make_PuO()'s convention: no fission in either group (unchanged);
    fast D taken larger than thermal by the same ratio as PuO's, for
    consistency; fast absorption taken only modestly smaller than
    thermal (this phase has no giant thermal resonance the way PuO
    does, so a less dramatic fast/thermal absorption ratio than PuO's
    is used); Sigma_s12 (downscatter) taken larger than PuO's --
    light-element content (H, C in PVC/paper) is a more effective
    moderator than PuO's heavy nuclei, slowing neutrons down more
    readily per collision.
    """
    Sigma_a_th = 0.0095
    return Phase(
        name    = "Combustible",

        # Effective medium thermal properties
        rho     = 900.0,     # kg/m^3  bulk waste density
        Cp      = 800.0,     # J/kg/K  effective heat capacity
        K       = 2.0,       # W/m/K   effective conductivity

        # Nuclear: [fast, thermal]
        D       = np.array([0.25, 0.13]),
        Sigma_a = np.array([Sigma_a_th / 5.0, Sigma_a_th]),
        Sigma_f = np.array([0.0, 0.0]),
        nu_Sf   = np.array([0.0, 0.0]),
        v_n     = np.array([3.0e7, 2.2e5]),
        Sigma_s12 = 0.04,
        # Left at the class default (alpha_D=0.0): tested a nonzero
        # value (0.0015, matching PuO's illustrative magnitude) to see
        # whether it would offset this phase's uncompensated 1/v
        # absorption decrease -- it barely moved the result (k_eff at
        # T2=1500K only dropped from 1.0162 to 1.0150, still clearly
        # net positive), and flattening it fully would need alpha_D
        # ~0.029, 19x that value, with no data or physical basis to
        # support it. PVC, concrete, and paper -- this phase's stated
        # dominant components -- are light-element materials with
        # well-behaved 1/v thermal absorption and no significant
        # resonance structure; the only mentioned metallic content is
        # invoked solely for thermal conductivity (see K comment
        # above), with no isotope/fraction specified to derive a
        # resonance-broadening coefficient from. So alpha_D=0 remains
        # the physically defensible choice here, and the system's net
        # POSITIVE temperature coefficient (driven by this phase's
        # uncompensated 1/v absorption drop, since it's the one that
        # actually heats up the most -- furnace-side, exothermic
        # pyrolysis) is a genuine result of that assumption, not an
        # artifact to tune away. See project history /
        # feedback_comparison.py.

        # PVC pyrolysis (unchanged)
        q       = 1.6e7,
        k_arr   = 2.0e13,
        E_act   = 1.46e5,
        # omega0: density of the COMBUSTIBLE/volatile fraction of this
        # waste matrix, kg/m^3 -- distinct from rho (900 kg/m^3, the
        # WHOLE material's bulk density including inert metal/concrete
        # content). Set to half of rho: roughly half the bulk material
        # is assumed to be volatile organic content capable of
        # pyrolysing, the rest inert filler/residue that never burns.
        omega0  = 0.5 * 900.0,
    )
