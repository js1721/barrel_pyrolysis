"""
sources.py
==========
Intrinsic neutron source of the fuel (PuO2) phase.

A stored drum is not at an arbitrary power: its neutron population is
sustained by the plutonium's own neutron emission -- spontaneous
fission (mainly Pu-238, Pu-240, Pu-242) plus (alpha,n) reactions on
oxygen in the oxide. For a subcritical drum this fixes the initial
power uniquely,

    P_0 = -q Lambda / rho_0,

so the absolute power scale follows from the isotopic vector rather
than being chosen (see solver.Solver.initialise).

Yields, neutrons / s / g of isotope:
  SF    -- Reilly et al., "Passive Nondestructive Assay of Nuclear
           Materials" (NUREG/CR-5550, 1991), Table 11-1; Pu-240 value
           consistent with later evaluations (IAEA NDS 2015 update).
  (a,n) -- in the OXIDE, same reference family (PANDA ch. 11). These
           are approximate and depend on O-17/O-18 content and
           impurities; confirm against your own isotopic data.
Light-element (alpha,n) sources in the combustibles (e.g. fluorine,
beryllium, boron impurities) are NOT included and can dominate in
some waste streams.
"""

SF_YIELD = {          # n/s/g, spontaneous fission
    "Pu238": 2.59e3, "Pu239": 2.18e-2, "Pu240": 1.02e3,
    "Pu241": 5.0e-2, "Pu242": 1.72e3, "Am241": 1.18,
}
AN_OXIDE_YIELD = {    # n/s/g, (alpha,n) on oxygen in the oxide (approximate)
    "Pu238": 1.34e4, "Pu239": 3.81e1, "Pu240": 1.41e2,
    "Pu241": 1.3,    "Pu242": 2.0,    "Am241": 2.69e3,
}

# Indicative isotopic vectors (mass fractions of Pu, plus Am-241 per g
# Pu). Reactor grade: a typical LWR discharge composition; Am-241 grows
# in from Pu-241 decay during storage (half-life 14.3 y) and is set to
# zero here -- add it for aged material, where it can dominate (a,n).
PRESETS = {
    "weapons": {"Pu238": 0.0001, "Pu239": 0.938, "Pu240": 0.058,
                "Pu241": 0.0035, "Pu242": 0.0002, "Am241": 0.0},
    "reactor": {"Pu238": 0.013, "Pu239": 0.603, "Pu240": 0.243,
                "Pu241": 0.091, "Pu242": 0.050, "Am241": 0.0},
}

PU_MASS_FRACTION_IN_PUO2 = 239.5 / (239.5 + 2 * 16.0)   # ~0.882


def specific_yield(isotopics) -> dict:
    """Neutrons / s / g Pu, split into SF and (alpha,n)."""
    iso = PRESETS[isotopics] if isinstance(isotopics, str) else dict(isotopics)
    sf = sum(f * SF_YIELD[k] for k, f in iso.items())
    an = sum(f * AN_OXIDE_YIELD[k] for k, f in iso.items())
    return {"sf": sf, "alpha_n": an, "total": sf + an}


def source_density(isotopics, oxide_density_g_cc: float) -> float:
    """Intrinsic source, neutrons / s / cm^3 of the PuO2 phase."""
    rho_pu = oxide_density_g_cc * PU_MASS_FRACTION_IN_PUO2
    return specific_yield(isotopics)["total"] * rho_pu


if __name__ == "__main__":
    for name in PRESETS:
        y = specific_yield(name)
        print(f"{name:8s}: SF {y['sf']:7.1f} + (a,n) {y['alpha_n']:7.1f} = "
              f"{y['total']:7.1f} n/s/g Pu  ->  "
              f"{source_density(name, 11.46):.3e} n/s/cm^3 of PuO2")
