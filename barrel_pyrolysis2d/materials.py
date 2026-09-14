import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class DelayedGroup:
    """Single delayed neutron precursor group."""
    beta_i:   float    # delayed fraction
    lambda_i: float    # decay constant s^-1


# 6-group delayed neutron data for Pu-239
DELAYED_GROUPS = [
    DelayedGroup(0.0002, 0.0124),
    DelayedGroup(0.0022, 0.0305),
    DelayedGroup(0.0060, 0.111 ),
    DelayedGroup(0.0026, 0.301 ),
    DelayedGroup(0.0014, 1.14  ),
    DelayedGroup(0.0006, 3.01  ),
]

BETA  = sum(g.beta_i   for g in DELAYED_GROUPS)
I_GRP = len(DELAYED_GROUPS)


@dataclass
class Material:
    """
    All material properties for one phase.
    Nuclear cross sections in cm^-1.
    Lengths in cm, temperatures in K.
    """
    name: str

    # --- Thermal ---
    rho:  float    # density          kg/m^3
    Cp:   float    # specific heat    J/kg/K
    K:    float    # conductivity     W/m/K

    # --- Nuclear (one-group) ---
    D:       float   # diffusion coeff  cm
    Sigma_a: float   # absorption       cm^-1
    Sigma_f: float   # fission          cm^-1
    nu:      float   # neutrons/fission
    chi_p:   float   # prompt spectrum

    # --- Pyrolysis ---
    q:      float    # heat of combustion  J/kg
    k_arr:  float    # pre-exponential     s^-1
    E_act:  float    # activation energy   J/mol
    omega0: float    # initial fuel fraction

    # --- Temperature dependence ---
    T_ref: float = 293.0

    def Sigma_a_T(self, T: float) -> float:
        """Doppler broadened absorption cross section."""
        return self.Sigma_a * np.sqrt(
            self.T_ref / np.maximum(T, 1.0)
        )

    def Sigma_f_T(self, T: float) -> float:
        """Doppler broadened fission cross section."""
        return self.Sigma_f * np.sqrt(
            self.T_ref / np.maximum(T, 1.0)
        )

    def D_T(self, T: float) -> float:
        """Diffusion coefficient (constant for now)."""
        return self.D


def make_PuO() -> Material:
    return Material(
        name    = "PuO",
        rho     = 11460.0,
        Cp      = 230.0,
        K       = 8.0,
        D       = 0.8,
        Sigma_a = 0.12,
        Sigma_f = 0.08,
        nu      = 2.88,
        chi_p   = 1.0,
        q       = 0.0,
        k_arr   = 0.0,
        E_act   = 1.0,
        omega0  = 1.0,
    )


def make_combustible() -> Material:
    return Material(
        name    = "Combustible",
        rho     = 900.0,
        Cp      = 2000.0,
        K       = 0.2,
        D       = 2.5,
        Sigma_a = 0.001,
        Sigma_f = 0.0,
        nu      = 0.0,
        chi_p   = 0.0,
        q       = 3.0e6,
        k_arr   = 1.0e8,
        E_act   = 1.25e5,
        omega0  = 1.0,
    )
    