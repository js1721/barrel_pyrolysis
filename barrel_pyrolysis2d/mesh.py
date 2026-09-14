import numpy as np
from dataclasses import dataclass


@dataclass
class CylindricalMesh2D:
    """
    2D cylindrical mesh on (r,z) in [0,R] x [0,H].
    Cell-centred finite difference with azimuthal symmetry.

    Indexing: node (i,j) -> flat index i*Nz + j
        i = 0,...,Nr-1   radial
        j = 0,...,Nz-1   axial

    Boundary conditions:
        r = 0  : symmetry   (Neumann dT/dr = 0)
        r = R  : insulating (Neumann dT/dr = 0)
        z = 0  : heated     (Neumann -K dT/dz = q_ext)
        z = H  : insulating (Neumann dT/dz = 0)

    For neutronics:
        r = R  : vacuum
        z = 0  : vacuum
        z = H  : vacuum
    """
    R:  float
    H:  float
    Nr: int
    Nz: int

    def __post_init__(self):
        self.dr = self.R / self.Nr
        self.dz = self.H / self.Nz

        # Cell centres
        self.r = np.linspace(
            self.dr / 2, self.R - self.dr / 2, self.Nr
        )
        self.z = np.linspace(
            self.dz / 2, self.H - self.dz / 2, self.Nz
        )

        # Cell edges
        self.r_edge = np.linspace(0.0, self.R, self.Nr + 1)
        self.z_edge = np.linspace(0.0, self.H, self.Nz + 1)

        # 2D coordinate grids
        self.R2D, self.Z2D = np.meshgrid(
            self.r, self.z, indexing='ij'
        )

        # Total nodes per phase
        self.N = self.Nr * self.Nz

        # Cell volumes (annular rings times dz)
        # dV_i = pi*(r_{i+1/2}^2 - r_{i-1/2}^2) * dz
        r_in  = self.r_edge[:-1]
        r_out = self.r_edge[1:]
        dV_r  = np.pi * (r_out**2 - r_in**2)
        self.dV = np.outer(
            dV_r, np.full(self.Nz, self.dz)
        )   # shape (Nr, Nz)

    def idx(self, i: int, j: int) -> int:
        """Flat index from (r-index, z-index)."""
        return i * self.Nz + j

    def ij(self, n: int) -> tuple:
        """(r-index, z-index) from flat index."""
        return divmod(n, self.Nz)