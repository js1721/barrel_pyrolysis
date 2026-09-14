"""
mesh.py
=======
1D slab mesh on z in [0, H].
Cell-centred finite difference.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Mesh1D:
    H: float   # slab height cm
    N: int     # number of cells

    def __post_init__(self):
        self.dz = self.H / self.N
        self.z  = np.linspace(
            self.dz / 2,
            self.H - self.dz / 2,
            self.N
        )