"""Tissue the lattice is built from, independent of the cavity around it."""
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TissueModel(Protocol):
    """Signed-distance tissue in native centimetres, Y up.

    `low` and `high` bound it. `distance` is negative inside. `inside` selects lattice nodes in the tissue.
    `boundary` places the surface between an interior node and an exterior neighbour. `bed` selects nodes that stay
    pinned. `surface_points` lie on the surface. `surface_kind` is "fat", "liver", or "kidney", the albedo of the
    deformable mesh. `texture_cm` is the width of one repeat of that albedo.
    """

    low: np.ndarray
    high: np.ndarray
    surface_kind: str
    texture_cm: float

    def distance(self, points):
        """Return the signed distance of each sample, negative inside the tissue."""

    def inside(self, positions):
        """Return which lattice positions lie in the tissue."""

    def boundary(self, positions, a, b, inside):
        """Return the surface between an interior lattice node and an exterior neighbour."""

    def bed(self, positions):
        """Return which positions stay pinned."""

    def surface_points(self):
        """Return points on the tissue surface, in centimetres."""
