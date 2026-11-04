"""Implicit kidney with an exophytic tumor, for the scissors excision scene. Native cm, Y up."""
from dataclasses import dataclass

import numpy as np

from .grid import implicit_boundary


def smooth_min(a, b, k):
    """Return min(a, b) where the two differ by more than `k`, blended into a rounded join of width `k` elsewhere."""
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - h * h * k / 4


def ellipsoid_distance(points, half_extents):
    """Return a signed distance bound to the origin-centred ellipsoid of `half_extents`: exact on its surface,
    negative inside."""
    k0 = np.linalg.norm(points / half_extents, axis=1)
    k1 = np.maximum(np.linalg.norm(points / np.square(half_extents), axis=1), 1e-12)
    return k0 * (k0 - 1) / k1


@dataclass(frozen=True)
class Kidney:
    """A kidney resting on its medial border with an exophytic tumor on its lateral border, as a signed distance in cm.

    The kidney is an ellipsoid of `half_extents` (length along x, width along y, thickness along z) centred on the
    origin, its lateral border up. The tumor is an ellipsoid of `tumor_half_extents` whose centre lies `tumor_offset`
    above the top of the kidney. Their join is rounded over `blend`: traction on the tumor stretches it into a neck
    narrower than the tumor. The lower `bed_depth` of the kidney lies in its fatty bed and is pinned there.

    Only the kidney above `bed_height` is tissue. Below it the kidney is pinned and never moves, so the lattice keeps
    just the pinned layer the free tissue rests on: the forces on every free node stay as they are.
    """
    half_extents: tuple[float, float, float] = (5.5, 3.0, 1.75)
    tumor_half_extents: tuple[float, float, float] = (.75, .55, .75)
    tumor_offset: float = .6
    blend: float = .3
    bed_depth: float = 1.5

    @property
    def tumor_center(self):
        return np.array((0, self.half_extents[1] + self.tumor_offset, 0))

    @property
    def bed_height(self):
        return self.bed_depth - self.half_extents[1]

    @property
    def low(self):
        # The lattice spans the whole kidney: its node layers then sit where they sit without the pinned bed cut off.
        return -np.array(self.half_extents, dtype=np.float32)

    @property
    def high(self):
        a, b, c = self.half_extents
        return np.array((a, b + self.tumor_offset + self.tumor_half_extents[1], c), dtype=np.float32)

    def distance(self, points):
        """Return the signed distance of `points`, shape (n, 3), negative inside the tissue."""
        body = ellipsoid_distance(points, np.array(self.half_extents))
        tumor = ellipsoid_distance(points - self.tumor_center, np.array(self.tumor_half_extents))
        return np.maximum(smooth_min(body, tumor, self.blend), self.bed_height - points[:, 1])

    def tumor_weight(self, points, width):
        """Return 1 for `points` inside the tumor's ellipsoid and 0 outside it, ramping over `width` cm across its
        surface.

        The rounded join between tumor and kidney is kidney tissue, its capsule drawn up onto the tumor.
        """
        tumor = ellipsoid_distance(points - self.tumor_center, np.array(self.tumor_half_extents))
        return np.clip(.5 - tumor / width, 0, 1)

    def inside(self, positions):
        return self.distance(positions.astype(np.float64)) < 0

    def boundary(self, positions, a, b, inside):
        return implicit_boundary(self.distance, positions, a, b)

    def bed(self, positions):
        return positions[:, 1] < self.bed_height
