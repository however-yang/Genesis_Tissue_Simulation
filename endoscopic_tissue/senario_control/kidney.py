"""Implicit kidney with an exophytic tumor, for the scissors excision scene. Native cm, Y up."""
from dataclasses import dataclass, replace

import numpy as np

from ..grid import Tab, implicit_boundary
from .instrument_control.commands import Command
from .instrument_control.joints import SCISSORS_FACE_WIDTH, scissors_frame
from .instrument_control.smoothing import Stage


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


# Kidney scene: the grasper takes the top of the tumor on the kidney's lateral
# border and pulls it up, then the scissors cut the neck that the traction draws
# out of the rounded join between tumor and kidney, which excises the whole
# tumor. With the grasper hinge 16 mm up the tumor rises 7 to 12 mm, and at
# y = 3.725, midway between node layers, the neck spans about 11 mm along x and
# 8 mm along z, all of it kidney and join below the tumor. The scissors lie in
# that plane (roll 90 degrees) and come from the front right, turned 75 degrees
# from -x toward -z, so the shaft stays off the endoscope's line of sight. The
# width axis is +y, so hinges sit SCISSORS_FACE_WIDTH below the face.
KIDNEY = Kidney()
KIDNEY_TAB = Tab(low=(-1.0, 2.8, -1.0), high=(1.0, 4.2, 1.0), attached=1)
KIDNEY_PLANE_Y = 3.725
KIDNEY_PITCH_DEG, KIDNEY_ROLL_DEG, KIDNEY_YAW_DEG = 0.0, 90.0, -75.0
KIDNEY_AXIS = scissors_frame(0.0, 90.0, -75.0)[0]
# The first bite puts the neck 4 to 12.7 mm out along the blades, which fully
# open straddle it, while the wrist envelope clears the kidney in front of the
# neck by 1 mm. Closing, the blades push the back of the neck out past the
# 12.5 mm reach of the edges: the second bite slides the opened blades 5 mm
# deeper into the cut to sever it.
KIDNEY_BITES = tuple(tuple((.06, KIDNEY_PLANE_Y - SCISSORS_FACE_WIDTH, .01) - reach * KIDNEY_AXIS) for reach in (.84, .34))
# 1.5 cm back along the scissors axis from the first bite.
KIDNEY_PARK = tuple(np.array(KIDNEY_BITES[0]) - 1.5 * KIDNEY_AXIS)
# The grasper's tips close 3 mm right of and 2 mm below the top of the tumor,
# then its hinge rises 16 mm and holds the traction through every snip: raising
# it between snips lifts the neck, so the next snip cuts it a node layer lower
# and the step between the two cuts can leave a node of the cut surface linked
# to neither side. Once the scissors are out it lifts the excised tumor 1 cm.
KIDNEY_GRASP_BITE = (-.894, 5.542, 0.0)
KIDNEY_GRASP_READY, KIDNEY_GRASP_LIFT = (-1.374, 6.182, 0.0), (-.894, 7.142, 0.0)


def kidney_stages():
    """Return the stages that pull the tumor up by its top and cut the neck under it in successive snips.

    The open scissors move in along their axis to the first bite. After each snip but the last the blades open in
    place and slide deeper into the cut. After the last snip the closed blades leave along their axis and open clear
    of the tissue, and the grasper lifts the excised tumor away.
    """
    ready = Command(hinge=KIDNEY_GRASP_READY, blade_center=KIDNEY_PARK,
                    scissors_pitch_deg=KIDNEY_PITCH_DEG, scissors_roll_deg=KIDNEY_ROLL_DEG,
                    scissors_yaw_deg=KIDNEY_YAW_DEG)
    held = replace(ready, hinge=KIDNEY_GRASP_LIFT, closure=1)
    stages = [Stage("settle", 30, replace(ready, hinge=(-2.374, 7.382, 0.0))), Stage("approach", 90, ready),
              Stage("bite", 45, replace(ready, hinge=KIDNEY_GRASP_BITE)),
              Stage("close", 45, replace(ready, hinge=KIDNEY_GRASP_BITE, closure=1)), Stage("lift", 120, held),
              Stage("blade_approach", 120, replace(held, blade_center=KIDNEY_BITES[0]))]
    for number, bite in enumerate(KIDNEY_BITES, 1):
        if number > 1:
            stages.append(Stage(f"reposition_{number}", 60, replace(held, blade_center=bite)))
        # The jaw closes over the first half and stays closed, so a fracture
        # near full closure still has time to run to the tips.
        stages.append(Stage(f"incise_{number}", 150, replace(held, blade_center=bite, scissors_closure=1,
                                                              scissors_cutting=True), ramp=.5))
        if number < len(KIDNEY_BITES):
            stages.append(Stage(f"open_{number}", 30, replace(held, blade_center=bite)))
    extracted = replace(held, hinge=tuple(np.array(KIDNEY_GRASP_LIFT) + (0, 1.0, 0)), blade_center=KIDNEY_PARK)
    return (*stages, Stage("withdraw", 120, replace(held, blade_center=KIDNEY_PARK, scissors_closure=1)),
            Stage("release", 45, replace(held, blade_center=KIDNEY_PARK)), Stage("extract", 75, extracted),
            Stage("hold", 30, extracted))

