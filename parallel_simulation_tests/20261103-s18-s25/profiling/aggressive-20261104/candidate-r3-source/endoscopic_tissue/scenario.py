"""Scripted baseline with measured grasp/lift/cut criteria, independent of rendering."""
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .grid import METERS_PER_UNIT, Tab, build_grid
from .instrument_geometry import load_grasper
from .kidney import Kidney
from .native_backend import Command, Observation, ScissorsPose, Settings, Surface, scissors_frame


@dataclass(frozen=True)
class Stage:
    name: str
    steps: int
    command: Command
    # Fraction of the stage over which the command reaches its target; the
    # rest of the stage holds it.
    ramp: float = 1.0


STAGES = (
    Stage("settle", 20, Command()),
    Stage("approach", 45, Command(hinge=(-1.4, 1.734, -1.288))),
    Stage("close", 40, Command(hinge=(-1.4, 1.734, -1.288), closure=1)),
    Stage("lift", 60, Command(hinge=(-1.65, 2.434, -1.288), closure=1)),
    Stage("blade_approach", 90, Command((-1.65, 2.434, -1.288), 1, (0.15, 1.05, -0.6))),
    Stage("incise", 100, Command((-1.65, 2.434, -1.288), 1, (0.15, 0.15, -0.6))),
    Stage("withdraw", 180, Command((-1.65, 2.434, -1.288), 1, (0.15, 2.2, -0.6))),
    Stage("hold", 30, Command((-1.65, 2.434, -1.288), 1, (0.15, 2.2, -0.6))),
)


@dataclass(frozen=True)
class ScissorsScene:
    """One scissors demonstration: its tissue, shear plane, acceptance region and scripted stages."""
    # Tissue grown on the block, or with `shape` the box around the part cut
    # off: the cut stays in it, and the blades pinch its uncut tissue.
    tab: Tab
    # Height of the material cells, cm.
    vertical_spacing: float
    # Native axis normal to the shear plane, and the rest coordinate on it at
    # which the poses place the shear plane, midway between node layers. The
    # cut follows the tissue as the blades deform it, so the material columns
    # along this axis part near that coordinate, at their own levels (see
    # cut_ribbon_tick).
    plane_axis: int
    plane: float
    # Offset of the blades' shear face from `plane` along the plane axis, cm.
    face_offset: float
    # Every material column along the plane axis through this box must end
    # severed exactly once.
    cut_low: tuple[float, float, float]
    cut_high: tuple[float, float, float]
    # Unit direction in which the cut advances through the tab, snip after snip.
    advance: tuple[float, float, float]
    stages: tuple[Stage, ...]
    # Demo camera, Genesis metres: one fixed (position, look-at) for the whole
    # run, then a close-up from this offset to the wound centre.
    shot: tuple[tuple[float, float, float], tuple[float, float, float]]
    closeup_offset: tuple[float, float, float]
    # Grasper contact model of the flow, see Settings.contact_model.
    contact_model: int
    # Tissue in place of the block and its tab, and its Young's modulus.
    shape: Kidney | None = None
    young_pa: float = Settings.young_pa
    # Material pieces at the end: 1 when the cut leaves the tissue whole, 2
    # when it excises the shape's tumor, which the grasper then holds.
    pieces: int = 1

    def build_grid(self, density):
        """Return the lattice of the scene's tissue, of `density` in kg/m^3."""
        if self.shape is not None:
            return build_grid(SCISSORS_SPACING, density=density, vertical_spacing=self.vertical_spacing,
                              shape=self.shape)
        return build_grid(SCISSORS_SPACING, density=density, vertical_spacing=self.vertical_spacing, tab=self.tab)


# Lateral material cell of the scissors scenes, cm.
SCISSORS_SPACING = .1
# Width coordinate of the blades' shared shear face in the hinge frame (see
# load_scissors): the shear plane passes this far from the hinge.
SCISSORS_FACE_WIDTH = .0282
GRASP_ABOVE, GRASP_BITE, GRASP_LIFT = (-2.1, 3.2, 0), (-1.94, 1.67, 0), (-2.1, 2.40, 0)


def grasp_stages(park: Command):
    """Return the stages that grasp and lift the block while the scissors wait at `park`."""
    return (Stage("settle", 30, replace(park, hinge=GRASP_ABOVE, closure=0)),
            Stage("approach", 60, replace(park, hinge=GRASP_BITE, closure=0)),
            Stage("close", 60, replace(park, hinge=GRASP_BITE, closure=1)),
            Stage("lift", 75, replace(park, hinge=GRASP_LIFT, closure=1)))


# Flap scene: a 6 mm flap on the block's +x face is split from its free end to
# its root. The cutting edges end 12.5 mm from the hinge and the 8.36 mm wrist
# envelope must stay clear of the free end, so the scissors reach about 6.5 mm
# into the flap. Its free end sags 0.9 mm, along the shear plane z = 0.05. The
# width axis is native -z, so hinges sit at z = 0.05 + SCISSORS_FACE_WIDTH.
FLAP_TAB = Tab(low=(3.0, .3, -1.0), high=(3.6, .7, 1.0), attached=0)
FLAP_PLANE_Z = .05
FLAP_PARK = (5.6, 1.2, .0782)
# Hinges on the scissors axis tilted 19.9 degrees down, through the sagged
# mid-plane of the free end at (3.61, 0.41). Between snips the closed tips clear
# the free end by 1 mm. The edge ends of the first two bites reach x = 3.40 and
# 3.23 cm.
FLAP_CLEAR = (4.94, .891, .0782)
# Each bite is (hinge, downward tilt in degrees). At 19.9 degrees closing blades
# squeeze the root's lower corner out past the edge ends, and a deeper bite would
# sink the tips into the block face under the flap. The last bite turns the
# wrist to 6 degrees: its axis meets the root at (3.03, 0.40), the edge ends
# stop at x = 3.03 cm, tissue squeezed toward the tips meets the block face,
# and the wrist envelope stays 1 mm off the free end.
FLAP_BITES = (((4.575, .759, .0782), 19.9), ((4.405, .698, .0782), 19.9), ((4.273, .531, .0782), 6.0))


def flap_snips():
    """Return the stages of successive snips that split the flap from its free end to its root.

    Each snip moves the open scissors to straddle the flap, closes them with cutting enabled, slides the closed
    blades out of their own cut and opens them clear of the tissue: opening inside the slit would press the blade
    bodies into the fresh walls. The last snip withdraws to the park pose.
    """
    stages = []
    for number, (bite, pitch) in enumerate(FLAP_BITES, 1):
        last = number == len(FLAP_BITES)
        retreat = FLAP_PARK if last else FLAP_CLEAR
        # The wrist turns only on the way in, clear of tissue: the closed blades
        # leave along their own axis.
        stages += [
            Stage("blade_approach" if number == 1 else f"reposition_{number}", 120 if number == 1 else 60,
                  Command(GRASP_LIFT, 1, bite, scissors_pitch_deg=pitch)),
            # The jaw closes over the first half and stays closed, so a fracture
            # near full closure still has time to run to the tips.
            Stage(f"incise_{number}", 150, Command(GRASP_LIFT, 1, bite, scissors_closure=1, scissors_cutting=True,
                                                   scissors_pitch_deg=pitch), ramp=.5),
            Stage("withdraw" if last else f"withdraw_{number}", 120 if last else 60,
                  Command(GRASP_LIFT, 1, retreat, scissors_closure=1, scissors_pitch_deg=pitch)),
            Stage("release" if last else f"release_{number}", 45 if last else 30,
                  Command(GRASP_LIFT, 1, retreat, scissors_pitch_deg=pitch)),
        ]
    return tuple(stages)


# The flap keeps the anchor-assisted grasp. Held by pad pressure and friction
# alone (contact model 1), the lift drags the bite along the pads with 0.5 to
# 0.7 times their 2.5 to 3.7 N of pressure, beyond static friction (0.6), and
# the flap slides out of the jaws before the first snip.
FLAP_SCENE = ScissorsScene(
    FLAP_TAB, .05, 2, FLAP_PLANE_Z, 0.0, (3.0, .2, -1.2), (3.8, .8, 1.2), (-1, 0, 0),
    (*grasp_stages(Command(blade_center=FLAP_PARK)), *flap_snips(),
     Stage("hold", 30, Command(GRASP_LIFT, 1, FLAP_PARK, scissors_pitch_deg=FLAP_BITES[-1][1]))),
    ((.010, -.080, .062), (.010, 0, .004)), (.003, -.014, .028), contact_model=2)


# Wall scene: a 4 mm thick, 7 mm tall wall stands on the block top, flush with
# its front face, and is cut along its length on the horizontal plane
# y = 1.05, 3.5 mm above the block top and midway between node layers. A cut
# higher up a taller wall lets the near blade bend the wall back out of reach
# instead of pinching it. The far blade body reaches 1.9 mm below its face and
# clears the block top behind the wall. Before the blades reach it the wall
# settles 0.25 mm, yet the face rides 0.25 mm above the plane: in the later
# snips the far blade body sweeps along the uncut lower wall, which no edge can
# cut, and 0.5 mm lower it digs in deep enough to press its back 0.2 to 0.36 mm
# into the tissue. Its cells are 1 mm tall: with 0.5 mm cells the settling alone
# moves a node layer next to the shear face. The scissors come from the front right and cut toward -x, lying in the
# plane (roll 90 degrees) and turned 55 degrees from -x toward the wall, the
# hinge 6 mm in front of it: fully open, the near blade stays in front of the
# wall and the far blade crosses it almost square. Closed, the blade axis
# crosses the whole wall within the 12.5 mm reach of the cutting edges and the
# tips end 0.7 mm behind its back face: tips closing on wall tissue they cannot
# cut would stall the jaws open. The wall thus lies in the outer half of the
# blades, and the closing blades push it 3 to 4 mm toward the tips before their
# edges reach it: each snip cuts about 5 mm of wall. Between snips the scissors
# open in place and slide 5 mm along the wall, the far blade at the end of the
# cut. The wrist stays in front of the block, its envelope 0.59 mm off the front
# face. The width axis is +y, so hinges sit SCISSORS_FACE_WIDTH below the shear
# face. The cut follows the wall as the grasper and the blades deform it and
# can land a few node layers off the plane, so the cut box spans the wall from
# half a cell above the block top to half a cell below its top.
WALL_TAB = Tab(low=(0.0, .7, 1.6), high=(2.5, 1.4, 2.0), attached=1)
WALL_PLANE_Y, WALL_FACE_OFFSET = 1.05, .025
WALL_HINGE_Y = WALL_PLANE_Y + WALL_FACE_OFFSET - SCISSORS_FACE_WIDTH
WALL_ORIENTATION = dict(scissors_pitch_deg=0.0, scissors_roll_deg=90.0, scissors_yaw_deg=-55.0)
WALL_AXIS = scissors_frame(0.0, 90.0, -55.0)[0]
# Fully open, the far blade crosses the front face 0.1 mm short of the hinge's
# x: the first bite keeps it past the wall's end, the later ones at the end of
# the cut left by the snip before.
WALL_BITES = ((2.55, WALL_HINGE_Y, 2.60), (2.05, WALL_HINGE_Y, 2.60), (1.55, WALL_HINGE_Y, 2.60))
# 1.5 cm back along the scissors axis from the first and last bites: the open
# scissors move in along it, the closed blades leave along it.
WALL_PARK, WALL_EXIT = (3.41, WALL_HINGE_Y, 3.83), (2.41, WALL_HINGE_Y, 3.83)
# The grasper jaws straddle the wall and close on the free end of the strip
# the first snip cuts, their tips at (2.42, 1.20), then lift it. During the
# first snip they wait 6 mm up their own axis: the grasper moves 0.22 mm per
# step at most and would take 175 steps from its park pose.
WALL_GRASP_READY, WALL_GRASP_BITE, WALL_GRASP_LIFT = (.87, 3.27, 1.80), (1.23, 2.79, 1.80), (1.13, 3.19, 1.80)
# Traction variant: before any cut the jaws close on the top of the wall's free
# end, their tips at (2.48, 1.30) so that the pads stay above the near blade,
# lift it 1.2 mm, and raise it 0.5 mm more after every snip. The first lift
# stays small: it raises the settled wall at the shear plane by up to 0.6 mm at
# its free end and hardly at all 15 mm away, so the face stays between the node
# layers that straddle the plane at rest, the layers it cuts.
WALL_HOLD_READY, WALL_HOLD_BITE, WALL_HOLD_LIFT = (.93, 3.37, 1.80), (1.29, 2.89, 1.80), (1.29, 3.01, 1.80)
WALL_HOLD_RAISE = .05


def wall_stages(traction=False):
    """Return the stages that cut one wound along the wall in successive snips.

    The open scissors move in along their axis to the first bite. Each snip closes the blades with cutting enabled.
    Between snips the open blades slide along the wall to the next bite, the far blade inside the cut: entering the
    cut anew through its mouth on the front face, the far blade body has to push the lip below the plane down against
    the block and pushes the whole wall back instead. After the last snip the closed blades leave along their own axis
    and open clear of the tissue, and the grasper keeps holding the strip.

    Without `traction` the open grasper moves above the wall's free end during the first snip; after it the blades
    open in place and the grasper takes the free end of the cut strip and lifts it off the cut. With `traction` the
    grasper takes the top of the free end and lifts it before the scissors move in, and raises it after every snip.
    """
    bites, park, exit_ = WALL_BITES, WALL_PARK, WALL_EXIT
    if traction:
        lift = np.array(WALL_HOLD_LIFT)
        ready = Command(WALL_HOLD_READY, 0, park, **WALL_ORIENTATION)
        held = replace(ready, hinge=tuple(lift), closure=1)
        stages = [Stage("settle", 30, replace(ready, hinge=GRASP_ABOVE)), Stage("approach", 165, ready),
                  Stage("bite", 45, replace(ready, hinge=WALL_HOLD_BITE)),
                  Stage("close", 45, replace(ready, hinge=WALL_HOLD_BITE, closure=1)), Stage("lift", 60, held),
                  Stage("blade_approach", 120, replace(held, blade_center=bites[0]))]
    else:
        ready = Command(WALL_GRASP_READY, 0, bites[0], **WALL_ORIENTATION)
        held = replace(ready, hinge=WALL_GRASP_LIFT, closure=1)
        stages = [Stage("settle", 30, replace(ready, hinge=GRASP_ABOVE, blade_center=park)),
                  Stage("blade_approach", 120, ready)]
    for number, bite in enumerate(bites, 1):
        grasper = ready if number == 1 and not traction else held
        if number > 1:
            stages.append(Stage(f"reposition_{number}", 90, replace(held, blade_center=bite)))
        # The jaw closes over the first half and stays closed, so a fracture
        # near full closure still has time to run to the tips.
        stages.append(Stage(f"incise_{number}", 150, replace(grasper, blade_center=bite, scissors_closure=1,
                                                              scissors_cutting=True), ramp=.5))
        if number < len(bites):
            stages.append(Stage(f"open_{number}", 30, replace(grasper, blade_center=bite)))
        if traction:
            lift = lift + (0, WALL_HOLD_RAISE, 0)
            held = replace(held, hinge=tuple(lift))
            last = number == len(bites)
            stages.append(Stage(f"raise_{number}", 30, replace(held, blade_center=bite, scissors_closure=float(last))))
        elif number == 1:
            stages += [Stage("approach", 45, replace(ready, hinge=WALL_GRASP_BITE)),
                       Stage("close", 45, replace(ready, hinge=WALL_GRASP_BITE, closure=1)),
                       Stage("lift", 60, held)]
    stages += [Stage("withdraw", 120, replace(held, blade_center=exit_, scissors_closure=1)),
               Stage("release", 45, replace(held, blade_center=exit_))]
    return (*stages, Stage("hold", 30, replace(held, blade_center=exit_)))


# The wall flows keep the anchor-assisted grasp. Held by pad pressure and
# friction alone (contact model 1), the bite sticks to the pads until a closing
# snip drags it along them with more than static friction (0.6) times their 0.5
# to 1.5 N of pressure: the strip slides out of the jaws in the first or second
# snip, also with a jaw drive four times as stiff.
WALL_SCENE = ScissorsScene(
    WALL_TAB, .1, 1, WALL_PLANE_Y, WALL_FACE_OFFSET, (1.45, .75, 1.5), (2.6, 1.35, 2.1), (-1, 0, 0), wall_stages(),
    ((.008, -.078, .048), (.006, -.006, .008)), (-.004, -.026, .022), contact_model=2)
WALL_TRACTION_SCENE = replace(WALL_SCENE, stages=wall_stages(traction=True))


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
KIDNEY_ORIENTATION = dict(scissors_pitch_deg=0.0, scissors_roll_deg=90.0, scissors_yaw_deg=-75.0)
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
    ready = Command(KIDNEY_GRASP_READY, 0, KIDNEY_PARK, **KIDNEY_ORIENTATION)
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


KIDNEY_SCENE = ScissorsScene(
    KIDNEY_TAB, .1, 1, KIDNEY_PLANE_Y, 0.0, (-.3, 2.85, -.25), (.3, 3.15, .25), tuple(KIDNEY_AXIS), kidney_stages(),
    ((-.026, -.07, .076), (.003, 0, .034)), (-.012, -.03, .025), contact_model=2, shape=KIDNEY, young_pa=15000,
    pieces=2)

SCISSORS_SCENES = {"wall": WALL_SCENE, "wall-traction": WALL_TRACTION_SCENE, "flap": FLAP_SCENE,
                   "kidney": KIDNEY_SCENE}


def scissors_settings():
    # The grasper enters from the upper left. Its jaws close along z: a bite
    # that squeezes the block from both sides lifts it reliably, while jaws
    # opening in the x-y plane mostly drag the surface by friction.
    # The jaw drive closes the first snip fully. In the later snips of the wall
    # the far blade body, below the shear plane, sweeps along the uncut lower
    # wall, which no edge can cut, and holds the jaws 1.5 to 2.5 mm open
    # whatever the drive: a stiffer drive only presses the blade back into it.
    # The scissors hold their pose against up to 5 N: a far blade driving uncut
    # wall along its length meets 1.5 to 6 N and would push a 1 N drive several
    # millimetres off its bite, short of the planned cut.
    # Three solver substeps per frame. With one, the third snip fractures with
    # 2-3 N on one blade instead of 0.1-0.4 N, some snips after less than
    # 0.1 mm of two-blade squeeze, and the blade backs overlap the tissue by
    # more than 0.2 mm: the squeeze rule and the overlap acceptance fail. With
    # two, the blade backs overlap the wall by 0.21-0.34 mm and the flap's
    # grasp keeps less than half of its anchors.
    return Settings(substeps=3, cutting_tool="scissors", jaw_axis=(.6, -.8, 0), jaw_normal=(0, 0, 1),
                    blade_preload_n=5.0, scissors_jaw_stiffness=2500)


def interpolate(a: Command, b: Command, fraction: float):
    u = np.clip(fraction, 0, 1)
    u = u * u * (3 - 2 * u)
    hinge = tuple((1 - u) * np.array(a.hinge) + u * np.array(b.hinge))
    blade = tuple((1 - u) * np.array(a.blade_center) + u * np.array(b.blade_center))
    return Command(hinge, (1 - u) * a.closure + u * b.closure, blade, b.blade_enabled,
                   (1 - u) * a.scissors_closure + u * b.scissors_closure, b.scissors_cutting,
                   (1 - u) * a.scissors_pitch_deg + u * b.scissors_pitch_deg,
                   (1 - u) * a.scissors_roll_deg + u * b.scissors_roll_deg,
                   (1 - u) * a.scissors_yaw_deg + u * b.scissors_yaw_deg)


def stage_start(name: str, previous: Command, observed: Observation):
    # Withdraw from the solved pose. Starting from the old drive target would
    # first push farther into tissue to catch up with an unfinished incision.
    if name.startswith("withdraw"):
        return replace(previous, blade_center=tuple(observed.blade[:3]))
    return previous


def drive_command(requested: Command, observed: Observation, settings: Settings):
    # Bound actuator preload using the accepted pose, as in the Unity operator
    # controller. A blocked knife must not accumulate the rest of a scripted path.
    travel = np.array(requested.blade_center) - observed.blade[:3]
    limit = settings.blade_preload_n / (settings.blade_drive_stiffness * METERS_PER_UNIT)
    distance = np.linalg.norm(travel)
    center = observed.blade[:3] + travel * min(1, limit / max(distance, 1e-12))
    return replace(requested, blade_center=tuple(center))


def severed_edges(baseline, current):
    """Count undirected edges once using positive-axis slots."""
    return np.count_nonzero((baseline[:, 1::2] >= 0) & (current[:, 1::2] < 0))


def edge_separation(scissors: ScissorsPose, point):
    """Return the opening in cm between the two solved cutting edges at the axial coordinate of `point`.

    The opening is measured along the scissors normal, toward which the first blade opens. It turns negative once the
    edges have crossed there, so it keeps decreasing while the blades close.
    """
    radius = (point - scissors.hinge) @ scissors.axis
    ends = []
    for a, b in scissors.edges:
        ta, tb = (a - scissors.hinge) @ scissors.axis, (b - scissors.hinge) @ scissors.axis
        ends.append(a + (radius - ta) / (tb - ta) * (b - a))
    return float((ends[0] - ends[1]) @ scissors.normal)


@dataclass
class SnipRecord:
    """Fracture of one scissors snip, from the stage that closed the blades."""
    stage: str
    fracture_left_n: float
    fracture_right_n: float
    # Closing travel of the two edges where they pinch the uncut tab (its free
    # end, or the front of the slit after an earlier snip) from the start of the
    # step in which the native fracture law registers both blades loading the
    # tissue, up to the last step before fracture. One loaded blade only bends
    # the tab, so its travel is not counted.
    two_blade_squeeze_m: float


@dataclass
class Metrics:
    completed: bool = False
    grasp_acquired: bool = False
    lift_m: float = 0.0
    edges_before_incision: int = 0
    cut_edges: int = 0
    observed_cut_edges: int = 0
    retained_cut_edges: int = 0
    new_edges_during_withdrawal: int = 0
    cut_wall_triangles: int = 0
    final_grip_points: int = 0
    current_grip_points: int = 0
    grip_lost_after_capture: bool = False
    # Friction grasp only (contact model 1): the material nodes between the
    # pressed pads at the end of 'close', how many of them the pads still
    # enclose, and their median slide along the pads since then.
    bite_nodes: int = 0
    held_bite_nodes: int = 0
    grip_slip_m: float = 0.0
    # Non-cutting parts: the knife back and handle, or the scissors' blade
    # backs, wrist and shaft. Accepted up to 0.2 mm.
    max_knife_body_penetration_m: float = 0.0
    # Scissors only: the blade-back share of the overlap above.
    max_blade_body_overlap_m: float = 0.0
    peak_blade_force_n: float = 0.0
    max_node_displacement_m: float = 0.0
    degenerate_cut_wall_triangles: int = 0
    # Scissors only: every snip starts its shear after both blades squeeze the
    # tab. The loads and travel below are the weakest over all snips.
    fractures: int = 0
    fracture_left_n: float = 0.0
    fracture_right_n: float = 0.0
    two_blade_squeeze_m: float = 0.0
    snips: list[SnipRecord] = field(default_factory=list)
    edges_cut_without_fracture: int = 0
    edges_cut_outside_tab: int = 0
    # Material columns through the scene's cut box that no snip severed, and
    # those severed more than once, which leaves a sliver between the cuts:
    # both zero when the wound is one surface through the planned box.
    uncut_columns: int = 0
    multiply_cut_columns: int = 0
    # Severed skin links, both nodes outside the material, more than a diagonal
    # cell from every severed material link: a crack running along the skin
    # away from the wound.
    stray_skin_cuts: int = 0
    slit_length_m: float = 0.0
    material_components: int = 0
    # Excision only: the material pieces that hold gripped nodes, and the
    # nodes of the tumor outside the piece the grasper holds.
    gripped_pieces: int = 0
    tumor_nodes_left: int = 0
    # Material pieces counted over every simulated link, the shell around the
    # material included: a piece cut off but still tied to the rest through
    # the shell counts once with it.
    lattice_pieces: int = 0
    acceptance_passed: bool = False


class Sequence:
    def __init__(self, initial: Observation, settings: Settings = Settings(), scene: ScissorsScene | None = None,
                 rest_positions=None, inside=None):
        self.initial_neighbors = initial.neighbors.copy()
        self.initial_nodes = initial.nodes.copy()
        self.cut_baseline = initial.neighbors.copy()
        self.grip_baseline = initial.nodes.copy()
        self.tracked = np.zeros(len(initial.nodes), dtype=bool)
        self.committed_edges = np.zeros_like(initial.neighbors, dtype=bool)
        self.metrics = Metrics()
        self.settings = settings
        self.scissors = settings.cutting_tool == "scissors"
        if self.scissors:
            assert scene is not None and rest_positions is not None and inside is not None
            self.scene = scene
            # A cut is in the tab when the midpoint of its edge lies in the
            # tab's own volume or in its outer shell cells, never in the block.
            self.tab_low = np.array(scene.tab.low) - .2
            self.tab_low[scene.tab.attached] = scene.tab.low[scene.tab.attached] - 1e-3
            self.tab_high = np.array(scene.tab.high) + .2
            # The blades pinch the uncut tab at its leading end along the cut,
            # on the tab's centre line along the cut otherwise.
            self.advance = np.array(scene.advance, dtype=float)
            centre, half = (np.array(scene.tab.low) + scene.tab.high) / 2, (np.array(scene.tab.high) - scene.tab.low) / 2
            self.free_end = centre - self.advance * (np.abs(self.advance) @ half)
            self.rest_positions, self.inside = rest_positions, inside != 0
            self.stage = None
            self.stage_fractures = 0
            self.squeeze_separation = None
            self.previous_separation = None
            self.previous_cut_edges = 0
        # Contact model 1 holds tissue by pad pressure and friction alone and
        # reports no anchors: the grasp is measured on the material nodes
        # between the pads.
        self.friction_grasp = settings.contact_model == 1
        if self.friction_grasp:
            assert self.scissors, "the friction grasp is measured on the material nodes of a scissors scene"
            self.pads = load_grasper(settings.jaw_length, settings.jaw_width, settings.instrument_model).bounds
            axis = np.array(settings.jaw_axis) / np.linalg.norm(settings.jaw_axis)
            normal = np.array(settings.jaw_normal) - axis * (axis @ settings.jaw_normal)
            normal /= np.linalg.norm(normal)
            self.jaw_frame = np.stack((axis, normal, np.cross(axis, normal)))
            self.bite_ids = self.bite_local = None

    def between_pads(self, points, obs: Observation):
        """Return which of `points` lie between the pads of the solved grasper, and the points in the jaw frame.

        The jaw frame is (axis, opening normal, width) at the solved hinge. As in the native contact model, each jaw
        turns by asin(gap / (2 * jaw length)) about the width axis and carries its pad as a box in its own turned frame.
        A point lies between the pads when it lies inward of both pad faces, within the length of both pads and within
        their width.
        """
        local = (points - obs.hinge) @ self.jaw_frame.T
        sn = min(obs.jaw_gap / (2 * self.settings.jaw_length), .95)
        cs = np.sqrt(1 - sn * sn)
        enclosed = np.ones(len(points), dtype=bool)
        for side, (center, half) in zip((1, -1), self.pads.reshape(2, 2, 3)):
            along = local[:, 0] * cs + side * local[:, 1] * sn
            across = local[:, 1] * cs - side * local[:, 0] * sn
            enclosed &= ((np.abs(along - center[0]) <= half[0]) & (side * (across - center[1]) <= -half[1])
                         & (np.abs(local[:, 2] - center[2]) <= half[2]))
        return enclosed, local

    def record_step(self, stage: str, obs: Observation, knife_body_depth_m: float, blade_body_depth_m: float = 0.0):
        m = self.metrics
        m.observed_cut_edges = int(severed_edges(self.initial_neighbors, obs.neighbors))
        m.current_grip_points = obs.grip_points
        m.max_knife_body_penetration_m = max(m.max_knife_body_penetration_m, knife_body_depth_m)
        m.max_blade_body_overlap_m = max(m.max_blade_body_overlap_m, blade_body_depth_m)
        m.peak_blade_force_n = max(m.peak_blade_force_n, float(np.linalg.norm(obs.blade[3:6]) * METERS_PER_UNIT))
        m.max_node_displacement_m = max(m.max_node_displacement_m,
            float(np.linalg.norm(obs.nodes - self.initial_nodes, axis=1).max() * METERS_PER_UNIT))
        if m.grasp_acquired and self.friction_grasp:
            enclosed, local = self.between_pads(obs.nodes[self.bite_ids], obs)
            m.held_bite_nodes = int(np.count_nonzero(enclosed))
            # Slide along the pads only: the normal component follows the squeeze.
            m.grip_slip_m = float(np.median(np.linalg.norm((local - self.bite_local)[:, ::2], axis=1))
                                  * METERS_PER_UNIT)
            # The median bite node has left the pads.
            m.grip_lost_after_capture |= 2 * m.held_bite_nodes < m.bite_nodes
        elif m.grasp_acquired:
            m.grip_lost_after_capture |= obs.grip_points == 0
        if self.scissors:
            self.record_scissors(stage, obs)
        # This small attached patch has a 60 mm operating envelope. A node
        # leaving it is a failed scene, even if its floating-point state is finite.
        if m.max_node_displacement_m > .06:
            raise RuntimeError("Material left the 60 mm operating envelope; stop before exporting runaway cut geometry")

    def record_scissors(self, stage: str, obs: Observation):
        m, loads = self.metrics, obs.scissors.loads
        if stage != self.stage:
            self.stage, self.stage_fractures = stage, loads.fractures
            self.squeeze_separation = self.previous_separation = None
            # The blades pinch the uncut tab nearest its free end: the free end
            # itself for the first snip, the front of the slit afterwards, which
            # a snip following the deformed tissue can push past the cut box.
            # An uncut column counts where its material crosses the shear face
            # now, at its present position: traction can lift the tab far off
            # its rest, and the tissue around the tab's base need not reach it.
            axis = self.scene.plane_axis
            slot = 2 * axis + 1
            ids, column, _ = self.column_links(self.scene.tab.low, self.scene.tab.high)
            severed = self.column_cuts(obs.neighbors, self.scene.tab.low, self.scene.tab.high) > 0
            face = self.scene.plane + self.scene.face_offset
            crossing = (obs.nodes[ids, axis] < face) & (obs.nodes[self.initial_neighbors[ids, slot], axis] >= face)
            uncut = obs.nodes[ids[crossing & ~severed[column]]]
            self.pinch = self.free_end.copy()
            if len(uncut):
                self.pinch += self.advance * ((uncut @ self.advance).min() - self.free_end @ self.advance)
        pinch = self.pinch.copy()
        pinch[self.scene.plane_axis] = obs.scissors.hinge[self.scene.plane_axis]
        separation = edge_separation(obs.scissors, pinch)
        incising = stage.startswith("incise")
        fractured = loads.fractures > self.stage_fractures
        if incising and not any(snip.stage == stage for snip in m.snips):
            # The native law registers the squeeze within a step: count the
            # travel from the start of that step.
            if self.squeeze_separation is None and loads.is_squeezing:
                self.squeeze_separation = separation if self.previous_separation is None else self.previous_separation
            if fractured:
                # Two loaded blades first seen in the fracture step leave no
                # pressing interval: the travel is then negative or stays 0.
                squeeze = 0.0
                if self.squeeze_separation is not None and self.previous_separation is not None:
                    squeeze = (self.squeeze_separation - self.previous_separation) * METERS_PER_UNIT
                m.snips.append(SnipRecord(stage, loads.fracture_left_n, loads.fracture_right_n, squeeze))
                m.fracture_left_n = min(snip.fracture_left_n for snip in m.snips)
                m.fracture_right_n = min(snip.fracture_right_n for snip in m.snips)
                m.two_blade_squeeze_m = min(snip.two_blade_squeeze_m for snip in m.snips)
        m.fractures = loads.fractures
        # A severed edge must belong to a closing stroke that has fractured.
        new_edges = m.observed_cut_edges - self.previous_cut_edges
        if new_edges > 0 and not (incising and fractured):
            m.edges_cut_without_fracture += new_edges
        self.previous_cut_edges = m.observed_cut_edges
        self.previous_separation = separation

    def column_links(self, low, high):
        """Return the material links along the plane axis through the box from `low` to `high`, grouped in columns.

        A column is the line of nodes along the plane axis at one position across it. Its links that count are the
        material links along that axis, those with a node inside the material, whose lower node lies inside the box
        by more than 1e-3 cm. Returns the lower node of each such link, the index of its column, and the number of
        columns.
        """
        axis = self.scene.plane_axis
        across = [k for k in range(3) if k != axis]
        ids = np.flatnonzero(self.initial_neighbors[:, 2 * axis + 1] >= 0)
        other = self.initial_neighbors[ids, 2 * axis + 1]
        ids = ids[self.inside[ids] | self.inside[other]]
        rest = self.rest_positions[ids]
        in_box = np.all((rest > np.array(low) + 1e-3) & (rest < np.array(high) - 1e-3), axis=1)
        ids, rest = ids[in_box], rest[in_box]
        keys, column = np.unique(np.round(rest[:, across], 4), axis=0, return_inverse=True)
        return ids, column.reshape(-1), len(keys)

    def column_cuts(self, neighbors, low, high):
        """Return how many times each material column through the box from `low` to `high` is severed in it.

        Each severed link of a column (see column_links) counts once.
        """
        ids, column, count = self.column_links(low, high)
        return np.bincount(column, weights=neighbors[ids, 2 * self.scene.plane_axis + 1] < 0,
                           minlength=count).astype(int)

    def severed_edge_midpoints(self, neighbors):
        """Rest-space midpoints of edges severed since the start, for the positive-axis slots."""
        ids, slots = np.nonzero((self.initial_neighbors[:, 1::2] >= 0) & (neighbors[:, 1::2] < 0))
        other = self.initial_neighbors[ids, 2 * slots + 1]
        material = self.inside[ids] | self.inside[other]
        return (self.rest_positions[ids] + self.rest_positions[other]) / 2, material

    def stray_skin_cuts(self, neighbors):
        """Count the severed skin links farther than a diagonal cell from every severed material link.

        Skin links join two nodes outside the material. A wound severs the skin beside the material links it severs,
        so a severed skin link farther away is a crack running along the skin away from the wound.
        """
        midpoints, material = self.severed_edge_midpoints(neighbors)
        if np.all(material):
            return 0
        if not np.any(material):
            return int(np.count_nonzero(~material))
        distance, _ = cKDTree(midpoints[material]).query(midpoints[~material], distance_upper_bound=1.5 * SCISSORS_SPACING)
        return int(np.count_nonzero(np.isinf(distance)))

    def pieces(self, neighbors, members):
        """Return the piece of every node: nodes of the mask `members` linked through intact links among them share
        their piece."""
        ids = np.flatnonzero(members)
        rows = np.repeat(ids, 6)
        cols = neighbors[ids].reshape(-1)
        keep = (cols >= 0) & members[np.maximum(cols, 0)]
        graph = coo_matrix((np.ones(np.count_nonzero(keep)), (rows[keep], cols[keep])), shape=(len(neighbors),) * 2)
        return connected_components(graph, directed=False)[1]

    def finish_stage(self, name, obs: Observation, surface: Surface):
        m = self.metrics
        wall = surface.anchors[surface.triangles[:, 0], 3] > .5
        triangles = surface.vertices[surface.triangles[wall]]
        m.cut_wall_triangles = len(triangles)
        twice_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                            triangles[:, 2] - triangles[:, 0]), axis=1)
        m.degenerate_cut_wall_triangles = max(m.degenerate_cut_wall_triangles,
                                               int(np.count_nonzero(twice_area < 1e-8)))
        if name == "close":
            self.grip_baseline = obs.nodes.copy()
            if self.friction_grasp:
                enclosed, local = self.between_pads(obs.nodes, obs)
                self.tracked = enclosed & self.inside
                self.bite_ids = np.flatnonzero(self.tracked)
                self.bite_local = local[self.bite_ids]
                m.bite_nodes = m.held_bite_nodes = len(self.bite_ids)
                # Friction holds the bite only where tissue presses both pads.
                # At the end of the closing stage the weaker pad carries 0.18 to
                # 0.22 N on the wall and 1.8 N on the flap, while closing pads
                # first touch the tissue with at most 0.011 N: 0.05 N lies a
                # factor of four from both. Pressure on both inner faces needs
                # tissue between them, so the tip gap adds no evidence.
                m.grasp_acquired = bool(m.bite_nodes and min(obs.jaw_load_n) >= .05)
            else:
                self.tracked = obs.grip_flags != 0
                m.grasp_acquired = bool(obs.grip_points > 0 and np.any(self.tracked))
            if not m.grasp_acquired:
                raise RuntimeError("Jaws closed but no material was captured; cutting has not started")
        elif name == "lift":
            m.lift_m = float(np.median(obs.nodes[self.tracked, 1] - self.grip_baseline[self.tracked, 1]) * METERS_PER_UNIT)
            if m.lift_m < 0.001:
                raise RuntimeError(f"Measured tissue lift {m.lift_m * 1000:.3f} mm is below 1 mm")
        elif name == "blade_approach":
            m.edges_before_incision = int(severed_edges(self.initial_neighbors, obs.neighbors))
            if m.edges_before_incision:
                raise RuntimeError("Unexpected severing before incision")
            self.cut_baseline = obs.neighbors.copy()
        elif name.startswith("incise"):
            self.committed_edges = (self.cut_baseline >= 0) & (obs.neighbors < 0)
            m.cut_edges = int(severed_edges(self.cut_baseline, obs.neighbors))
            if not m.cut_edges:
                raise RuntimeError("Blade stroke completed without a confirmed cut")
        elif name == "hold":
            m.retained_cut_edges = int(np.count_nonzero(self.committed_edges[:, 1::2] & (obs.neighbors[:, 1::2] < 0)))
            m.new_edges_during_withdrawal = int(severed_edges(self.cut_baseline, obs.neighbors)) - m.cut_edges
            m.final_grip_points = obs.grip_points
            m.completed = True
            # The jaws enclose 2 to 4 mm of the bite along the pads, so a median
            # slide of more than one material cell (1 mm) takes much of it out of
            # them. Within a cell, the tracked interior nodes also follow the
            # shear of the tissue between them and the pad faces.
            held = m.grip_slip_m <= .001 if self.friction_grasp else obs.grip_points > 0
            passed = bool(m.grasp_acquired and m.lift_m >= 0.001 and m.cut_edges > 0
                          and m.cut_wall_triangles > 0
                          and m.retained_cut_edges == m.cut_edges and held
                          and m.new_edges_during_withdrawal == 0
                          and not m.grip_lost_after_capture
                          and m.degenerate_cut_wall_triangles == 0
                          and m.max_knife_body_penetration_m <= .0002
                          and m.max_node_displacement_m <= .06)
            if self.scissors:
                midpoints, material = self.severed_edge_midpoints(obs.neighbors)
                in_tab = np.all((midpoints >= self.tab_low) & (midpoints <= self.tab_high), axis=1)
                m.edges_cut_outside_tab = int(np.count_nonzero(~in_tab))
                along = midpoints[material] @ self.advance
                m.slit_length_m = float(np.ptp(along) * METERS_PER_UNIT) if len(along) else 0.0
                pieces = self.pieces(obs.neighbors, self.inside)
                m.material_components = len(np.unique(pieces[self.inside]))
                lattice = self.pieces(obs.neighbors, np.ones(len(self.inside), dtype=bool))
                m.lattice_pieces = len(np.unique(lattice[self.inside]))
                if self.scene.pieces == 2:
                    gripped = np.unique(pieces[(obs.grip_flags != 0) & self.inside])
                    m.gripped_pieces = len(gripped)
                    tumor = self.inside & (self.scene.shape.tumor_weight(self.rest_positions, 1e-3) > .5)
                    m.tumor_nodes_left = int(np.count_nonzero(tumor & ~np.isin(pieces, gripped[:1])))
                m.stray_skin_cuts = self.stray_skin_cuts(obs.neighbors)
                cuts = self.column_cuts(obs.neighbors, self.scene.cut_low, self.scene.cut_high)
                m.uncut_columns, m.multiply_cut_columns = int(np.count_nonzero(cuts == 0)), int(np.count_nonzero(cuts > 1))
                required = .5 * self.settings.start_force_n
                # The cut box sets the shortest accepted wound, so the slit
                # length is recorded only.
                passed &= bool(m.fractures >= 1 and m.edges_cut_without_fracture == 0
                               and m.fracture_left_n >= .999 * required and m.fracture_right_n >= .999 * required
                               and m.two_blade_squeeze_m >= .0005 and m.edges_cut_outside_tab == 0
                               and m.material_components == m.lattice_pieces == self.scene.pieces
                               and m.uncut_columns == 0
                               and m.multiply_cut_columns == 0 and m.stray_skin_cuts == 0
                               and (self.scene.pieces == 1 or m.gripped_pieces == 1 and m.tumor_nodes_left == 0))
            m.acceptance_passed = passed
            if not m.acceptance_passed:
                raise RuntimeError("Interaction quality failed; inspect report metrics for grip loss or slip, "
                                   "cut geometry, new cuts on withdrawal, knife-body overlap (>0.2 mm), material "
                                   "leaving the 60 mm workspace, a scissor cut without two-blade compression, a "
                                   "wound left uncut, cut twice or running along the skin, or an excision that "
                                   "leaves tumor behind, stays tied to the rest or that the grasper does not "
                                   "hold it")
