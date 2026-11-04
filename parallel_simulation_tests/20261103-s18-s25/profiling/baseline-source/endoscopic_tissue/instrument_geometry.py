"""One hinge frame for the visible grasper and its native contact pads."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


DAVINCI_ASSETS = Path(__file__).parent / "assets/instruments/davinci/meshes/classic/tool"


@dataclass(frozen=True)
class GrasperGeometry:
    vertices: tuple[np.ndarray, np.ndarray]
    faces: tuple[np.ndarray, np.ndarray]
    pad_masks: tuple[np.ndarray, np.ndarray]
    bounds: np.ndarray
    length: float
    shaft_length: float = 4.0
    shaft_radius: float = 0.1

    def posed_jaws(self, hinge, axis, normal, gap):
        sn = np.clip(gap / (2 * self.length), 0, .95)
        cs = np.sqrt(1 - sn * sn)
        width = np.cross(axis, normal)
        result = []
        for side, vertices in zip((1, -1), self.vertices):
            direction = axis * cs + side * normal * sn
            opening = normal * cs - side * axis * sn
            result.append(hinge + vertices @ np.stack((direction, opening, width)))
        return tuple(result)


def load_grasper(length, width, model="davinci"):
    if model == "davinci":
        mesh = trimesh.load(DAVINCI_ASSETS / "gripper_cadiere_8mm.stl", force="mesh", process=True)
        # The source xacro rotates the jaw by -pi/2 about Z and offsets its
        # lateral mounting plane by 1.95 mm. Local X points toward the tip.
        upper = np.column_stack((mesh.vertices[:, 1], -mesh.vertices[:, 0],
                                 mesh.vertices[:, 2] + .00195)) * 100
        upper *= (length / 1.99, width / .52, width / .52)
        lower = upper * (1, -1, -1)
        vertices = (upper, lower)
        masks = tuple(v[:, 0] >= length * (.55 / 1.99) - 1e-7 for v in vertices)
        bounds = []
        for v, mask in zip(vertices, masks):
            low, high = v[mask].min(axis=0), v[mask].max(axis=0)
            bounds.append(np.concatenate(((low + high) / 2, (high - low) / 2)))
        return GrasperGeometry(vertices, (mesh.faces, mesh.faces), masks,
                               np.array(bounds, dtype=np.float32), length,
                               shaft_radius=.418 * width / .52)
    if model != "legacy":
        raise ValueError(f"Unknown instrument model: {model}")
    assets = Path(__file__).parent / "assets/instruments"
    sources = tuple(trimesh.load(assets / name, force="mesh", process=False) for name in
                    ("Haptic_grasper_jaws_up.obj", "Haptic_grasper_jaws_down.obj"))
    # OBJ opening is X, width is Y and the tip points along -Z. Both halves
    # share the measured hinge; independently centering them destroys the bite.
    vertices = tuple(np.column_stack((-17.5 - mesh.vertices[:, 2], mesh.vertices[:, 0],
                                     9.0 - mesh.vertices[:, 1])) * (length / 25.5) for mesh in sources)
    width_scale = width / (2 * max(np.abs(v[:, 2]).max() for v in vertices))
    for v in vertices:
        v[:, 2] *= width_scale
    masks = tuple(mesh.vertices[:, 2] <= -25.5 + 1e-4 for mesh in sources)
    bounds = []
    for v, mask in zip(vertices, masks):
        low, high = v[mask].min(axis=0), v[mask].max(axis=0)
        bounds.extend(((low + high) / 2, (high - low) / 2))
    return GrasperGeometry(vertices, tuple(mesh.faces for mesh in sources), masks,
                           np.ascontiguousarray(bounds, dtype=np.float32).reshape(2, 6), length)


@dataclass(frozen=True)
class KnifeGeometry:
    boxes: np.ndarray
    handle_a: np.ndarray
    handle_b: np.ndarray
    handle_radius: float

    def blade_mesh(self, center, axis=(0, 0, 1)):
        box = self.boxes[0]
        mesh = trimesh.creation.box(extents=2 * box[3:6])
        mesh.vertices = (mesh.vertices + box[:3]) @ knife_frame(axis) + center
        return mesh

    def handle_points(self, center, axis=(0, 0, 1)):
        local = np.stack((self.handle_a, self.handle_b)) @ knife_frame((0, 0, 1)).T
        return local @ knife_frame(axis) + center


def knife_frame(axis):
    axis = np.array(axis)
    length = np.linalg.norm(axis)
    if not np.isfinite(axis).all() or length < 1e-8:
        raise ValueError("Knife axis must be finite and nonzero")
    along = axis / length
    normal = np.array((0, 1, 0)) - along * along[1]
    if np.linalg.norm(normal) < 1e-6:
        normal = np.array((1, 0, 0))
    normal = normal / np.linalg.norm(normal)
    return np.stack((along, normal, np.cross(along, normal)))


def knife_geometry(length, model="davinci"):
    # The sharpened edge extends 2 mm beyond the blunt back at both ends.
    # Use this same finite blade shape for rendering and contact: a full-length
    # rectangular back otherwise wedges into the uncut ends of the incision.
    if model not in ("davinci", "legacy"):
        raise ValueError(f"Unknown instrument model: {model}")
    half = length / 2 - .2
    height, radius = (.65, .418) if model == "davinci" else (.28, .09)
    a = np.array((0, height, -half))
    b = a + (4, 3, -5)
    frame = np.array(((0, 0, 1), (0, 1, 0), (-1, 0, 0)))
    along = (b - a) / np.linalg.norm(b - a)
    normal = np.cross(along, (1, 0, 0))
    normal /= np.linalg.norm(normal)
    # The pitch clevis extends 1.5 mm toward the tip beyond the wrist pivot.
    body_start = a - .16 * along if model == "davinci" else a
    boxes = np.array(((0, height / 2 + .02, 0, half, height / 2, .025, 1, 0, 0, 0, 1, 0),
                      (*(((body_start + b) / 2) @ frame.T), np.linalg.norm(b - body_start) / 2, radius, radius,
                       *(along @ frame.T), *(normal @ frame.T))), dtype=np.float32)
    return KnifeGeometry(boxes, a, b, radius)


def load_davinci_wrist():
    """Return original wrist meshes in the jaw hinge frame, in centimetres."""
    roll = trimesh.load(DAVINCI_ASSETS / "wrist_8mm/roll.stl", force="mesh", process=True)
    pitch = trimesh.load(DAVINCI_ASSETS / "wrist_8mm/pitch.stl", force="mesh", process=True)
    yaw = trimesh.load(DAVINCI_ASSETS / "wrist_8mm/yaw.stl", force="mesh", process=True)
    pitch_frame = trimesh.transformations.euler_matrix(0, -np.pi / 2, 0)[:3, :3]
    pitch_visual = trimesh.transformations.euler_matrix(np.pi / 2, 0, np.pi)[:3, :3]
    pitch.vertices = pitch.vertices @ pitch_visual.T @ pitch_frame.T + (0, 0, .0091)
    for mesh in (roll, pitch):
        mesh.vertices = (mesh.vertices[:, (2, 0, 1)] - (.0182, 0, 0)) * 100
    yaw.vertices = yaw.vertices * (100, -100, -100)
    return roll, pitch, yaw


@dataclass(frozen=True)
class ScissorsGeometry:
    jaws: GrasperGeometry
    edges: np.ndarray
    backs: np.ndarray
    max_gap: float


# Sharpened span of the 179 blades along the tip axis (cm from the hinge). The
# root boss is blunt; beyond this span the tips curve strongly about the hinge axis.
SCISSORS_EDGE_SPAN = (.4, 1.25)
# The back box leaves out the bevel within this distance of the fitted edge.
SCISSORS_BACK_CLEARANCE = .03
# The source wrist protrudes 1.5 mm beyond the hinge. Its contact and overlap
# envelope is a conservative capsule from there back along the shaft, with the
# same 8.36 mm diameter as the visible assembly.
SCISSORS_WRIST_FRONT = .15


def load_scissors():
    """Load the dVRK 179 Monopolar Curved Scissors blades in the hinge frame, in centimetres.

    Local coordinates are (tip, opening, width) about the hinge, closed. The first blade, `tip_179_2`, opens toward
    +opening and carries its sharp side at -opening; the second, `tip_179_1`, is its mirror. Along
    `SCISSORS_EDGE_SPAN`, each cutting edge takes its opening coordinate from a straight least-squares fit through the
    sharpest vertex of every slice, and its width coordinate from the mean contact face between the two blades. Both
    edges thus sweep one flat shear plane as the blades turn about the width axis. Each blade back is an oriented box
    along its edge on the blunt side, `SCISSORS_BACK_CLEARANCE` away from the edge, spanning the blade's thickness on
    its own side of the shear plane. These straight shapes
    approximate the curvature of the blades about the hinge axis. The maximum tip gap follows the 80 degree jaw limit
    of the source Xacro.
    """
    directory = Path(__file__).parent / "assets/instruments/dvrk_scissors/meshes/instruments/tip/179"
    meshes = tuple(trimesh.load(directory / f"tip_179_{side}.obj", force="mesh", process=True)
                   for side in (2, 1))
    # The Xacro rotates each blade by pi about its joint axis and moves the
    # hinge 13.05 mm along the tip. The source mimic signs open the sharp sides
    # away from each other, so the opening sign follows the bevels instead.
    vertices = tuple(np.column_stack((1.305 - mesh.vertices[:, 1] * .1,
                                     mesh.vertices[:, 0] * .1, mesh.vertices[:, 2] * .1))
                     for mesh in meshes)
    spans = tuple(v[(v[:, 0] >= SCISSORS_EDGE_SPAN[0]) & (v[:, 0] <= SCISSORS_EDGE_SPAN[1])] for v in vertices)
    slices = tuple(np.round(span[:, 0] / .025).astype(int) for span in spans)
    common = np.intersect1d(*slices)
    # The first blade lies on the +width side of the second.
    face = [(spans[0][slices[0] == k, 2].min() + spans[1][slices[1] == k, 2].max()) / 2 for k in common]
    # A flat shear face at the mean width. The material grid is 2 mm across the
    # face, and a face following the curvature runs within 0.4 mm of a node
    # layer near the tips, where closing blades push the tissue: a layer
    # crossing it gets severed on both sides and frees its shell nodes.
    face_width = np.mean(face)
    ends = np.array(SCISSORS_EDGE_SPAN)
    edges, backs = [], []
    for toward, span, slice_ids in zip((-1, 1), spans, slices):
        order = np.lexsort((-toward * span[:, 1], slice_ids))
        first = np.r_[True, slice_ids[order][1:] != slice_ids[order][:-1]]
        sharp = span[order][first]
        opening_fit = np.polyfit(sharp[:, 0], sharp[:, 1], 1)
        edges.append(np.column_stack((ends, np.polyval(opening_fit, ends), np.full(2, face_width))))
        along = edges[-1][1] - edges[-1][0]
        along /= np.linalg.norm(along)
        blunt = np.array((0, -toward, 0)) - along * -toward * along[1]
        blunt /= np.linalg.norm(blunt)
        frame = np.stack((along, blunt, np.cross(along, blunt)))
        local = (span - edges[-1][0]) @ frame.T
        # Local +z points from the shear face into this blade for both blades.
        body = local[(local[:, 1] >= SCISSORS_BACK_CLEARANCE) & (local[:, 2] >= 0)]
        low, high = body.min(axis=0), body.max(axis=0)
        low[1:] = SCISSORS_BACK_CLEARANCE, 0
        backs.append(np.concatenate((edges[-1][0] + ((low + high) / 2) @ frame, (high - low) / 2,
                                     frame[0], frame[1])))
    masks = tuple(v[:, 0] >= .3 for v in vertices)
    bounds = np.array([np.r_[v[m].mean(axis=0), np.ptp(v[m], axis=0) / 2]
                       for v, m in zip(vertices, masks)], dtype=np.float32)
    jaws = GrasperGeometry(vertices, tuple(mesh.faces for mesh in meshes), masks,
                           bounds, 1.305, shaft_length=6.0, shaft_radius=.418)
    max_gap = 2 * jaws.length * np.sin(1.39626 / 2)
    return ScissorsGeometry(jaws, np.array(edges, dtype=np.float32), np.array(backs, dtype=np.float32), max_gap)


def scissors_body_depths(surface_vertices, triangles, geometry, hinge, axis, normal, gap):
    """Sample reconstructed triangles against the scissors' non-cutting parts, in cm.

    Returns the deepest overlap of tissue with the blade backs (`geometry.backs`) and, separately, with the capsule
    enclosing the wrist and shaft. Both are in native contact, and both are zero when clear.
    """
    width = np.cross(axis, normal)
    orientations = [np.stack((box[6:9], box[9:12], np.cross(box[6:9], box[9:12]))) for box in geometry.backs]
    # Only the samples that can lie in a part reach the exact tests below, which thus find the deepest overlap of all
    # samples. In the hinge frame (axis, normal, width) one box bounds both backs at any opening, as they turn about the
    # width axis: the in-plane radius of their farthest corner, and their span along that axis. Another box bounds the
    # capsule, whose 6 cm shaft would make one sphere around all parts cover the whole block. A sample is a vertex or a
    # convex combination of its triangle's vertices, so a triangle whose vertices all lie beyond one face of a box has
    # no sample in it. The 0.1 mm margin exceeds the float32 rounding of the samples many times over.
    signs = (np.arange(8)[:, None] >> np.arange(3) & 1) * 2 - 1
    extremes = np.concatenate([box[:3] + (signs * box[3:6]) @ orientation
                               for box, orientation in zip(geometry.backs, orientations)])
    reach = np.linalg.norm(extremes[:, :2], axis=1).max() + .01
    radius = geometry.jaws.shaft_radius + .01
    lows = np.array(((-reach, -reach, extremes[:, 2].min() - .01),
                     (-geometry.jaws.shaft_length - radius, -radius, -radius)))[..., None]
    highs = np.array(((reach, reach, extremes[:, 2].max() + .01),
                      (SCISSORS_WRIST_FRONT + radius, radius, radius)))[..., None]
    # Coordinates run along the rows, so every comparison below is a contiguous sweep.
    coords = np.stack((axis, normal, width)) @ (surface_vertices - hinge).T
    corners = tuple(coords[:, triangles[:, k]] for k in range(3))
    lowest = np.minimum(np.minimum(*corners[:2]), corners[2])
    highest = np.maximum(np.maximum(*corners[:2]), corners[2])
    near_vertices = np.zeros(len(surface_vertices), dtype=bool)
    near_triangles = np.zeros(len(triangles), dtype=bool)
    for low, high in zip(lows, highs):
        near_vertices |= ((coords >= low) & (coords <= high)).all(axis=0)
        near_triangles |= ((highest >= low) & (lowest <= high)).all(axis=0)
    # numpy multiplies a single row through other BLAS kernels than many rows, which round differently: two samples
    # always reach the tests, keeping the result bit-identical to the tests over all samples.
    near_vertices[:2] = True
    v = surface_vertices[triangles[near_triangles]]
    samples = np.concatenate((surface_vertices[near_vertices], v.mean(axis=1),
                              (v[:, 0] + v[:, 1]) / 2, (v[:, 1] + v[:, 2]) / 2,
                              (v[:, 2] + v[:, 0]) / 2))
    relative = samples - hinge
    sn = np.clip(gap / (2 * geometry.jaws.length), 0, .95)
    cs = np.sqrt(1 - sn * sn)
    body = 0.0
    for side, box, orientation in zip((1, -1), geometry.backs, orientations):
        frame = np.stack((axis * cs + side * normal * sn,
                          normal * cs - side * axis * sn, width))
        local = relative @ frame.T - box[:3]
        clearance = box[3:6] - np.abs(local @ orientation.T)
        body = max(body, clearance.min(axis=1).max())
    along = np.clip(relative @ axis, -geometry.jaws.shaft_length, SCISSORS_WRIST_FRONT)
    radial = np.linalg.norm(relative - along[:, None] * axis, axis=1)
    wrist = geometry.jaws.shaft_radius - radial.min()
    return max(0.0, body), max(0.0, wrist)
