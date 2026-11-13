"""Supine liver inside the peritoneal cavity, from the imported tetrahedral model. Native cm, Y up.

The mesh is the Gmsh liver used by the laparoscopic scene. Its long axis runs left to right and its visceral
depression faces straight down. Two ligament sites on the upper surface stay pinned. The cavity shell is the same
supine torso that frames that scene.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy import ndimage

from .grid import implicit_boundary
from .instrument_geometry import load_grasper
from .native_backend import Settings


MESH_PATH = Path(__file__).parent / "assets" / "liver" / "liver3_refined_1.msh"
# Long-axis resolution of the source scene, in voxels. The resulting cell stays inside the lattice limits.
LONG_AXIS_VOXELS = 56
# The scope looks along -Z, so view-left is native -X. The lobe moves two centimetres that way from the source placement.
LOBE_SHIFT = np.array((-2.0, 0.0, 0.0))
# The tools' first pose moves one centimetre view-right, which is native +X. Later poses follow the lobe.
INSTRUMENT_START_SHIFT = np.array((1.0, 0.0, 0.0))
# Offset is the source scene's placement plus LOBE_SHIFT, in cm. The depression already faces model -Y, world down.
POSE_OFFSET = np.array((-3.0, -1.0, 6.0)) + LOBE_SHIFT
# Normalized sites in the unposed model box, and the radius of each pinned patch, in cm.
LIGAMENT_SITES = ((0.8, 0.85, 0.35), (0.35, 0.85, 0.65))
LIGAMENT_RADIUS = 2.5
# Supine peritoneal shell, centred on the world origin, in cm. The scope sits just outside the +Z opening,
# above the liver, and looks back along the cavity. The grasped end is then on the right.
TORSO_RADIUS_X = 18.0
TORSO_RADIUS_Y = 9.0
TORSO_Z_MIN = -12.0
TORSO_Z_MAX = 18.0
# The diaphragm stands this far, in cm, past the surface point furthest along the sight line.
DIAPHRAGM_INSET = 3.0
# Azimuth of the wall-texture join, on the lower wall under the liver so the lobe and the far opening cover it.
SCOPE_WALL = np.array((-5.0, -9.0, 0.0))
# Pad centre of the grasper, centimetres along the jaw from the hinge. The jaws have to open wide enough that the
# pads clear the side, which is thicker than the block the default opening was set on.
GRASP_PAD_CENTRE = float(load_grasper(Settings.jaw_length, Settings.jaw_width).bounds[0, 0])
GRASP_OPEN_GAP = 2.6
# Tip gap the closed grasper holds, in centimetres. The pads meet the side at this opening. Driving the tips shut
# carries the side down to the contact floor.
GRASP_CLOSED_GAP = 1.0


def _mesh_arrays(path):
    """Return node coordinates and linear tetrahedra from a Gmsh 4.1 ASCII mesh."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    node_at = lines.index("$Nodes")
    blocks, count, _, _ = (int(value) for value in lines[node_at + 1].split())
    index = node_at + 2
    nodes = np.empty((count, 3), dtype=np.float64)
    seen = 0
    for _ in range(blocks):
        block_count = int(lines[index].split()[3])
        index += 1 + block_count
        for _point in range(block_count):
            nodes[seen] = [float(value) for value in lines[index].split()]
            seen += 1
            index += 1
    element_at = lines.index("$Elements")
    blocks, _, _, _ = (int(value) for value in lines[element_at + 1].split())
    index = element_at + 2
    tets = []
    for _ in range(blocks):
        kind = int(lines[index].split()[2])
        block_count = int(lines[index].split()[3])
        index += 1
        if kind == 4:
            for _element in range(block_count):
                tets.append([int(value) - 1 for value in lines[index].split()[1:]])
                index += 1
        else:
            index += block_count
    return nodes, np.asarray(tets, dtype=np.int32)


def pose_points(points, pivot):
    """Turn the visceral depression straight down about the long axis, then apply the scene offset.

    The long axis is X. The pose that aimed the ligament face at the head, (x, y, z) to (x, -z, y), points the
    depression toward the feet. A further quarter turn about X, (x, y, z) to (x, z, -y), brings that face to
    world -Y. `pivot` is the model centre shared with the ligament sites.
    """
    local = np.asarray(points, dtype=np.float64) - pivot
    toward_head = np.column_stack((local[:, 0], -local[:, 2], local[:, 1]))
    turned = np.column_stack((toward_head[:, 0], toward_head[:, 2], -toward_head[:, 1]))
    return turned + pivot + POSE_OFFSET


def _solid(vertices, tets, origin, pitch, dims):
    """Return the voxels whose centres lie in any tetrahedron."""
    solid = np.zeros(dims, dtype=bool)
    scaled = (vertices - origin) / pitch
    for tet in tets:
        corners = scaled[tet]
        low = np.maximum(np.floor(corners.min(axis=0)).astype(int), 0)
        high = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 1, np.array(dims))
        if np.any(high <= low):
            continue
        axes = [np.arange(low[axis], high[axis]) + 0.5 for axis in range(3)]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        matrix = np.ones((4, 4))
        matrix[:, :3] = corners
        barycentric = np.linalg.solve(matrix.T, np.column_stack((points, np.ones(len(points)))).T).T
        inside = np.all(barycentric >= -1e-6, axis=1)
        if not np.any(inside):
            continue
        mask = inside.reshape([high[axis] - low[axis] for axis in range(3)])
        solid[low[0]:high[0], low[1]:high[1], low[2]:high[2]] |= mask
    return solid


@dataclass(frozen=True)
class Liver:
    """The posed liver as a signed distance in cm, negative inside, with two pinned ligament patches.

    `distance_field` is sampled at voxel centres. `origin` is the corner of that grid and `pitch` its spacing.
    """
    origin: np.ndarray
    pitch: float
    distance_field: np.ndarray
    low: np.ndarray
    high: np.ndarray
    ligaments: np.ndarray

    def distance(self, points):
        """Return the signed distance of `points`, shape (n, 3)."""
        coords = (np.asarray(points, dtype=np.float64) - self.origin) / self.pitch - 0.5
        return ndimage.map_coordinates(self.distance_field, coords.T, order=1, mode="nearest")

    def inside(self, positions):
        return self.distance(positions.astype(np.float64)) < 0

    def boundary(self, positions, a, b, inside):
        return implicit_boundary(self.distance, positions, a, b)

    def bed(self, positions):
        nearest = np.linalg.norm(positions[:, None, :] - self.ligaments, axis=2).min(axis=1)
        return nearest <= LIGAMENT_RADIUS


def load_liver(path=MESH_PATH, pitch=None):
    """Load the liver mesh, pose it supine, and return its signed-distance lattice."""
    nodes, tets = _mesh_arrays(path)
    model_low, model_high = nodes.min(axis=0), nodes.max(axis=0)
    pivot = (model_low + model_high) / 2
    posed = pose_points(nodes, pivot)
    span = posed.max(axis=0) - posed.min(axis=0)
    if pitch is None:
        pitch = float(np.clip(span.max() / LONG_AXIS_VOXELS, 0.1, 0.7))
    pad = 3 * pitch
    origin = posed.min(axis=0) - pad
    dims = tuple(int(value) for value in np.ceil((posed.max(axis=0) + pad - origin) / pitch))
    solid = _solid(posed, tets, origin, pitch, dims)
    # Voxel spikes a single cell wide stitch into one edge. Opening removes them; the cut rim is
    # many cells wide and stays. The remaining distance is constant on each staircase face, so one
    # cell of blur separates neighboring contour points. The zero set stays within a cell of the mesh.
    solid = ndimage.binary_opening(solid, iterations=1)
    inside = ndimage.distance_transform_edt(solid)
    outside = ndimage.distance_transform_edt(~solid)
    field = (outside - inside).astype(np.float32) * np.float32(pitch)
    field = ndimage.gaussian_filter(field, sigma=1.0).astype(np.float32)
    sites = np.array(LIGAMENT_SITES) * (model_high - model_low) + model_low
    return Liver(origin.astype(np.float32), float(pitch), field, posed.min(axis=0).astype(np.float32),
                 posed.max(axis=0).astype(np.float32), pose_points(sites, pivot).astype(np.float32))


def cavity_mesh(segments=36, slices=24):
    """Return the supine torso shell, vertices in cm and triangle faces, with normals toward the cavity."""
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    heights = np.linspace(TORSO_Z_MIN, TORSO_Z_MAX, slices)
    circle = np.stack((np.cos(angles) * TORSO_RADIUS_X, np.sin(angles) * TORSO_RADIUS_Y), axis=1)
    vertices = np.column_stack((np.tile(circle, (slices, 1)), np.repeat(heights, segments))).astype(np.float32)
    faces = []
    for slice_index in range(slices - 1):
        row = slice_index * segments
        nxt = row + segments
        for segment in range(segments):
            following = (segment + 1) % segments
            faces.append((row + segment, row + following, nxt + segment))
            faces.append((row + following, nxt + following, nxt + segment))
    # The increasing-angle order points outward. The opposite order points into the cavity.
    return vertices, np.asarray(faces, dtype=np.int32)[:, ::-1]


def _lattice_origin(liver):
    """Return the node origin and cell size build_grid uses for this liver."""
    low, high = np.asarray(liver.low, dtype=np.float32), np.asarray(liver.high, dtype=np.float32)
    cell = np.full(3, liver.pitch, dtype=np.float32)
    dims = np.ceil((high - low) / cell).astype(np.int32) + 4
    origin = (low + high) / 2 - dims.astype(np.float32) * cell / 2
    return origin, cell


class LiverCut(NamedTuple):
    """Jaw hinge, vertical shear plane, blade height and three hinge depths, in cm."""
    grasp: np.ndarray
    plane: float
    blade_y: float
    depths: np.ndarray
    cell: float


def liver_cut(liver):
    """Place a lateral grasp and three vertical bites stepping into the free side.

    The blades point toward -X and open up and down, so the shear plane is vertical and the side they can reach is
    the +X end. The grasper reaches the same way and its jaws close vertically on the caudal end, where the side is
    thin enough to fit the opening. The three bites share one lattice mid-plane a centimetre headward of that grasp
    and step toward the liver between closures.
    """
    pitch = float(liver.pitch)
    zs = np.arange(float(liver.low[2]) + 1.0, float(liver.high[2]) - 1.0, pitch)
    xs = np.linspace(float(liver.high[0]) - 1.6, float(liver.high[0]) + 0.2, 36)
    ys = np.linspace(float(liver.low[1]) - 0.3, float(liver.high[1]) + 0.3, 48)
    grid_z, grid_x, grid_y = np.meshgrid(zs, xs, ys, indexing="ij")
    samples = np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3)
    inside = liver.distance(samples).reshape(grid_z.shape) < 0
    sections = []
    for iz, z in enumerate(zs):
        chosen = None
        for ix in range(len(xs) - 1, -1, -1):
            column = ys[inside[iz, ix]]
            if len(column) < 2:
                continue
            thick = float(column.max() - column.min())
            if 1.15 <= thick <= 1.45 and float(xs[ix]) >= float(liver.high[0]) - 0.7:
                chosen = (float(xs[ix]), float(column.min()), float(column.max()))
                break
        if chosen is not None:
            sections.append((float(z), *chosen))
    sections = np.asarray(sections)
    if len(sections) < 4:
        raise ValueError("the posed liver has no free side to grasp")
    gaps = np.flatnonzero(np.diff(sections[:, 0]) > pitch * 1.5)
    begins = np.concatenate(((0,), gaps + 1))
    ends = np.concatenate((gaps + 1, (len(sections),)))
    run = max((sections[begin:end] for begin, end in zip(begins, ends)), key=len)
    if len(run) < 4:
        raise ValueError("the posed liver side is too short for three cuts")
    held = run[0]
    grasp = np.array((held[1] + GRASP_PAD_CENTRE, (held[2] + held[3]) / 2, held[0]))
    origin, cell = _lattice_origin(liver)
    target_z = float(grasp[2]) + 1.0
    layer = int(np.floor((target_z - origin[2]) / cell[2]))
    plane = float(origin[2] + (layer + 0.5) * cell[2])
    # The blade height is the middle of the outer side, plus the rise an 8.5 mm jaw lift gives that side.
    ys = np.linspace(float(liver.low[1]) - 0.2, float(liver.high[1]) + 0.2, 64)
    rest_y = None
    for x in np.linspace(float(liver.high[0]) + 0.3, float(liver.high[0]) - 1.4, 40):
        samples = np.column_stack((np.full(len(ys), x), ys, np.full(len(ys), plane)))
        column = ys[liver.distance(samples) < 0]
        if len(column) >= 2 and float(column.max() - column.min()) >= 0.9:
            rest_y = float((column.min() + column.max()) / 2)
            break
    if rest_y is None:
        raise ValueError("the vertical shear plane misses the free side")
    blade_y = rest_y + 0.53
    xs = np.linspace(float(liver.high[0]) + 0.4, float(liver.high[0]) - 2.0, 160)
    surface = liver.distance(np.column_stack((xs, np.full(len(xs), blade_y), np.full(len(xs), plane)))) < 0
    if not np.any(surface):
        raise ValueError("the vertical shear plane misses the free side")
    # The first hinge is where the blades meet the raised side. Each later hinge steps 1.2 mm further along the shaft.
    depths = float(xs[np.flatnonzero(surface)[0]]) + np.array((0.58, 0.46, 0.34))
    return LiverCut(grasp, plane, blade_y, depths, float(cell[2]))


def endoscope_pose(liver):
    """Return the scope (position, look-at) in native cm, looking back along the cavity at the other face.

    The scope is just outside the +Z opening and a little above the unshifted liver centre. The far opening then
    falls above the lobe and the grasped end is on the right.
    """
    focus = (np.asarray(liver.low) + np.asarray(liver.high)) / 2
    position = np.array((float(focus[0]), float(focus[1]) + 3.5, TORSO_Z_MAX + 8.0))
    view = focus - position
    view /= np.linalg.norm(view)
    # The lamp exposes the look-at (see LAMP_IRRADIANCE). Projecting the liver centre onto the optical axis puts
    # that exposure on the tissue. Removing LOBE_SHIFT here keeps the scope on the previous framing, so the lobe
    # sits view-left of centre.
    along = float(np.dot(focus - position, view))
    framing = position - LOBE_SHIFT
    return framing, framing + view * along


def diaphragm_mesh(liver, segments=36):
    """Return the elliptical diaphragm that closes the cavity beyond the liver.

    The wall stands DIAPHRAGM_INSET centimetres past the surface point furthest along the sight line, on the side
    facing away from the scope. Its normal points toward the scope.
    """
    nodes, tets = _mesh_arrays(MESH_PATH)
    low, high = nodes.min(axis=0), nodes.max(axis=0)
    posed = pose_points(nodes, (low + high) / 2)
    corners = (tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]])
    faces = np.sort(np.concatenate(corners), axis=1)
    key = np.dtype([("a", np.int32), ("b", np.int32), ("c", np.int32)])
    coded = np.ascontiguousarray(faces).view(key).ravel()
    unique, counts = np.unique(coded, return_counts=True)
    # A face of two tetrahedra is interior. The inset is measured from the boundary.
    boundary = unique[counts == 1].view(np.int32).reshape(-1, 3)
    surface = posed[np.unique(boundary)]
    position, look = endoscope_pose(liver)
    view = look - position
    view /= np.linalg.norm(view)
    far = surface[np.argmax((surface - position) @ view)]
    # Inward is native -Z, so the wall is the vertical section of the cavity there.
    wall_z = far[2] - DIAPHRAGM_INSET
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rim = np.column_stack((np.cos(angles) * TORSO_RADIUS_X, np.sin(angles) * TORSO_RADIUS_Y,
                           np.full(segments, wall_z)))
    vertices = np.vstack((np.array((0.0, 0.0, wall_z)), rim))
    triangles = [(0, index, index + 1) for index in range(1, segments)]
    triangles.append((0, segments, 1))
    return vertices.astype(np.float32), np.asarray(triangles, dtype=np.int32)
