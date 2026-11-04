"""Port of BackgroundGrid/GridConventions for a finite, thick tissue patch.

Native coordinates are right-handed, Y up, 1 unit = 1 cm. Explicit NumPy
dtypes below are part of the external CUDA ABI, not Genesis solver storage.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

CORNER_OFFSETS = np.array(
    ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), dtype=np.int32)
EDGE_CORNERS = np.array(
    ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
     (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)), dtype=np.int32)
ISECT_DTYPE = np.dtype([
    ("position", "<f4", 3), ("normal", "<f4", 3),
    ("a", "<i4"), ("b", "<i4"), ("t", "<f4"), ("inside_a", "<i4")])
BEND_DTYPE = np.dtype([
    ("i", "<i4"), ("j", "<i4"), ("k", "<i4"), ("theta", "<f4"),
    ("alive", "<i4"), ("padding", "<i4", 3)])
METERS_PER_UNIT = 0.01
# Proper rotation, determinant +1: native (x,y,z) -> Genesis (x,-z,y).
TO_GENESIS = np.array(((1, 0, 0), (0, 0, -1), (0, 1, 0)), dtype=np.float32)


@dataclass(frozen=True)
class Tab:
    """Box of tissue grown from one positive face of the block, in cm.

    `low` and `high` are its corners outside the block, and its `low` face along axis `attached` lies on the block's
    positive face of that axis. Faces within 1e-6 cm of a block face are snapped onto it.
    """
    low: tuple[float, float, float]
    high: tuple[float, float, float]
    attached: int


@dataclass
class TissueGrid:
    spacing: float
    cell_size: np.ndarray
    dims: tuple[int, int, int]
    origin: np.ndarray
    positions: np.ndarray
    inside: np.ndarray
    active: np.ndarray
    pinned: np.ndarray
    mass: np.ndarray
    neighbors: np.ndarray
    rest_neighbors: np.ndarray
    bends: np.ndarray
    corners: np.ndarray
    occupied: np.ndarray
    intersections: np.ndarray
    offsets: np.ndarray
    counts: np.ndarray
    surface_edges: np.ndarray
    edges: np.ndarray
    connectivity: np.ndarray


def coordinates(dims):
    z, y, x = np.indices(tuple(reversed(dims)), dtype=np.int32)
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)


def connectivity_table():
    """All 2^12 cut masks; deterministic component IDs in local-corner order."""
    labels = np.tile(np.arange(8, dtype=np.int32), (4096, 1))
    masks = np.arange(4096, dtype=np.int32)
    for _ in range(8):
        for bit, (a, b) in enumerate(EDGE_CORNERS):
            live = (masks & (1 << bit)) == 0
            minimum = np.minimum(labels[:, a], labels[:, b])
            labels[live, a] = minimum[live]
            labels[live, b] = minimum[live]
    roots = labels == np.arange(8)
    rank = np.cumsum(roots, axis=1, dtype=np.int32) - 1
    groups = np.take_along_axis(rank, labels, axis=1)
    return np.column_stack((roots.sum(axis=1), groups)).astype(np.int32)


def union_exit(points, axis, direction, lows, highs):
    """Coordinate where rays from interior points leave a union of axis-aligned boxes.

    Each ray starts at one of `points`, runs along its `axis` in the sign of `direction`, and leaves every box
    whose open interior contains its cross-section. Boxes that overlap along the ray chain into one interval.
    """
    rows = np.arange(len(points))
    within = (lows < points[:, None, :]) & (points[:, None, :] < highs)
    within[rows, :, axis] = True
    covers = within.all(axis=2)
    # Project onto u = direction * x so every ray runs toward +u.
    positive = direction[:, None] > 0
    near = np.where(positive, lows[:, axis].T, -highs[:, axis].T)
    far = np.where(positive, highs[:, axis].T, -lows[:, axis].T)
    start = direction * points[rows, axis]
    exit = np.where(covers & (near < start[:, None]) & (start[:, None] < far), far, -np.inf).max(axis=1)
    for _ in range(len(lows) - 1):
        chained = np.where(covers & (near < exit[:, None]) & (exit[:, None] < far), far, -np.inf).max(axis=1)
        exit = np.maximum(exit, chained)
    assert np.isfinite(exit).all(), "an interior point lies outside every box"
    return direction * exit


BLOCK_SIZE = (6.0, 1.4, 4.0)


@dataclass(frozen=True)
class BoxUnion:
    """Tissue made of axis-aligned boxes, in cm: the block, centred on the origin, and an optional tab.

    The tissue bed is the bottom face of the block, `half` its half extents.
    """
    lows: np.ndarray
    highs: np.ndarray
    half: np.ndarray

    @property
    def low(self):
        return self.lows.min(axis=0)

    @property
    def high(self):
        return self.highs.max(axis=0)

    def inside(self, positions):
        return ((self.lows < positions[:, None, :]) & (positions[:, None, :] < self.highs)).all(axis=2).any(axis=1)

    def boundary(self, positions, a, b, inside):
        """Return the fraction from node `a` to node `b` at which each crossing lattice edge leaves the tissue, and the
        outward normal there."""
        # Analytic box intersection: interpolating max-form SDF values near a corner
        # does not in general land on the zero set, even when the edge changes sign.
        axis = np.abs(positions[b] - positions[a]).argmax(axis=1)
        interior = np.where(inside[a] != 0, a, b)
        outside = np.where(inside[a] != 0, b, a)
        direction = np.sign(positions[outside, axis] - positions[interior, axis])
        exit = union_exit(positions[interior], axis, direction, self.lows, self.highs)
        t = (exit - positions[a, axis]) / (positions[b, axis] - positions[a, axis])
        # An axis-aligned edge with an interior cross-section leaves through a face
        # normal to its own axis.
        normals = np.zeros((len(a), 3), dtype=np.float32)
        normals[np.arange(len(a)), axis] = direction
        return t, normals

    def bed(self, positions):
        # A narrow basal attachment represents the tissue bed; the upper flap is mobile.
        return positions[:, 1] <= -self.half[1]


def implicit_boundary(distance, positions, a, b):
    """Return the fraction from node `a` to node `b` where `distance` crosses zero on each lattice edge, and the outward
    unit normal there.

    `distance` maps points of shape (n, 3) to signed distances, negative inside the tissue. Each edge has one end on
    each side; 40 bisections place the crossing far below the float32 resolution of the lattice. The normal is the
    central-difference gradient of `distance`.
    """
    pa, pb = positions[a].astype(np.float64), positions[b].astype(np.float64)
    lo, hi = np.zeros(len(a)), np.ones(len(a))
    sign_a = np.sign(distance(pa))
    for _ in range(40):
        mid = (lo + hi) / 2
        same = np.sign(distance(pa + (pb - pa) * mid[:, None])) == sign_a
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
    t = (lo + hi) / 2
    hit = pa + (pb - pa) * t[:, None]
    step = 1e-4
    gradient = np.stack([distance(hit + step * unit) - distance(hit - step * unit) for unit in np.eye(3)], axis=1)
    return t, (gradient / np.linalg.norm(gradient, axis=1, keepdims=True)).astype(np.float32)


def build_grid(spacing=0.6, size=BLOCK_SIZE, density=1000.0, vertical_spacing=None, tab=None, shape=None):
    """Build the lattice of a tissue: the block of `size` with an optional `tab`, or else an implicit `shape`.

    A shape gives its bounds `low` and `high`, `inside(positions)`, `boundary(positions, a, b, inside)` (the crossing
    fraction and outward normal on lattice edges from `a` to `b`) and `bed(positions)`, the nodes pinned to the tissue
    bed where they are active.
    """
    if not np.isfinite(spacing) or not 0.1 <= spacing <= 0.7:
        raise ValueError("spacing must be 0.1..0.7 native units (1..7 mm)")
    cell_size = np.full(3, spacing, dtype=np.float32)
    if vertical_spacing is not None:
        if not np.isfinite(vertical_spacing) or not .01 <= vertical_spacing <= spacing:
            raise ValueError("vertical spacing must be 0.01..spacing native units")
        cell_size[1] = vertical_spacing
    if shape is None:
        half = np.array(size, dtype=np.float32) / 2
        lows, highs = [-half], [half]
        if tab is not None:
            low, high = np.array(tab.low, dtype=np.float32), np.array(tab.high, dtype=np.float32)
            if not (np.isfinite((low, high)).all() and tab.attached in (0, 1, 2)
                    and np.all(high - low >= 2 * cell_size - 1e-6) and abs(low[tab.attached] - half[tab.attached]) < 1e-6):
                raise ValueError("tab must span two cells per axis and grow from a positive face of the block")
            # Faces flush with block faces reuse the block's exact values, so both
            # surfaces stay on one node layer. The tab box also reaches one cell into
            # the block: nodes on a face shared by two boxes lie outside both and
            # would detach the tab.
            for bound, face in ((low, -half), (low, half), (high, -half), (high, half)):
                flush = np.abs(bound - face) < 1e-6
                bound[flush] = face[flush]
            low[tab.attached] = half[tab.attached] - cell_size[tab.attached]
            lows.append(low)
            highs.append(high)
        shape = BoxUnion(np.array(lows), np.array(highs), half)
    low, high = shape.low, shape.high
    dims = tuple(int(v) for v in np.ceil((high - low) / cell_size).astype(int) + 4)
    origin = (low + high) / 2 - np.array(dims, dtype=np.float32) * cell_size / 2
    nc = tuple(d + 1 for d in dims)
    ijk = coordinates(nc)
    strides = np.array((1, nc[0], nc[0] * nc[1]), dtype=np.int32)
    positions = np.ascontiguousarray(origin + ijk * cell_size, dtype=np.float32)
    inside = shape.inside(positions).astype(np.int32)
    voxel_ijk = coordinates(dims)
    corners = ((voxel_ijk[:, None, :] + CORNER_OFFSETS) @ strides).astype(np.int32)
    occupied = inside[corners].any(axis=1).astype(np.int32)
    active = np.zeros(len(positions), dtype=np.int32)
    active[corners[occupied != 0]] = 1
    pinned = (shape.bed(positions) & (active != 0)).astype(np.int32)

    ca = corners[:, EDGE_CORNERS[:, 0]]
    cb = corners[:, EDGE_CORNERS[:, 1]]
    crossing = inside[ca] != inside[cb]
    counts = crossing.sum(axis=1, dtype=np.int32)
    offsets = np.cumsum(counts, dtype=np.int32) - counts
    a, b = ca[crossing], cb[crossing]
    t, normals = shape.boundary(positions, a, b, inside)
    hit = positions[a] + (positions[b] - positions[a]) * t[:, None]
    intersections = np.empty(len(hit), dtype=ISECT_DTYPE)
    intersections["position"], intersections["normal"] = hit, normals
    intersections["a"], intersections["b"] = a, b
    intersections["t"], intersections["inside_a"] = t, inside[a]

    edge_parts, surface_parts = [], []
    neighbors = np.full((len(positions), 6), -1, dtype=np.int32)
    rest = np.zeros((len(positions), 6, 3), dtype=np.float32)
    for axis in range(3):
        valid = ijk[:, axis] < dims[axis]
        a = np.flatnonzero(valid)
        b = a + strides[axis]
        live = (active[a] != 0) & (active[b] != 0)
        neighbors[a[live], 2 * axis + 1] = b[live]
        neighbors[b[live], 2 * axis] = a[live]
        rest[a[live], 2 * axis + 1, axis] = cell_size[axis]
        rest[b[live], 2 * axis, axis] = -cell_size[axis]
        # Four incident voxels are needed for a complete dual-contouring quad.
        for other in range(3):
            if other != axis:
                valid &= (ijk[:, other] > 0) & (ijk[:, other] < dims[other])
        a = np.flatnonzero(valid)
        edge = np.column_stack((np.full(len(a), axis), ijk[a])).astype(np.int32)
        edge_parts.append(edge)
        cross = inside[a] != inside[a + strides[axis]]
        surface = np.zeros((np.count_nonzero(cross), 8), dtype=np.int32)
        surface[:, :4] = edge[cross]
        surface[:, 4] = inside[a[cross]]
        surface_parts.append(surface)

    pairs = np.array(list(combinations(range(6), 2)))
    bends = np.zeros((len(positions), 15), dtype=BEND_DTYPE)
    bends["i"] = np.arange(len(positions))[:, None]
    bends["j"], bends["k"] = neighbors[:, pairs[:, 0]], neighbors[:, pairs[:, 1]]
    bends["alive"] = (bends["j"] >= 0) & (bends["k"] >= 0)
    opposite = pairs[:, 0] // 2 == pairs[:, 1] // 2
    bends["theta"] = np.where(opposite, np.pi, np.pi / 2)
    # Lump one quarter of each physical tetrahedron's mass onto its vertices.
    # Native mechanics uses the all-active voxel envelope, including its shell.
    binary_corners = corners[:, (0, 1, 3, 2, 4, 5, 7, 6)]
    supported = np.all(active[binary_corners] != 0, axis=1)
    tets = np.array(((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7),
                     (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)))
    ids = binary_corners[supported][:, tets].reshape(-1)
    mass = (np.bincount(ids, minlength=len(positions)) * density *
            np.prod(cell_size * METERS_PER_UNIT) / 24).astype(np.float32)
    assert np.all(mass[active != 0] > 0)
    return TissueGrid(spacing, cell_size, dims, origin, positions, inside, active, pinned, mass,
                      neighbors, rest, bends, corners, occupied, intersections, offsets,
                      counts, np.concatenate(surface_parts), np.concatenate(edge_parts), connectivity_table())


def to_genesis(points):
    return points @ TO_GENESIS.T * METERS_PER_UNIT
