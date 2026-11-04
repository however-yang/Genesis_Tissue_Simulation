"""Find which tets make k_RestrictMaterialDirection zero the material direction of the held piece in a dump of
debug_material_dump.py: for each live tet with a movable node, the volume ratio along the PCG direction, as the cubic
J(a) of the step length a, against the floor min(0.15, J(0)) over a in [0, 1].

Usage: python restriction_cause.py <run folder> <frame>   (piece and links from <run folder>/hold.npz)

A tet blocks when J(a) drops below its floor; it does so at once (a near 0) when it starts at the floor and its
first-order rate J'(0) is negative, or later along the step when only the higher-order terms carry it there.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from endoscopic_tissue import scenario  # noqa: E402

FLOOR = .15
run, frame = Path(sys.argv[1]), int(sys.argv[2])
grid = scenario.KIDNEY_SCENE.build_grid(1050)
rest = grid.positions.astype(np.float64)
hold = np.load(run / "hold.npz")
links = hold["neighbors"]
active = grid.active != 0
source = np.repeat(np.arange(len(links)), 6)
target = links.reshape(-1)
keep = (target >= 0) & active[source]
_, labels = connected_components(coo_matrix((np.ones(keep.sum()), (source[keep], target[keep])), shape=(len(links),) * 2),
                                 directed=False)
gripped = hold["grip_flags"] != 0
piece = active & (labels == np.bincount(labels[gripped]).argmax())
binary_corners = grid.corners[:, (0, 1, 3, 2, 4, 5, 7, 6)]
supported = np.all(grid.active[binary_corners] != 0, axis=1)
kuhn = np.array(((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)))
tets = binary_corners[supported][:, kuhn].reshape(-1, 4)
tets = tets[piece[tets].all(axis=1)]
alive = np.ones(len(tets), dtype=bool)
for a in range(4):
    for b in range(a + 1, 4):
        axis_link = np.count_nonzero(np.abs(rest[tets[:, b]] - rest[tets[:, a]]) > 1e-5, axis=1) == 1
        alive &= ~axis_link | (links[tets[:, a]] == tets[:, b, None]).any(axis=1)
tets = tets[alive]
path = run.parent / f"{run.name}-dump" / f"crease-{frame:04d}.bin"
n = np.fromfile(path, dtype=np.int32, count=1)[0]
positions = np.fromfile(path, dtype=np.float32, offset=8, count=3 * n).reshape(n, 3).astype(np.float64)
flags = np.fromfile(path, dtype=np.int32, offset=8 + 4 * 18 * n, count=n)
directions = np.fromfile(path, dtype=np.float32, offset=8 + 4 * 19 * n, count=6 * n)
free, used = directions[:3 * n].reshape(n, 3).astype(np.float64), directions[3 * n:].reshape(n, 3).astype(np.float64)
inverse_volume = 1 / np.linalg.det(rest[tets[:, 1:]] - rest[tets[:, :1]])
e = positions[tets[:, 1:]] - positions[tets[:, :1]]
d = free[tets[:, 1:]] - free[tets[:, :1]]
# J(a) = det(e + a d) / det(rest edges), expanded in powers of a.
j0 = np.einsum("ti,ti->t", e[:, 0], np.cross(e[:, 1], e[:, 2]))
j1 = (np.einsum("ti,ti->t", d[:, 0], np.cross(e[:, 1], e[:, 2])) + np.einsum("ti,ti->t", e[:, 0], np.cross(d[:, 1], e[:, 2]))
      + np.einsum("ti,ti->t", e[:, 0], np.cross(e[:, 1], d[:, 2])))
j2 = (np.einsum("ti,ti->t", d[:, 0], np.cross(d[:, 1], e[:, 2])) + np.einsum("ti,ti->t", d[:, 0], np.cross(e[:, 1], d[:, 2]))
      + np.einsum("ti,ti->t", e[:, 0], np.cross(d[:, 1], d[:, 2])))
j3 = np.einsum("ti,ti->t", d[:, 0], np.cross(d[:, 1], d[:, 2]))
j0, j1, j2, j3 = (v * inverse_volume for v in (j0, j1, j2, j3))
steps = np.linspace(0, 1, 201)
path_j = j0[:, None] + steps * (j1[:, None] + steps * (j2[:, None] + steps * j3[:, None]))
floor = np.minimum(FLOOR, j0)
blocking = (path_j < floor[:, None] - 1e-7).any(axis=1)
first = np.where(blocking, np.argmax(path_j < floor[:, None] - 1e-7, axis=1), -1)
movable = ((flags & 16) != 0) & ((flags & 2) == 0)
blocking &= movable[tets].any(axis=1)
at_floor = j0 < FLOOR + 5e-4
offset = (rest[tets].mean(axis=1) - scenario.KIDNEY.tumor_center) * 10
crease = (np.abs(offset[:, 0]) < 2) & (offset[:, 2] > 2)
print(f"{run.name} frame {frame}: {len(tets)} live tets, {np.count_nonzero(at_floor)} at the floor; blocking tets {blocking.sum()} "
      f"({np.count_nonzero(blocking & crease)} in the +z crease)")
for label, group in (("start at the floor, first-order compression", blocking & at_floor & (j1 < 0)),
                     ("start at the floor, higher-order dip", blocking & at_floor & (j1 >= 0)),
                     ("start above the floor", blocking & ~at_floor)):
    if group.any():
        print(f"  {label}: {group.sum()} (crease {np.count_nonzero(group & crease)}), J(0) p50 {np.median(j0[group]):.3f}, "
              f"J'(0) p50 {np.median(j1[group]):.3f}, first crossing at a p50 {np.median(steps[first[group]]):.3f}, "
              f"largest node step {np.abs(d[group]).max() * 1e4:.0f} um")
# Nodes the restriction zeroed, and whether a blocking tet of each is at the floor.
zeroed = movable & piece & (np.linalg.norm(free, axis=1) > 1e-5) & (np.linalg.norm(used, axis=1) < .01 * np.linalg.norm(free, axis=1))
node_floor = np.zeros(n, dtype=bool)
np.logical_or.at(node_floor, tets[blocking & at_floor].reshape(-1), True)
node_above = np.zeros(n, dtype=bool)
np.logical_or.at(node_above, tets[blocking & ~at_floor].reshape(-1), True)
print(f"  zeroed nodes {zeroed.sum()}: touching a blocking floor tet {np.count_nonzero(zeroed & node_floor)}, "
      f"only blocking tets above the floor {np.count_nonzero(zeroed & ~node_floor & node_above)}, "
      f"contact or anchor witnesses among their neighbours' tets: n/a")
