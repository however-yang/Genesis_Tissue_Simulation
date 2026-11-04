"""Emulate, on a dump of debug_material_dump.py, what the volume-floor treatment of the material direction does to the
held piece: the tangent-cone projection of k_ProjectRecoveryDirection (Jacobi passes that average each node's
corrections, as the solver runs it, or Gauss-Seidel passes tet after tet), then the node-by-node restriction of
k_RestrictMaterialDirection over eight passes. Reports the floor tets the projection leaves compressing and the nodes
the restriction then zeroes.

Usage: python projection_emulation.py <run folder> <frame>   (piece and links from <run folder>/hold.npz)

The projection weighs each node's correction by its inverse lumped stiffness; this emulation uses equal weights.
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
# Optional: the restriction's floor as a fraction of FLOOR (1 = the solver's), and the projection's target rate.
MARGIN = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
TARGET = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
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
x = np.fromfile(path, dtype=np.float32, offset=8, count=3 * n).reshape(n, 3).astype(np.float64)
flags = np.fromfile(path, dtype=np.int32, offset=8 + 4 * 18 * n, count=n)
free = np.fromfile(path, dtype=np.float32, offset=8 + 4 * 19 * n, count=3 * n).reshape(n, 3).astype(np.float64)
fixed = (flags & 2) != 0
inverse_volume = 1 / np.linalg.det(rest[tets[:, 1:]] - rest[tets[:, :1]])


def volume_path(direction, subset=slice(None)):
    """Return J(0), J'(0), and the cubic coefficients of J along x + a*direction for the tets of subset."""
    t = tets[subset]
    e = x[t[:, 1:]] - x[t[:, :1]]
    d = direction[t[:, 1:]] - direction[t[:, :1]]
    tri = lambda a, b, c: np.einsum("ti,ti->t", a, np.cross(b, c))
    j0 = tri(e[:, 0], e[:, 1], e[:, 2])
    j1 = tri(d[:, 0], e[:, 1], e[:, 2]) + tri(e[:, 0], d[:, 1], e[:, 2]) + tri(e[:, 0], e[:, 1], d[:, 2])
    j2 = tri(d[:, 0], d[:, 1], e[:, 2]) + tri(d[:, 0], e[:, 1], d[:, 2]) + tri(e[:, 0], d[:, 1], d[:, 2])
    j3 = tri(d[:, 0], d[:, 1], d[:, 2])
    s = inverse_volume[subset]
    return j0 * s, j1 * s, j2 * s, j3 * s


def volume_gradient(subset):
    """Return dJ/dx of each node of the tets of subset, shape (m, 4, 3)."""
    t = tets[subset]
    e = x[t[:, 1:]] - x[t[:, :1]]
    g = np.stack([np.cross(e[:, 1], e[:, 2]), np.cross(e[:, 2], e[:, 0]), np.cross(e[:, 0], e[:, 1])], axis=1)
    g = g * inverse_volume[subset][:, None, None]
    return np.concatenate([-g.sum(axis=1, keepdims=True), g], axis=1)


def restrict(direction):
    """Apply eight passes of the node-by-node restriction; return the restricted direction."""
    direction = direction.copy()
    steps = np.linspace(0, 1, 101)
    for _ in range(8):
        j0, j1, j2, j3 = volume_path(direction)
        floor = np.minimum(MARGIN * FLOOR, j0)
        path_j = j0[:, None] + steps * (j1[:, None] + steps * (j2[:, None] + steps * j3[:, None]))
        bad = (path_j < floor[:, None] - 1e-9)
        lo = np.where(bad.any(axis=1), steps[np.maximum(np.argmax(bad, axis=1) - 1, 0)], 1.0)
        scale = np.ones(n)
        np.minimum.at(scale, tets[bad.any(axis=1)].reshape(-1), np.repeat(lo[bad.any(axis=1)], 4))
        if (scale >= 1).all():
            break
        direction *= scale[:, None]
    return direction


floor_tets = np.flatnonzero(volume_path(free)[0] <= FLOOR + .001)
print(f"{run.name} frame {frame}: {len(tets)} tets, {len(floor_tets)} at the floor")
for method in ("none", "jacobi", "gauss-seidel"):
    direction = free.copy()
    direction[fixed] = 0
    if method != "none":
        g = volume_gradient(floor_tets)
        movable = ~fixed[tets[floor_tets]]
        for _ in range(64):
            j0, j1, _, _ = volume_path(direction, floor_tets)
            if method == "jacobi":
                violation = TARGET - .9 * np.maximum(0, j0 - FLOOR) - j1
                act = violation > 0
                h = g * movable[:, :, None]
                denominator = np.einsum("tki,tki->t", g, h)
                act &= denominator > 1e-20
                correction = np.zeros_like(direction)
                count = np.zeros(n)
                for k in range(4):
                    np.add.at(correction, tets[floor_tets[act], k], h[act, k] * (violation[act] / denominator[act])[:, None])
                    np.add.at(count, tets[floor_tets[act], k], movable[act, k])
                update = count > 0
                direction[update] += correction[update] / count[update, None]
            else:
                for m, t in enumerate(floor_tets):
                    ids = tets[t]
                    j0t, j1t, _, _ = volume_path(direction, [t])
                    violation = TARGET - .9 * max(0, j0t[0] - FLOOR) - j1t[0]
                    if violation <= 0:
                        continue
                    h = g[m] * movable[m][:, None]
                    denominator = np.sum(g[m] * h)
                    if denominator > 1e-20:
                        direction[ids] += h * violation / denominator
    j0, j1, _, _ = volume_path(direction, floor_tets)
    restricted = restrict(direction)
    length, kept = np.linalg.norm(direction, axis=1), np.linalg.norm(restricted, axis=1)
    zeroed = piece & ~fixed & (length > 1e-5) & (kept < .01 * length)
    crease = piece & (np.abs((rest[:, 0] - scenario.KIDNEY.tumor_center[0]) * 10) < 2) & ((rest[:, 2] - scenario.KIDNEY.tumor_center[2]) * 10 > 2)
    offset = (rest - scenario.KIDNEY.tumor_center) * 10
    still = floor_tets[j1 < 0]
    fixed_count = fixed[tets[still]].sum(axis=1)
    centres = offset[tets[still]].mean(axis=1)
    print(f"    compressing floor tets by fixed nodes 0/1/2/3/4: {np.bincount(fixed_count, minlength=5)}; rest y of those "
          f"without fixed nodes: {np.round(np.sort(centres[fixed_count == 0][:, 1]), 1)}")
    zc = offset[zeroed]
    print(f"    zeroed nodes rest y p10/p50/p90 {np.percentile(zc[:, 1], [10, 50, 90]).round(1) if len(zc) else '-'}")
    print(f"  {method:12s}: floor tets still compressing {np.count_nonzero(j1 < 0)}, zeroed nodes {zeroed.sum()} "
          f"(crease {np.count_nonzero(zeroed & crease)}), crease step median before/after restriction "
          f"{np.median(length[crease & ~fixed]) * 1e4:.0f}/{np.median(kept[crease & ~fixed]) * 1e4:.0f} um")
