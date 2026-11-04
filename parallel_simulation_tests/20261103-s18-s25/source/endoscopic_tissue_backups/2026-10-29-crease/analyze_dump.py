"""Read the per-node dumps of debug_crease_dump.py for one kidney run and show what holds the held piece where it is:
per region, the elastic force of the solver's tissue energy at the dumped positions, the contact and anchor force the
solver assembled, their sum with gravity, the local vertex step the solver would take next and the share of that step
its volume bound keeps.

Usage: python analyze_dump.py <run folder> [<frame> ...]   (dumps in <run folder>-dump, piece from <run folder>/hold.npz)

Forces are in mN. The native force unit of the solver is 0.01 N (kg cm / s^2), and a force is minus a gradient.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from endoscopic_tissue import scenario  # noqa: E402

FLOOR, YOUNG, POISSON, HARDENING = .15, 15000., .25, 80.
run = Path(sys.argv[1])
frames = [int(f) for f in sys.argv[2:]] or sorted(int(p.stem.split("-")[1]) for p in Path(f"{run}-dump").glob("crease-*.bin"))
grid = scenario.KIDNEY_SCENE.build_grid(1050)
rest = grid.positions.astype(np.float64) * .01
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
mu = YOUNG / (2 * (1 + POISSON))
lam = YOUNG / (3 * (1 - 2 * POISSON)) + mu / 3
rest_edges = rest[tets[:, 1:]] - rest[tets[:, :1]]
volume = np.abs(np.linalg.det(rest_edges)) / 6
inverse = np.linalg.inv(np.transpose(rest_edges, (0, 2, 1)))
pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
rest_lengths = np.stack([np.linalg.norm(rest[tets[:, b]] - rest[tets[:, a]], axis=1) for a, b in pairs], axis=1)


def elastic_gradient(x):
    """Return the gradient (N) of the solver's tissue energy at positions x (m), and the volume ratios."""
    edges = x[tets[:, 1:]] - x[tets[:, :1]]
    F = np.transpose(edges, (0, 2, 1)) @ inverse
    J = np.linalg.det(F)
    cof = np.stack([np.cross(F[:, :, 1], F[:, :, 2]), np.cross(F[:, :, 2], F[:, :, 0]),
                    np.cross(F[:, :, 0], F[:, :, 1])], axis=2)
    dpsi = -mu + lam * (J - 1)
    low = J < 2 * FLOOR
    j = np.maximum(J[low], 2 * FLOOR * 1e-4)
    d, l = j - 2 * FLOOR, np.log(j / (2 * FLOOR))
    curvature = -lam * (2 * l + 4 * d / j - d * d / (j * j))
    dpsi[low] += -lam * (2 * d * l + d * d / j) + curvature * (J[low] - j)
    H = (volume[:, None, None] * (mu * F + dpsi[:, None, None] * cof)) @ np.transpose(inverse, (0, 2, 1))
    gradient = np.zeros_like(x)
    for k in range(3):
        np.add.at(gradient, tets[:, k + 1], H[:, :, k])
        np.add.at(gradient, tets[:, 0], -H[:, :, k])
    for e, (a, b) in enumerate(pairs):
        d = x[tets[:, b]] - x[tets[:, a]]
        length = np.linalg.norm(d, axis=1)
        strain = np.maximum(length / rest_lengths[:, e] - 1, 0)
        force = (3 * mu * HARDENING * volume * strain ** 5 / rest_lengths[:, e] / np.maximum(length, 1e-12))[:, None] * d
        np.add.at(gradient, tets[:, b], force)
        np.add.at(gradient, tets[:, a], -force)
    return gradient, J


settings = scenario.scissors_settings()
jaw_axis = np.array(settings.jaw_axis) / np.linalg.norm(settings.jaw_axis)
jaw_normal = np.array(settings.jaw_normal) - jaw_axis * (jaw_axis @ settings.jaw_normal)
jaw_normal /= np.linalg.norm(jaw_normal)
jaw_frame = np.stack((jaw_axis, jaw_normal, np.cross(jaw_axis, jaw_normal)))
offset = (grid.positions - scenario.KIDNEY.tumor_center) * 10
regions = {"crease (+z half, |x|<2 mm)": piece & (np.abs(offset[:, 0]) < 2) & (offset[:, 2] > 2) & (offset[:, 1] > -3),
           "+z half, |x|>3 mm": piece & (np.abs(offset[:, 0]) > 3) & (offset[:, 2] > 2) & (offset[:, 1] > -3),
           "-z half": piece & (offset[:, 2] < -2) & (offset[:, 1] > -3),
           "whole piece": piece}
for frame in frames:
    raw = np.fromfile(f"{run}-dump/crease-{frame:04d}.bin", dtype=np.int32, count=2)
    n = raw[0]
    data = np.fromfile(f"{run}-dump/crease-{frame:04d}.bin", dtype=np.float32, offset=8, count=n * 18)
    positions, steps, bounds, interaction = (data[:3 * n].reshape(n, 3), data[3 * n:6 * n].reshape(n, 3),
                                             data[6 * n:9 * n].reshape(n, 3)[:, 0], data[9 * n:18 * n].reshape(n, 9))
    flags = np.fromfile(f"{run}-dump/crease-{frame:04d}.bin", dtype=np.int32, offset=8 + 4 * 18 * n, count=n)
    # The material solve block of debug_material_dump.py, when the build has it.
    material = Path(f"{run}-dump/crease-{frame:04d}.bin").stat().st_size >= 8 + 4 * n * (19 + 9)
    if material:
        base = 8 + 4 * 19 * n
        directions = np.fromfile(f"{run}-dump/crease-{frame:04d}.bin", dtype=np.float32, offset=base, count=6 * n)
        free_direction, used_direction = directions[:3 * n].reshape(n, 3), directions[3 * n:].reshape(n, 3)
        line = np.fromfile(f"{run}-dump/crease-{frame:04d}.bin", dtype=np.int32, offset=base + 24 * n, count=n)
        line_gain = np.fromfile(f"{run}-dump/crease-{frame:04d}.bin", dtype=np.float32, offset=base + 28 * n, count=n)
        component = np.fromfile(f"{run}-dump/crease-{frame:04d}.bin", dtype=np.int32, offset=base + 32 * n, count=n)
    gradient, J = elastic_gradient(positions.astype(np.float64) * .01)
    elastic = -gradient * 1000
    # Only witnesses get their interaction assembled: the buffer keeps stale values at every other node.
    contact = -interaction[:, :3].astype(np.float64) * 10 * ((flags & 12) != 0)[:, None]
    gravity = np.zeros_like(elastic)
    gravity[:, 1] = -9.81 * grid.mass * 1000
    total = elastic + contact + gravity
    step = np.linalg.norm(steps, axis=1) * 1e4
    print(f"== {run.name} frame {frame}: piece {piece.sum()} nodes, tets at the floor {np.count_nonzero(J < FLOOR + 5e-4)}")
    for name, region in regions.items():
        movable = region & ((flags & 16) != 0)
        if not movable.any():
            continue
        e, c, t = (np.linalg.norm(v[movable], axis=1) for v in (elastic, contact, total))
        bound_zero = movable & (bounds < .01) & (step > 1)
        print(f"  {name}: {movable.sum()} movable, contact witnesses {np.count_nonzero(movable & (flags & 4 != 0))}, "
              f"anchor witnesses {np.count_nonzero(movable & (flags & 8 != 0))}, fixed in the material solve "
              f"{np.count_nonzero(movable & (flags & 2 != 0))}")
        print(f"    elastic p50/p95/max {np.median(e):.3f}/{np.percentile(e, 95):.3f}/{e.max():.2f} mN, contact+anchor "
              f"p95/max {np.percentile(c, 95):.3f}/{c.max():.2f} mN, sum with gravity p50/p95/max {np.median(t):.3f}/"
              f"{np.percentile(t, 95):.3f}/{t.max():.2f} mN")
        print(f"    next local step p50/p95/max {np.median(step[movable]):.2f}/{np.percentile(step[movable], 95):.2f}/"
              f"{step[movable].max():.1f} um, steps above 1 um the volume bound stops: {bound_zero.sum()}")
        if material:
            solved = movable & ((flags & 2) == 0)
            free_length = np.linalg.norm(free_direction[solved], axis=1) * 1e4
            used_length = np.linalg.norm(used_direction[solved], axis=1) * 1e4
            blocked = (free_length > .1) & (used_length < .01 * free_length)
            print(f"    material solve over {solved.sum()} nodes: PCG step p50/p95 {np.median(free_length):.2f}/"
                  f"{np.percentile(free_length, 95):.2f} um, after the volume restriction p50/p95 {np.median(used_length):.2f}/"
                  f"{np.percentile(used_length, 95):.2f} um, zeroed {blocked.sum()}")
    if material:
        roots, counts = np.unique(component[piece], return_counts=True)
        root = roots[np.argmax(counts)]
        print(f"  material line search of the piece's component (root {root}): chosen trial {line[root] & 15} "
              f"(0 = none), invalid trials {line[root] >> 4:05b}, energy change {line_gain[root]:.3e}")
    strong = piece & ((flags & 16) != 0) & (np.linalg.norm(total, axis=1) > 1)
    print(f"  nodes with more than 1 mN unbalanced: {strong.sum()}")
    order = np.argsort(-np.linalg.norm(total, axis=1) * strong)[:min(12, strong.sum())]
    local = (positions - hold["hinge_cm"]) @ jaw_frame.T * 10
    for i in order:
        print(f"    rest {offset[i].round(1)} jaw frame {local[i].round(1)} flags {flags[i]:2d} elastic "
              f"{np.linalg.norm(elastic[i]):.2f} contact {np.linalg.norm(contact[i]):.2f} sum {np.linalg.norm(total[i]):.2f} mN, "
              f"step {step[i]:.1f} um bound {bounds[i]:.2f}")
