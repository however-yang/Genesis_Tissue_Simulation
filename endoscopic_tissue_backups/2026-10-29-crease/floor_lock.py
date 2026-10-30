"""Check, on a stage snapshot of a kidney run, which nodes of the held piece the volume floor locks: the local Newton
step of k_UnifiedVertex (elasticity, barrier, hardening and strain damping at rest velocity), and whether its bound
on incident tets at the floor scales that step to zero.

Usage: python floor_lock.py <stage npz>

A vertex step is cut to 0.9*(J - floor)/(-dJ) by every incident tet whose volume it lowers, so a vertex of a tet at
the floor stops whenever its Newton step lowers that tet's volume. The report lists the locked nodes by rest position
(mm, x and z from the tumor axis, y from the tumor centre) and how far their unconstrained step would move them.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from endoscopic_tissue import scenario  # noqa: E402

FLOOR, YOUNG, POISSON, HARDENING, TAU, H = .15, 15000., .25, 80., .04, 1 / 90
grid = scenario.KIDNEY_SCENE.build_grid(1050)
rest = grid.positions.astype(np.float64) * .01
snapshot = np.load(sys.argv[1])
x = snapshot["nodes_cm"].astype(np.float64) * .01
links = snapshot["neighbors"]
active = grid.active != 0
source = np.repeat(np.arange(len(links)), 6)
target = links.reshape(-1)
keep = (target >= 0) & active[source]
_, labels = connected_components(coo_matrix((np.ones(keep.sum()), (source[keep], target[keep])), shape=(len(links),) * 2),
                                 directed=False)
gripped = snapshot["grip_flags"] != 0
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
B = np.linalg.inv(rest_edges)
# q of each corner: the row of the inverse rest shape that maps the corner's position into F (corner 0 takes minus the sum).
q = np.concatenate([-B.sum(axis=1, keepdims=True), B], axis=1)
edges = x[tets[:, 1:]] - x[tets[:, :1]]
F = np.transpose(edges, (0, 2, 1)) @ np.transpose(B, (0, 2, 1))
J = np.linalg.det(F)
cof = np.stack([np.cross(F[:, :, 1], F[:, :, 2]), np.cross(F[:, :, 2], F[:, :, 0]), np.cross(F[:, :, 0], F[:, :, 1])], axis=2)
gJ = np.einsum("tij,tkj->tki", cof, q)
activation = 2 * FLOOR
j = np.maximum(J, activation * 1e-4)
d, l = j - activation, np.log(j / activation)
low = J < activation
barrier_gradient = np.where(low, -lam * (2 * d * l + d * d / j), 0)
barrier_curvature = np.where(low, -lam * (2 * l + 4 * d / j - d * d / (j * j)), 0)
pressure = lam * (J - 1) - mu + barrier_gradient
Fq = np.einsum("tij,tkj->tki", F, q)
gradient = np.zeros_like(x)
hessian = np.zeros((len(x), 3, 3))
for k in range(4):
    node = tets[:, k]
    g = volume[:, None] * (mu * Fq[:, k] + pressure[:, None] * gJ[:, k])
    np.add.at(gradient, node, g)
    block = volume[:, None, None] * (mu * np.einsum("tj,tj->t", q[:, k], q[:, k])[:, None, None] * np.eye(3)
                                    + (lam + barrier_curvature)[:, None, None] * np.einsum("ti,tj->tij", gJ[:, k], gJ[:, k]))
    # Strain damping at zero rate adds curvature only (unifiedStrainDamping with G = I): mu*V*tau/(2h) times
    # (q' q'^T + |q'|^2 I), where q' = inverse(F_previous)^T q and F_previous = F.
    a = .5 * mu * volume * TAU / H
    qp = np.linalg.solve(np.transpose(F, (0, 2, 1)), q[:, k][:, :, None])[:, :, 0]
    block += a[:, None, None] * (np.einsum("ti,tj->tij", qp, qp)
                                 + np.einsum("tj,tj->t", qp, qp)[:, None, None] * np.eye(3))
    np.add.at(hessian, node, block)
gradient[:, 1] += 9.81 * grid.mass
mass = grid.mass.astype(np.float64)
hessian += (mass / H ** 2)[:, None, None] * np.eye(3)
free = np.flatnonzero(piece & ~gripped & (np.abs(hessian).sum(axis=(1, 2)) > 0))
step = np.zeros_like(x)
step[free] = -np.linalg.solve(hessian[free], gradient[free][:, :, None])[:, :, 0]
# The bound of k_UnifiedVertex: scale = min over incident live tets lowering J of 0.9*(J - floor)/(-dJ), at least 0.
scale = np.ones(len(x))
for k in range(4):
    node = tets[:, k]
    dj = np.einsum("ti,ti->t", gJ[:, k], step[node])
    bound = np.where(dj < 0, np.maximum(0, .9 * (J - FLOOR) / np.where(dj < 0, -dj, 1)), np.inf)
    np.minimum.at(scale, node, bound)
locked = np.zeros(len(x), dtype=bool)
locked[free] = scale[free] < 1e-3
length = np.linalg.norm(step, axis=1) * 1e6
center = scenario.KIDNEY.tumor_center
position = (grid.positions - center) * 10
print(f"{sys.argv[1]}: piece {piece.sum()} nodes, {len(tets)} tets, at the floor {np.count_nonzero(J < FLOOR + 5e-4)}, "
      f"free nodes {len(free)}, locked {locked.sum()}")
moving = locked & (length > 1)
print(f"  locked nodes whose unconstrained step exceeds 1 um: {moving.sum()}; their steps p50/max "
      f"{np.median(length[moving]) if moving.any() else 0:.1f}/{length[moving].max() if moving.any() else 0:.1f} um")
spots = np.round(position[moving]).astype(int)
if len(spots):
    values, counts = np.unique(spots[:, [0, 2]], axis=0, return_counts=True)
    print("  by rest (x, z) mm: " + ", ".join(f"{tuple(int(v) for v in values[k])}x{counts[k]}" for k in np.argsort(-counts)[:12]))
    heights = np.round(position[moving, 1]).astype(int)
    values, counts = np.unique(heights, return_counts=True)
    print("  by rest y mm: " + ", ".join(f"{v}:{c}" for v, c in zip(values, counts)))
free_moving = ~locked & (length > 1)
free_moving[~np.isin(np.arange(len(x)), free)] = False
print(f"  unlocked nodes with a step above 1 um: {free_moving.sum()}, p50/max {np.median(length[free_moving]) if free_moving.any() else 0:.1f}/"
      f"{length[free_moving].max() if free_moving.any() else 0:.1f} um")
