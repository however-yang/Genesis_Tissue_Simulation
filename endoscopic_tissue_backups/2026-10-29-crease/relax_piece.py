"""Relax the excised piece of a kidney run to static equilibrium offline, with the solver's tissue energy and the nodes
the grasper grips held where they are, and report whether its crease survives.

Usage: python relax_piece.py <stage npz> [--fix-pads] [--poisson NU] [--young PA]

The energy is the solver's (unified_vertex.cuh): stable Neo-Hookean elasticity with lambda = kappa + mu/3, the C2
compression barrier below twice the volume floor, tension-only edge hardening, and gravity. The piece is the lattice
component of the gripped nodes over the snapshot's intact links, its tetrahedra the Kuhn tetrahedra whose four nodes
and axis links lie in it. --fix-pads also holds every piece node between the jaw pads.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from endoscopic_tissue import scenario  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("snapshot")
parser.add_argument("--fix-pads", action="store_true")
parser.add_argument("--young", type=float, default=15000)
parser.add_argument("--poisson", type=float, default=.25)
parser.add_argument("--hardening", type=float, default=80)
parser.add_argument("--floor", type=float, default=.15)
parser.add_argument("--no-relax", action="store_true")
# A per-node dump of debug_crease_dump.py: its positions replace the snapshot's, and its contact and anchor witnesses
# (and with --fix-locked also the nodes whose next local step the volume bound stops) are held where they are.
parser.add_argument("--dump")
parser.add_argument("--fix-locked", action="store_true")
# Hold every node farther than this many cm from the tumor centre at rest, and drop the tets that have no free node.
parser.add_argument("--radius", type=float)
args = parser.parse_args()

grid = scenario.KIDNEY_SCENE.build_grid(1050)
rest = grid.positions.astype(np.float64) * .01
snapshot = np.load(args.snapshot)
nodes = snapshot["nodes_cm"].astype(np.float64) * .01
links = snapshot["neighbors"]
active = grid.active != 0

# The held piece: the component of the gripped nodes over the intact links.
source = np.repeat(np.arange(len(links)), 6)
target = links.reshape(-1)
keep = (target >= 0) & active[source]
graph = coo_matrix((np.ones(keep.sum()), (source[keep], target[keep])), shape=(len(links),) * 2)
_, labels = connected_components(graph, directed=False)
gripped = snapshot["grip_flags"] != 0
piece_label = np.bincount(labels[gripped]).argmax()
piece = active & (labels == piece_label)

binary_corners = grid.corners[:, (0, 1, 3, 2, 4, 5, 7, 6)]
supported = np.all(grid.active[binary_corners] != 0, axis=1)
kuhn = np.array(((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)))
tets = binary_corners[supported][:, kuhn].reshape(-1, 4)
tets = tets[piece[tets].all(axis=1)]
# A tet is alive when every axis link between its nodes is intact.
alive = np.ones(len(tets), dtype=bool)
for a in range(4):
    for b in range(a + 1, 4):
        step = rest[tets[:, b]] - rest[tets[:, a]]
        axis_link = np.count_nonzero(np.abs(step) > 1e-5, axis=1) == 1
        linked = (links[tets[:, a]] == tets[:, b, None]).any(axis=1)
        alive &= ~axis_link | linked
tets = tets[alive]

pads = np.zeros(len(nodes), dtype=bool)
if args.fix_pads:
    settings = scenario.scissors_settings()
    axis = np.array(settings.jaw_axis) / np.linalg.norm(settings.jaw_axis)
    normal = np.array(settings.jaw_normal) - axis * (axis @ settings.jaw_normal)
    normal /= np.linalg.norm(normal)
    local = (nodes - snapshot["hinge_cm"] * .01) @ np.stack((axis, normal, np.cross(axis, normal))).T
    gap = float(snapshot["jaw_gap_cm"]) * .01
    pads = piece & (local[:, 0] > 0) & (local[:, 0] < settings.jaw_length * .01) \
        & (np.abs(local[:, 1]) < gap / 2 + 2e-4) & (np.abs(local[:, 2]) < settings.jaw_width * .005 + 5e-4)
if args.dump:
    count = np.fromfile(args.dump, dtype=np.int32, count=1)[0]
    values = np.fromfile(args.dump, dtype=np.float32, offset=8, count=18 * count)
    nodes = values[:3 * count].reshape(count, 3).astype(np.float64) * .01
    dump_steps = np.linalg.norm(values[3 * count:6 * count].reshape(count, 3), axis=1) * 1e4
    dump_bounds = values[6 * count:9 * count].reshape(count, 3)[:, 0]
    dump_flags = np.fromfile(args.dump, dtype=np.int32, offset=8 + 4 * 18 * count, count=count)
    pads |= piece & ((dump_flags & 12) != 0)
    if args.fix_locked:
        pads |= piece & (dump_bounds < .01) & (dump_steps > 1)
if args.radius:
    far = np.linalg.norm(grid.positions - scenario.KIDNEY.tumor_center, axis=1) > args.radius
    pads |= piece & far
    tets = tets[~far[tets].all(axis=1)]
fixed = piece & (gripped | pads) | (grid.pinned != 0)
in_tet = np.zeros(len(nodes), dtype=bool)
in_tet[tets] = True
free = np.flatnonzero(piece & ~fixed & in_tet)
print(f"piece {piece.sum()} nodes, {len(tets)} tetrahedra, fixed {fixed.sum()} ({gripped.sum()} gripped)")

mu = args.young / (2 * (1 + args.poisson))
lam = args.young / (3 * (1 - 2 * args.poisson)) + mu / 3
rest_edges = rest[tets[:, 1:]] - rest[tets[:, :1]]
volume = np.abs(np.linalg.det(rest_edges)) / 6
inverse = np.linalg.inv(np.transpose(rest_edges, (0, 2, 1)))
pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
rest_lengths = np.stack([np.linalg.norm(rest[tets[:, b]] - rest[tets[:, a]], axis=1) for a, b in pairs], axis=1)
mass = grid.mass.astype(np.float64)
activation = 2 * args.floor


def energy(x_free, positions):
    x = positions.copy()
    x[free] = x_free.reshape(-1, 3)
    edges = x[tets[:, 1:]] - x[tets[:, :1]]
    F = np.transpose(edges, (0, 2, 1)) @ inverse
    J = np.linalg.det(F)
    cof = np.stack([np.cross(F[:, :, 1], F[:, :, 2]), np.cross(F[:, :, 2], F[:, :, 0]),
                    np.cross(F[:, :, 0], F[:, :, 1])], axis=2)
    psi = .5 * mu * (np.einsum("tij,tij->t", F, F) - 3) - mu * (J - 1) + .5 * lam * (J - 1) ** 2
    dpsi_dJ = -mu + lam * (J - 1)
    # The C2 barrier below twice the floor, extended quadratically below 1e-4 of it as in the solver.
    low = J < activation
    if low.any():
        j = np.maximum(J[low], activation * 1e-4)
        d, l = j - activation, np.log(j / activation)
        b_energy, b_gradient = -lam * d * d * l, -lam * (2 * d * l + d * d / j)
        b_curvature = -lam * (2 * l + 4 * d / j - d * d / (j * j))
        offset = J[low] - j
        psi[low] += b_energy + b_gradient * offset + .5 * b_curvature * offset ** 2
        dpsi_dJ[low] += b_gradient + b_curvature * offset
    P = mu * F + dpsi_dJ[:, None, None] * cof
    total = (volume * psi).sum()
    H = (volume[:, None, None] * P) @ np.transpose(inverse, (0, 2, 1))
    grad = np.zeros_like(x)
    for k in range(3):
        np.add.at(grad, tets[:, k + 1], H[:, :, k])
        np.add.at(grad, tets[:, 0], -H[:, :, k])
    # Tension-only edge hardening, .5*mu*V*hardening*strain^6 per tet edge.
    for e, (a, b) in enumerate(pairs):
        d = x[tets[:, b]] - x[tets[:, a]]
        length = np.linalg.norm(d, axis=1)
        strain = np.maximum(length / rest_lengths[:, e] - 1, 0)
        total += (.5 * mu * args.hardening * volume * strain ** 6).sum()
        force = (3 * mu * args.hardening * volume * strain ** 5 / rest_lengths[:, e] / np.maximum(length, 1e-12))[:, None] * d
        np.add.at(grad, tets[:, b], force)
        np.add.at(grad, tets[:, a], -force)
    total += 9.81 * (mass[free] * x[free, 1]).sum()
    grad[free, 1] += 9.81 * mass[free]
    return total, grad[free].reshape(-1)


def report(label, x):
    edges = x[tets[:, 1:]] - x[tets[:, :1]]
    J = np.linalg.det(np.transpose(edges, (0, 2, 1)) @ inverse)
    tumor = piece & (scenario.KIDNEY.tumor_weight(grid.positions, 1e-3) > .5) & (grid.inside != 0)
    center = scenario.KIDNEY.tumor_center
    columns = np.round((grid.positions[:, 0] - center[0]) / .1).astype(int)
    height = grid.positions[:, 1] - center[1]
    profile = []
    for column in range(-5, 6):
        pick = tumor & (columns == column) & (np.abs(height) < .15)
        profile.append(np.ptp(x[pick, 2]) * 1000 if pick.sum() > 1 else np.nan)
    print(f"  {label}: energy {energy(x[free].reshape(-1), x)[0] * 1e6:.3f} uJ, J<0.2 {np.count_nonzero(J < .2)}, "
          f"min J {J.min():.3f}, extents {np.ptp(x[piece], axis=0).round(5) * 1000} mm")
    print("    middle z thickness (mm) x=-5..5: " + " ".join(f"{v:5.1f}" for v in profile))


report("snapshot", nodes)
start_gradient = energy(nodes[free].reshape(-1), nodes)[1].reshape(-1, 3)
force = np.linalg.norm(start_gradient, axis=1)
print(f"  force at the snapshot: rms {np.sqrt(np.mean(force ** 2)) * 1e6:.2f} uN, p99 {np.percentile(force, 99) * 1e6:.2f} uN, "
      f"max {force.max() * 1e6:.2f} uN (gravity per node {9.81 * mass[free].mean() * 1e6:.2f} uN)")
# Where the unbalanced nodes are: rest position from the tumor centre, and position and force in the jaw frame.
settings = scenario.scissors_settings()
jaw_axis = np.array(settings.jaw_axis) / np.linalg.norm(settings.jaw_axis)
jaw_normal = np.array(settings.jaw_normal) - jaw_axis * (jaw_axis @ settings.jaw_normal)
jaw_normal /= np.linalg.norm(jaw_normal)
jaw_frame = np.stack((jaw_axis, jaw_normal, np.cross(jaw_axis, jaw_normal)))
strong = force > 2e-3
local = (nodes[free] - snapshot["hinge_cm"] * .01) @ jaw_frame.T * 1000
rest_mm = (grid.positions[free] - scenario.KIDNEY.tumor_center) * 10
direction = -start_gradient @ jaw_frame.T / np.maximum(force, 1e-12)[:, None]
print(f"  nodes with more than 2 mN: {strong.sum()}")
if strong.any():
    print(f"    rest mm min {rest_mm[strong].min(0).round(1)} max {rest_mm[strong].max(0).round(1)} median {np.median(rest_mm[strong], 0).round(1)}")
    print(f"    jaw frame mm (along, normal, width) min {local[strong].min(0).round(1)} max {local[strong].max(0).round(1)}")
    print(f"    mean force direction in the jaw frame {direction[strong].mean(0).round(2)}")
    on_grip = np.zeros(len(nodes), dtype=bool)
    for a in range(6):
        neighbor = links[:, a]
        on_grip |= (neighbor >= 0) & gripped[np.maximum(neighbor, 0)]
    print(f"    of them next to a gripped node: {np.count_nonzero(on_grip[free][strong])}")
value, gradient = energy(nodes[free].reshape(-1), nodes)
probe = np.random.default_rng(0).standard_normal(gradient.shape) * 1e-9
numeric = (energy(nodes[free].reshape(-1) + probe, nodes)[0] - energy(nodes[free].reshape(-1) - probe, nodes)[0]) / 2
print(f"  gradient check: directional {gradient @ probe:.6e} vs finite difference {numeric:.6e}")
if "--no-relax" in sys.argv:
    sys.exit()
result = minimize(energy, nodes[free].reshape(-1), args=(nodes,), jac=True, method="L-BFGS-B",
                  options={"maxiter": 20000, "maxcor": 30, "gtol": 1e-12, "ftol": 1e-15})
relaxed = nodes.copy()
relaxed[free] = result.x.reshape(-1, 3)
move = np.linalg.norm(relaxed - nodes, axis=1)
print(f"  L-BFGS: {result.nit} iterations, {result.message}, |grad| {np.abs(result.jac).max():.3e}, "
      f"largest move {move.max() * 1000:.3f} mm, rms move {np.sqrt(np.mean(move[free] ** 2)) * 1000:.3f} mm")
report("relaxed", relaxed)
# The energy along the straight path from the snapshot to the relaxed state: a barrier marks a separate local minimum.
path = [energy(((1 - a) * nodes[free] + a * relaxed[free]).reshape(-1), nodes)[0] * 1e6 for a in np.linspace(0, 1, 11)]
print("  energy along the straight path (uJ): " + " ".join(f"{v:.1f}" for v in path))
OUTPUT = Path(__file__).resolve().parents[2] / "endoscopic_tissue" / "outputs" / "crease-runs"
OUTPUT.mkdir(exist_ok=True)
snapshot_path = Path(args.snapshot)
np.save(OUTPUT / f"{snapshot_path.parent.name}-{snapshot_path.stem}-relaxed.npy", relaxed.astype(np.float32))
