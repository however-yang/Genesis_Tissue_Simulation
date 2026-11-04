"""List the tracked lattice nodes that lie between the two scissors edges at one step of a run probed by flap_probe.py.

Usage: python wedge_nodes.py <run folder> <step>

A node counts when its deformed position lies within 1 mm of the shear plane z = 0.05 cm and, projected on the plane,
between the two edge lines and within their length. Each line gives the node's rest position, whether it is a
material node, its distance from the plane and from the hinge (mm), and the cut state of its column: the rest z of the
column's severed z-link, or 'uncut'.
"""
import json
import sys
from pathlib import Path

import numpy as np

folder, step = Path(sys.argv[1]), int(sys.argv[2])
data = np.load(str(folder) + ".npz")
rest, inside, initial, ids = data["rest"], data["inside"] != 0, data["initial"], data["track_ids"]
nodes = data["track_nodes"][step - 1]
row = next(json.loads(l) for l in open(folder / "trajectory.jsonl") if json.loads(l)["step"] == step)
(a0, a1), (b0, b1) = (np.array(e)[:, :2] for e in row["scissors_edges_cm"])
cut = (data["cut_steps"] <= step) & (data["cut_slots"] == 5)
levels = {}
for node in data["cut_nodes"][cut]:
    levels.setdefault(tuple(np.round(rest[node, :2], 3)), []).append(round(float(rest[node, 2]), 2))


def side(p, q0, q1):
    d = q1 - q0
    return d[0] * (p[:, 1] - q0[1]) - d[1] * (p[:, 0] - q0[0])


def within(p, q0, q1):
    d = q1 - q0
    t = ((p - q0) @ d) / (d @ d)
    return (t >= 0) & (t <= 1)


plane = np.abs(nodes[:, 2] - .05) < .1
p = nodes[:, :2]
# The open wedge: beyond the crossing point of the two edge lines toward the tips, between the rays to both tips.
da, db = a1 - a0, b1 - b0
t = np.linalg.solve(np.stack((da, -db), axis=1), b0 - a0)
crossing = a0 + t[0] * da
cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
ra, rb = a1 - crossing, b1 - crossing
inside_a = np.sign(cross(ra, p - crossing)) == np.sign(cross(ra, rb))
inside_b = np.sign(cross(rb, p - crossing)) == np.sign(cross(rb, ra))
reach = max(np.linalg.norm(ra), np.linalg.norm(rb))
bisector = (ra / np.linalg.norm(ra) + rb / np.linalg.norm(rb))
bisector /= np.linalg.norm(bisector)
along = (p - crossing) @ bisector
between = inside_a & inside_b & (along >= 0) & (along <= reach + .05)
print(f"crossing {np.round(crossing, 3)}, tips {np.round(a1, 3)} {np.round(b1, 3)}")
hinge = np.array(row["scissors_edges_cm"])[0][0][:2]
selected = np.flatnonzero(plane & between)
print(f"{folder.name} step {step}: gap {row['scissors_gap_cm'] * 10:.2f} mm, edges A {np.round(a0, 3)}->{np.round(a1, 3)}, "
      f"B {np.round(b0, 3)}->{np.round(b1, 3)}, {len(selected)} nodes between")
for k in selected[np.argsort(rest[ids[selected], 0])]:
    node = ids[k]
    key = tuple(np.round(rest[node, :2], 3))
    print(f"  rest {np.round(rest[node], 3)} {'mat ' if inside[node] else 'skin'} off-plane {(nodes[k, 2] - .05) * 10:+.2f} "
          f"pos {np.round(nodes[k], 3)} column {levels.get(key, 'uncut')}")
