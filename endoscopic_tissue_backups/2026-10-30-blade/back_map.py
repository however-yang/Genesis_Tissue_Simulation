"""Show which tissue the scissors contacts press at one step of a run probed by blade_probe.py.

Usage: python back_map.py <run folder> <step> <pitch degrees>

Every contact is assigned to the tracked lattice node nearest its material witness. The table sums, per rest x of
those nodes and per contact kind, the contact count and the force on the tissue in newtons along the scissors axis
toward the tips, along the opening normal and along native z, and gives the witnesses' mean distance from the hinge
along the axis (cm).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from endoscopic_tissue.grid import METERS_PER_UNIT
from endoscopic_tissue.native_backend import scissors_frame

folder, step, pitch = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
data = np.load(str(folder) + ".npz")
rest, ids = data["rest"], data["track_ids"]
nodes = data["track_nodes"][step - 1]
here = data["contacts"][data["contact_steps"] == step]
row = next(json.loads(l) for l in open(folder / "trajectory.jsonl") if json.loads(l)["step"] == step)
hinge = np.array(row["blade_center_cm"])
axis, normal = scissors_frame(pitch)
frame = np.stack((axis, normal, (0, 0, 1)))
_, nearest = cKDTree(nodes).query(here[:, 1:4])
rest_x = np.round(rest[ids[nearest], 0], 2)
force = -here[:, 7:10] @ frame.T * METERS_PER_UNIT
reach = (here[:, 1:4] - hinge) @ axis
print(f"{folder.name} step {step}: gap {row['scissors_gap_cm'] * 10:.2f} mm, {len(here)} contacts")
names = {1: "edgeA", 2: "edgeB", 3: "backA", 4: "backB", 5: "wrist"}
for x in np.unique(rest_x):
    for kind in (1, 2, 3, 4, 5):
        mask = (rest_x == x) & (here[:, 0] == kind)
        if not mask.any():
            continue
        f = force[mask].sum(axis=0)
        print(f"  rest x {x:.2f} {names[kind]:5s} {int(mask.sum()):3d}: axis {f[0]:+.3f} opening {f[1]:+.3f} z {f[2]:+.3f} N"
              f"  reach {reach[mask].mean():.3f} cm")
