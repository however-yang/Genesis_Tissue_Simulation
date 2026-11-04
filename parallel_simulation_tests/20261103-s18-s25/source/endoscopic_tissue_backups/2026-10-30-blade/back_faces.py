"""Split the blade back contacts at one step of a run probed by blade_probe.py by the box face they act through (+z,
-z or in-plane), with the force on the tissue and the rest x and z of the pressed nodes.

Usage: python back_faces.py <run folder> <step> [pitch degrees, default 6]
"""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
from endoscopic_tissue.grid import METERS_PER_UNIT
from endoscopic_tissue.native_backend import scissors_frame
run, step = sys.argv[1], int(sys.argv[2])
axis, normal = scissors_frame(float(sys.argv[3]) if len(sys.argv) > 3 else 6.0)
frame = np.stack((axis, normal, (0, 0, 1)))
d = np.load(run + ".npz"); h = d["contacts"][d["contact_steps"] == step]
rest, ids = d["rest"], d["track_ids"]; nodes = d["track_nodes"][step - 1]
_, near = cKDTree(nodes).query(h[:, 1:4])
f = -h[:, 7:10] @ frame.T * METERS_PER_UNIT
nz = h[:, 6]
face = np.where(np.abs(nz) > .7, np.where(nz > 0, "+z", "-z"), "inplane")
print(run, step, len(h))
for kind in (3, 4):
    for fc in ("+z", "-z", "inplane"):
        m = (h[:, 0] == kind) & (face == fc)
        if not m.any(): continue
        fs = f[m].sum(0)
        rx = rest[ids[near[m]], 0]; rz = rest[ids[near[m]], 2]
        loaded = np.abs(f[m]).sum(1) > 1e-3
        print(f"  back{'A' if kind == 3 else 'B'} normal {fc:7s} n {int(m.sum()):3d} loaded {int(loaded.sum()):3d}: axis {fs[0]:+.3f} opening {fs[1]:+.3f} z {fs[2]:+.3f} N"
              f" | witness z {h[m,3][loaded].min() if loaded.any() else 0:.3f}..{h[m,3][loaded].max() if loaded.any() else 0:.3f} rest x {np.unique(np.round(rx[loaded],1)).tolist()} rest z {np.unique(np.round(rz[loaded],1)).tolist()}")
