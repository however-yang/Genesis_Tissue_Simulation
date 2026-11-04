"""Trace, between two steps of a run probed by blade_probe.py, the scissors' z offset from the requested pose and the
z force each contact kind puts on the scissors (N).

Usage: python lateral_force.py <run folder> <first step> <last step> <stride>
"""
import json, sys
import numpy as np
from endoscopic_tissue.grid import METERS_PER_UNIT
run, lo, hi, stride = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
d = np.load(run + ".npz"); steps, c = d["contact_steps"], d["contacts"]
rows = {json.loads(l)["step"]: json.loads(l) for l in open(run + "/trajectory.jsonl")}
names = {1: "eA", 2: "eB", 3: "bA", 4: "bB", 5: "wr"}
for s in range(lo, hi + 1, stride):
    r = rows[s]; b = np.array(r["blade_center_cm"]); q = np.array(r["requested_command"]["blade_center"])
    h = c[steps == s]
    # z force on the scissors (the energy gradient term summed) per kind, N
    parts = []
    for k in (1, 2, 3, 4, 5):
        m = h[:, 0] == k
        if m.any():
            fz = h[m, 9].sum() * METERS_PER_UNIT
            parts.append(f"{names[k]}{int(m.sum())}:{fz:+.2f}")
    L = r["scissors_loads"]
    print(f"{s}: gap {r['scissors_gap_cm']*10:.2f} dz {(b[2]-q[2])*10:+.3f} L {L['left_n']:.2f} R {L['right_n']:.2f} | z force on scissors " + " ".join(parts)
          + f" | total {h[:, 9].sum() * METERS_PER_UNIT:+.2f}")
