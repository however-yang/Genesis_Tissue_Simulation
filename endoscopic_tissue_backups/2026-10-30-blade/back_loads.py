"""Print, for runs probed by blade_probe.py, the largest summed load of the two blade backs through the third snip and
every fourth step from 990 the opening-direction force of back A and back B, their axis and z forces (N).

Usage: python back_loads.py <run folder>...   (third snip at the flap's 6 degree pitch)
"""
import json, sys
import numpy as np
from endoscopic_tissue.grid import METERS_PER_UNIT
from endoscopic_tissue.native_backend import scissors_frame
axis, normal = scissors_frame(6.0)
frame = np.stack((axis, normal, (0, 0, 1)))
for run in sys.argv[1:]:
    d = np.load(run + ".npz"); steps, c = d["contact_steps"], d["contacts"]
    rows = {json.loads(l)["step"]: json.loads(l) for l in open(run + "/trajectory.jsonl")}
    rest, ids = d["rest"], d["track_ids"]
    out = []
    peak = (0, 0)
    for s in range(990, 1030):
        h = c[steps == s]
        if not len(h): continue
        f = -h[:, 7:10] @ frame.T * METERS_PER_UNIT
        a = f[h[:, 0] == 3].sum(0); b = f[h[:, 0] == 4].sum(0)
        load = abs(a[1]) + abs(b[1])
        if load > peak[0]: peak = (load, s)
        if s % 4 == 2: out.append(f"{s}:{rows[s]['scissors_gap_cm']*10:.1f}mm A{a[1]:+.2f}/B{b[1]:+.2f} ax{(a[0]+b[0]):+.2f} z{(a[2]+b[2]):+.2f}")
    rep = json.load(open(run + "/report.json"))
    print(f"{run:20s} {rep['status'][:4]} peak backs {peak[0]:.2f} N at {peak[1]} | " + " ".join(out))
