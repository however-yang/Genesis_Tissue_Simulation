"""Print, for every run probed by blade_probe.py in the current folder, the distance in mm from the shear face to the
mid-plane of the two node layers that straddle it, at the root (rest x 3.1-3.3) and the free end (3.4-3.5), 40 steps
into each snip.

Usage: python misalignment.py
"""
import json, glob, os
import numpy as np
print("run, outcome, slit mid-plane minus shear face (mm) at the start of each snip, for root x 3.1-3.3 / free end x 3.4-3.5")
for f in sorted(glob.glob("*.npz")):
    run = f[:-4]
    rows = {json.loads(l)["step"]: json.loads(l) for l in open(os.path.join(run, "trajectory.jsonl"))}
    rep = json.load(open(os.path.join(run, "report.json")))
    d = np.load(f); r = d["rest"]; inside = d["inside"] != 0; ids = d["track_ids"]; tr = r[ids]
    starts = {}
    for s, row in sorted(rows.items()):
        if row["stage"].startswith("incise_") and row["stage"] not in starts: starts[row["stage"]] = s
    out = []
    for stage in ("incise_1", "incise_2", "incise_3"):
        s = starts[stage] + 40   # blades about to load the tissue
        n = d["track_nodes"][s - 1]; face = rows[s]["scissors_edges_cm"][0][0][2]
        vals = []
        for lo_x, hi_x in ((3.05, 3.35), (3.35, 3.55)):
            m = inside[ids] & (tr[:, 0] > lo_x) & (tr[:, 0] < hi_x) & (tr[:, 1] > .32) & (tr[:, 1] < .68)
            lo = m & (np.abs(tr[:, 2]) < 1e-4); up = m & (np.abs(tr[:, 2] - .1) < 1e-4)
            vals.append(((n[lo, 2].mean() + n[up, 2].mean()) / 2 - face) * 10)
        out.append(f"{stage[-1]}: {vals[0]:+.2f}/{vals[1]:+.2f}")
    print(f"{run:22s} {rep['status'][:4]} uncut {rep['metrics']['uncut_columns']} | " + "  ".join(out))
