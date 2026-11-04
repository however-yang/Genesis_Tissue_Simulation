"""Sum, through one stage of runs probed by blade_probe.py, the z force of blade back contacts that push tissue of
their own side of the shear plane across it (back A with a +z normal below the face, back B with a -z normal above).

Usage: python wrong_side.py <stage> <run folder>...
"""
import json, sys
import numpy as np
from endoscopic_tissue.grid import METERS_PER_UNIT
for run in sys.argv[2:]:
    stage = sys.argv[1]
    d = np.load(run + ".npz"); steps, c = d["contact_steps"], d["contacts"]
    rows = {json.loads(l)["step"]: json.loads(l) for l in open(run + "/trajectory.jsonl")}
    out = []; total = 0.0; first = None
    for s in sorted(k for k, r in rows.items() if r["stage"] == stage):
        h = c[steps == s]
        if not len(h): continue
        face = rows[s]["scissors_edges_cm"][0][0][2]
        fz = -h[:, 9] * METERS_PER_UNIT
        wrongA = (h[:, 0] == 3) & (h[:, 6] > .7) & (h[:, 3] < face - .005)
        wrongB = (h[:, 0] == 4) & (h[:, 6] < -.7) & (h[:, 3] > face + .005)
        w = np.abs(fz[wrongA]).sum() + np.abs(fz[wrongB]).sum()
        total += w
        if w > .02 and first is None: first = s
        if w > .02: out.append(f"{s}:{w:.2f}({int(wrongA.sum())}A/{int(wrongB.sum())}B)")
    rep = json.load(open(run + "/report.json"))
    print(f"{run:20s} {rep['status'][:4]} {stage} wrong-side push total {total:.2f} N*steps, first {first} | " + " ".join(out[:12]))
