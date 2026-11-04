"""Sum the scissors contact forces on the tissue per contact kind through a stage of runs probed by blade_probe.py.

Usage: python contact_summary.py <stage> <pitch degrees> <stride> <run folder>...

For every listed step the line gives, per kind (edge A, edge B, back A, back B, wrist), the contact count and the
force on the tissue in newtons along the scissors axis toward the tips, along the opening normal and along native z,
and the same for all kinds together. The scissors frame follows scissors_frame at the given pitch, without roll or yaw.
"""
import json
import sys
from pathlib import Path

import numpy as np

from endoscopic_tissue.grid import METERS_PER_UNIT
from endoscopic_tissue.native_backend import scissors_frame

stage, pitch, stride = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
axis, normal = scissors_frame(pitch)
frame = np.stack((axis, normal, (0, 0, 1)))
names = {1: "edgeA", 2: "edgeB", 3: "backA", 4: "backB", 5: "wrist"}
for folder in map(Path, sys.argv[4:]):
    data = np.load(str(folder) + ".npz")
    steps, contacts = data["contact_steps"], data["contacts"]
    rows = {json.loads(line)["step"]: json.loads(line) for line in open(folder / "trajectory.jsonl")}
    report = json.load(open(folder / "report.json"))
    stage_steps = [s for s, r in rows.items() if r["stage"] == stage]
    print(f"== {folder.name} ({report['status']}) {stage}")
    for k, step in enumerate(sorted(stage_steps)):
        if k % stride and k != len(stage_steps) - 1:
            continue
        here = contacts[steps == step]
        if not len(here):
            continue
        parts = []
        for kind in (1, 2, 3, 4, 5, None):
            mask = np.ones(len(here), dtype=bool) if kind is None else here[:, 0] == kind
            if not mask.any():
                continue
            force = -(here[mask, 7:10].sum(axis=0)) @ frame.T * METERS_PER_UNIT
            parts.append(f"{names.get(kind, 'all')} {int(mask.sum())}: {force[0]:+.2f} {force[1]:+.2f} {force[2]:+.2f}")
        r = rows[step]
        print(f"  {step:4d} gap {r['scissors_gap_cm'] * 10:5.2f} | " + " | ".join(parts))
