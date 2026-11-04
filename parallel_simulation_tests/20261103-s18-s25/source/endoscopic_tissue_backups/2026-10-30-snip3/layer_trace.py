"""Trace through a stage the z shift (mm) of the two node layers that straddle the shear plane in tab columns of runs
probed by flap_probe.py.

Usage: python layer_trace.py <stage> <stride> <x values comma separated> <run folder>...

For every listed rest x, the mean z shift of the material nodes of rest z = 0 and z = 0.1 cm over the tab's rest y range.
"""
import json
import sys
from pathlib import Path

import numpy as np

stage, stride, xs = sys.argv[1], int(sys.argv[2]), [float(v) for v in sys.argv[3].split(",")]
for folder in map(Path, sys.argv[4:]):
    data = np.load(str(folder) + ".npz")
    rest, inside, ids, track = data["rest"], data["inside"] != 0, data["track_ids"], data["track_nodes"]
    rows = [json.loads(line) for line in open(folder / "trajectory.jsonl")]
    steps = [r["step"] for r in rows if r["stage"] == stage]
    by_step = {r["step"]: r for r in rows}
    tr = rest[ids]
    near = lambda value, target: np.abs(value - target) < 1e-4
    print(f"== {folder.name} {stage}")
    for k, step in enumerate(steps):
        if k % stride and k != len(steps) - 1:
            continue
        shift = (track[step - 1] - tr) * 10
        parts = []
        for x in xs:
            column = inside[ids] & near(tr[:, 0], x) & (tr[:, 1] > .32) & (tr[:, 1] < .68)
            lower, upper, above = column & near(tr[:, 2], 0.0), column & near(tr[:, 2], 0.1), column & near(tr[:, 2], 0.2)
            parts.append(f"x{x}: {shift[lower, 2].mean():+.2f}/{shift[upper, 2].mean():+.2f}/{shift[above, 2].mean():+.2f}")
        r = by_step[step]
        print(f"  {step:4d} gap {r['scissors_gap_cm'] * 10:5.2f} L {r['scissors_loads']['left_n']:4.2f} | " + " | ".join(parts))
