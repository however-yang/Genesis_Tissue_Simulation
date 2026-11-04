"""Print how the flap's free end moves across the shear plane through a stage of runs probed by flap_probe.py.

Usage: python free_end_trace.py <stage> <stride> <run folder>...

Each line gives, every `stride` steps of the stage, the z shift in mm of the node layers z = 0 and z = 0.1 cm (the two
that straddle the shear plane z = 0.05) averaged over the tab's material nodes of rest x 3.5 and 3.4, the y and x shift
of those nodes, the scissors gap and loads, and the links severed so far in the stage.
"""
import json
import sys
from pathlib import Path

import numpy as np

stage, stride = sys.argv[1], int(sys.argv[2])
for folder in map(Path, sys.argv[3:]):
    data = np.load(str(folder) + ".npz")
    rest, inside, ids, track = data["rest"], data["inside"] != 0, data["track_ids"], data["track_nodes"]
    rows = [json.loads(line) for line in open(folder / "trajectory.jsonl")]
    steps = [r["step"] for r in rows if r["stage"] == stage]
    by_step = {r["step"]: r for r in rows}
    tracked_rest = rest[ids]
    near = lambda value, target: np.abs(value - target) < 1e-4
    tab = inside[ids] & (tracked_rest[:, 1] > .3) & (tracked_rest[:, 0] > 3.05)
    masks = {}
    for x in (3.5, 3.4):
        for z in (0.0, 0.1):
            masks[(x, z)] = tab & near(tracked_rest[:, 0], x) & near(tracked_rest[:, 2], z)
    cut_steps = data["cut_steps"]
    print(f"== {folder.name} {stage}")
    for k, step in enumerate(steps):
        if k % stride and k != len(steps) - 1:
            continue
        shift = (track[step - 1] - tracked_rest) * 10
        parts = []
        for x in (3.5, 3.4):
            lower, upper = masks[(x, 0.0)], masks[(x, 0.1)]
            both = lower | upper
            parts.append(f"x{x} dz {shift[lower, 2].mean():+.2f}/{shift[upper, 2].mean():+.2f} max {shift[both, 2].max():+.2f}"
                         f" dy {shift[both, 1].mean():+.2f}")
        r = by_step[step]
        loads = r["scissors_loads"]
        severed = int(np.count_nonzero((cut_steps >= steps[0]) & (cut_steps <= step)) // 2)
        print(f"  {step:4d} gap {r['scissors_gap_cm'] * 10:5.2f} L {loads['left_n']:4.2f} R {loads['right_n']:4.2f} "
              f"cut {severed:3d} | " + " | ".join(parts) + f" | hinge y {data['hinges'][step - 1][1] * 10:.2f}")
