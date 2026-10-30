"""Tabulate, per build, the x shift (mm) of the flap's root columns at the end of reposition_3 and the run outcome.

Usage: python pre_offset.py   (in the folder of runs probed by flap_probe.py)

The root columns are the tab's material nodes of rest x 3.1 cm and y 0.40 to 0.50 cm in the two node layers that
straddle the shear plane. Offset runs (om, op) and runs without repair (n0) are listed apart.
"""
import collections
import glob
import json

import numpy as np

by = collections.defaultdict(list)
for path in sorted(glob.glob("*.npz")):
    run = path[:-4]
    try:
        data = np.load(path)
        names = list(data["names"])
        report = json.load(open(run + "/report.json"))
    except (OSError, KeyError):
        continue
    if "reposition_3" not in names:
        continue
    rest, inside = data["rest"], data["inside"] != 0
    root = inside & (np.abs(rest[:, 0] - 3.1) < 1e-3) & (rest[:, 1] > .38) & (rest[:, 1] < .52) & (np.abs(rest[:, 2] - .05) < .06)
    nodes = data["stage_nodes"][names.index("reposition_3")]
    tag, build = run.split("-")[0], run.split("-", 2)[2]
    if tag in ("om", "op", "n0"):
        build += f" ({tag})"
    by[build].append(((nodes[root, 0] - rest[root, 0]).mean() * 10, report["status"] == "passed"))
for build, values in sorted(by.items()):
    shifts = np.array([v for v, _ in values])
    marks = " ".join(f"{v:+.2f}" + ("" if ok else "F") for v, ok in values)
    print(f"{build:26s} pass {sum(ok for _, ok in values)}/{len(values)}  mean {shifts.mean():+.3f}  {marks}")
