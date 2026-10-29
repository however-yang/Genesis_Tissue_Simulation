"""Compare probe_refine.py records of several runs: per stage group, how many steps needed contact repair, how deep the
contact surface was before and after it, and the repair time.

Usage: python refine_stats.py <tolerance cm> <label>=<jsonl>[,<jsonl>...] ...
"""
import json
import sys

import numpy as np

GROUPS = {"held": ("close", "lift", "blade_approach"), "cutting": ("incise_1", "open_1", "reposition_2", "incise_2"),
          "post-excision": ("withdraw", "release", "extract", "hold")}
tolerance = float(sys.argv[1])
for argument in sys.argv[2:]:
    label, paths = argument.split("=", 1)
    records = [json.loads(line) for path in paths.split(",") for line in open(path)]
    print(f"{label}: {len(paths.split(','))} runs")
    for group, stages in GROUPS.items():
        rows = [r for r in records if r["stage"] in stages]
        before = np.array([r["depth_before_cm"] for r in rows])
        after = np.array([r["depth_after_cm"] for r in rows])
        refine = np.array([r["refine_ms"] for r in rows])
        repaired = before > tolerance
        print(f"  {group:14s} steps {len(rows):5d}  repaired {repaired.mean():6.1%}  "
              f"depth before p50/p95/max {np.median(before) * 10:.3f}/{np.percentile(before, 95) * 10:.3f}/{before.max() * 10:.3f} mm  "
              f"after p95/max {np.percentile(after, 95) * 10:.3f}/{after.max() * 10:.3f} mm  "
              f"left above tolerance {(after > tolerance).mean():6.1%}  refine {refine.mean():5.2f} ms/step")
