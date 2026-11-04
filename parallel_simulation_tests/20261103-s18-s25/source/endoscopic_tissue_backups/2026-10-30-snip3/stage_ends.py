"""Print hinge, jaw gap and grip at the end of chosen stages for flap run folders."""
import json
import sys
from pathlib import Path

stages = sys.argv[1].split(",")
for folder in map(Path, sys.argv[2:]):
    rows = [json.loads(line) for line in open(folder / "trajectory.jsonl")]
    last = {}
    for row in rows:
        last[row["stage"]] = row
    status = json.load(open(folder / "report.json"))["status"]
    parts = []
    for name in stages:
        r = last.get(name)
        if r is None:
            continue
        h = r["hinge_cm"]
        c = r["command"]["hinge"]
        parts.append(f"{name}: hy {h[1]:.3f}/{c[1]:.3f} hx {h[0]:.3f}/{c[0]:.3f} gap {r['jaw_gap_cm']*10:.2f} "
                     f"grip {r['grip_points']} err {r['grip_error_m']*1000:.2f}")
    print(f"{folder.name} {status[:4]} | " + " | ".join(parts))
