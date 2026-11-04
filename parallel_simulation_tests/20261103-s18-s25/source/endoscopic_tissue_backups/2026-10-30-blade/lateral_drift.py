"""Tabulate, per build, the largest offset of the scissors from their requested pose along the blade width axis during
the snips of every flap run under the current folder and its snip3 and blade subfolders, by outcome.

Usage: python lateral_drift.py   (in outputs/stage56-runs)
"""
import collections, glob, json, os
import numpy as np
from endoscopic_tissue.native_backend import scissors_frame
by = collections.defaultdict(list)
for traj in sorted(glob.glob("*/trajectory.jsonl")) + sorted(glob.glob("snip3/*/trajectory.jsonl")) + sorted(glob.glob("blade/*/trajectory.jsonl")):
    folder = os.path.dirname(traj)
    try:
        rep = json.load(open(folder + "/report.json"))
    except OSError:
        continue
    if rep.get("scene") not in (None, "flap") and "flap" not in str(rep.get("scene")):
        continue
    rows = [json.loads(l) for l in open(traj)]
    if not rows or rows[0]["command"].get("scissors_pitch_deg") is None:
        continue
    if not any(r["stage"] == "incise_3" for r in rows):
        continue
    if abs(rows[0]["command"]["blade_center"][0] - 5.6) > 1e-3:   # flap scene parks at x 5.6
        continue
    drift = {}
    for r in rows:
        if not r["stage"].startswith(("incise", "withdraw")):
            continue
        b = np.array(r["blade_center_cm"]); q = np.array(r["requested_command"]["blade_center"])
        axis, normal = scissors_frame(r["command"]["scissors_pitch_deg"])
        width = np.cross(axis, normal)
        key = r["stage"]
        drift[key] = max(drift.get(key, 0), abs((b - q) @ width) * 10)
    name = os.path.basename(folder)
    build = name.split("-", 2)[2] if name.count("-") >= 2 else name
    tag = name.split("-")[0]
    if tag in ("om", "op", "n0"):
        build += f" ({tag})"
    by[build].append((rep["status"] == "passed", drift.get("incise_1", 0), drift.get("incise_2", 0), drift.get("incise_3", 0)))
for build, runs in sorted(by.items()):
    ok = [r for r in runs if r[0]]; bad = [r for r in runs if not r[0]]
    fmt = lambda rs: " ".join(f"{max(r[1:]):.2f}" for r in rs)
    print(f"{build:24s} pass {len(ok)}/{len(runs)} | max lateral drift per run (mm): pass [{fmt(ok)}]  fail [{fmt(bad)}]")
