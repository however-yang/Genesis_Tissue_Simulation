"""Print the rest z of the severed z-link of every tab column (its cut level link) for probed flap runs.

Usage: python level_map.py <run folder>...   One character per column: the lower node's rest z of its severed link
in tenths of a millimetre relative to the plane layer z = 0 ('0' for the link from z = 0 to 0.1 cm, '+' above it,
'-' below it, '.' uncut, '*' several), rows per rest y (top first), columns per rest x.
"""
import json
import sys
from pathlib import Path

import numpy as np

for folder in map(Path, sys.argv[1:]):
    data = np.load(str(folder) + ".npz")
    rest, inside, initial = data["rest"], data["inside"] != 0, data["initial"]
    cut_steps, cut_nodes, cut_slots = data["cut_steps"], data["cut_nodes"], data["cut_slots"]
    stage_of_step = {json.loads(l)["step"]: json.loads(l)["stage"] for l in open(folder / "trajectory.jsonl")}
    levels = {}
    for tick, node, slot in zip(cut_steps, cut_nodes, cut_slots):
        if slot != 5 or not (inside[node] or inside[initial[node, 5]]):
            continue
        x, y, z = (round(float(v), 3) for v in rest[node])
        if not (3.0 < x < 3.8 and .2 < y < .8):
            continue
        levels.setdefault((x, y), []).append((z, stage_of_step.get(int(tick), "?")))
    xs = sorted({k[0] for k in levels} | {3.1, 3.2, 3.3, 3.4, 3.5})
    ys = sorted({k[1] for k in levels} | {.35, .4, .45, .5, .55, .6, .65})
    print(f"== {folder.name}")
    for y in ys[::-1]:
        row = ""
        for x in xs:
            entries = levels.get((x, y), [])
            row += "." if not entries else "*" if len(entries) > 1 else "0" if abs(entries[0][0]) < 1e-3 else "+" if entries[0][0] > 0 else "-"
        print(f"   y {y:.2f} {row}")
    print("   x " + " ".join(f"{x:.1f}" for x in xs))
    odd = sorted((k, v) for k, v in levels.items() if len(v) > 1 or abs(v[0][0]) > 1e-3)
    for k, v in odd:
        print(f"   column {k}: {v}")
