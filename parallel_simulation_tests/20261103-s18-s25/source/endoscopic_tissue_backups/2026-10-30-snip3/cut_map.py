"""Print, for probed flap runs, which snip severed each material column of the cut box and how the flap sits before
each snip.

Usage: python cut_map.py <run folder>...   (each with its <folder>.npz from flap_probe.py beside it)

The map has one row per rest y layer (top first) and one character per rest x position of the tab: the snip number
that severed the column, '.' for an uncut column, '*' for a column severed more than once. The flap pose is the mean
deformed position of the tab's material nodes and of its free-end and root faces.
"""
import json
import sys
from pathlib import Path

import numpy as np

AXIS, LOW, HIGH = 2, np.array((3.0, .3, -1.0)), np.array((3.6, .7, 1.0))
for folder in map(Path, sys.argv[1:]):
    data = np.load(str(folder) + ".npz")
    rest, inside, initial = data["rest"], data["inside"] != 0, data["initial"]
    names = list(data["names"])
    stage_of_step = {}
    for line in open(folder / "trajectory.jsonl"):
        row = json.loads(line)
        stage_of_step[row["step"]] = row["stage"]
    report = json.load(open(folder / "report.json"))
    slot = 2 * AXIS + 1
    # Material links along the column axis inside the box, as in Sequence.column_links.
    ids = np.flatnonzero(initial[:, slot] >= 0)
    ids = ids[inside[ids] | inside[initial[ids, slot]]]
    box = np.all((rest[ids] > np.array((3.0, .2, -1.2)) + 1e-3) & (rest[ids] < np.array((3.8, .8, 1.2)) - 1e-3), axis=1)
    ids = ids[box]
    keys, column = np.unique(np.round(rest[ids][:, :2], 4), axis=0, return_inverse=True)
    column = column.reshape(-1)
    snip_of = np.zeros(len(keys), dtype=int)
    count = np.zeros(len(keys), dtype=int)
    level = np.full(len(keys), np.nan)
    cut_steps, cut_nodes, cut_slots = data["cut_steps"], data["cut_nodes"], data["cut_slots"]
    position = {node: k for k, node in enumerate(ids)}
    per_snip = {}
    for tick, node, s in zip(cut_steps, cut_nodes, cut_slots):
        stage = stage_of_step.get(int(tick), "?")
        per_snip.setdefault(stage, [0, 0])[0 if s // 2 == AXIS else 1] += 1
        if s == slot and node in position:
            k = column[position[node]]
            count[k] += 1
            snip_of[k] = int(stage.split("_")[1]) if stage.startswith("incise_") else 9
            level[k] = rest[node, AXIS]
    xs, ys = np.unique(keys[:, 0]), np.unique(keys[:, 1])
    print(f"== {folder.name}: {report['status']}, uncut {report['metrics']['uncut_columns']}, severed links per stage "
          f"(along z, across) {per_snip}")
    for y in ys[::-1]:
        line = ""
        for x in xs:
            k = np.flatnonzero((keys[:, 0] == x) & (keys[:, 1] == y))
            if not len(k):
                line += " "
                continue
            k = k[0]
            line += "." if count[k] == 0 else "*" if count[k] > 1 else str(snip_of[k])
        print(f"   y {y:.3f} {line}")
    print("   x " + " ".join(f"{x:.2f}" for x in xs[::2]))
    tab = inside & np.all((rest > LOW - 1e-3) & (rest < HIGH + 1e-3), axis=1)
    for name in ("lift", "blade_approach", "reposition_2", "reposition_3"):
        if name not in names:
            continue
        nodes = data["stage_nodes"][names.index(name)]
        parts = []
        # Deformed position of the two node layers that straddle the shear plane z = 0.05, per tab column x.
        for x in (3.1, 3.2, 3.3, 3.5):
            near = lambda value, target: np.abs(value - target) < 1e-4
            lower = tab & near(rest[:, 0], x) & near(rest[:, 2], 0.0)
            upper = tab & near(rest[:, 0], x) & near(rest[:, 2], 0.1)
            both = lower | upper
            shift = (nodes - rest) * 10
            parts.append(f"x{x}: dz {shift[lower, 2].mean():+.2f}/{shift[upper, 2].mean():+.2f} "
                         f"dy {shift[both, 1].mean():+.2f} dx {shift[both, 0].mean():+.2f}")
        print(f"   {name:14s} " + " | ".join(parts))
