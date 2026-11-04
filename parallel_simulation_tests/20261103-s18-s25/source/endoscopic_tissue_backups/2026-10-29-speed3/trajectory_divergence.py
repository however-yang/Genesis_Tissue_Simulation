"""Compare how fast pairs of wall runs diverge: two runs of one build against runs of the two builds.

Usage: python trajectory_divergence.py

The solver's float atomics make two runs of one build differ from the first contact on. If the new build changes the
physics only at rounding level, a pair of runs of different builds diverges at the same rate as a pair of one build.
"""
import json
from pathlib import Path

import numpy as np

folder = Path(r"D:\Simulation\genesis-world\endoscopic_tissue\outputs\speed2-runs")


def load(name):
    rows = [json.loads(line) for line in (folder / name / "trajectory.jsonl").read_text().splitlines()]
    state = np.array([r["hinge_cm"] + [r["jaw_gap_cm"]] + r["blade_center_cm"] for r in rows])
    return state, [r["stage"] for r in rows]


def divergence(a, b):
    sa, stages = load(a)
    sb, _ = load(b)
    n = min(len(sa), len(sb))
    gap = np.abs(sa[:n] - sb[:n]).max(axis=1)
    first = {threshold: int(np.argmax(gap > threshold)) if (gap > threshold).any() else None for threshold in (1e-6, 1e-4, 1e-2)}
    per_stage = {}
    for k in range(n):
        per_stage[stages[k]] = max(per_stage.get(stages[k], 0.0), gap[k])
    return first, per_stage


pairs = [("ab-base-wall-1", "ab-base-wall-2"), ("ab-base-wall-2", "ab-base-wall-3"), ("ab-new-wall-1", "ab-new-wall-2"),
         ("ab-new-wall-2", "ab-new-wall-3"), ("ab-base-wall-1", "ab-new-wall-1"), ("ab-base-wall-2", "ab-new-wall-2"),
         ("ab-base-wall-3", "ab-new-wall-3")]
stage_order = None
for a, b in pairs:
    first, per_stage = divergence(a, b)
    if stage_order is None:
        stage_order = list(per_stage)
        print("stages:", " ".join(stage_order))
    print(f"{a} vs {b}: first step above 1e-6/1e-4/1e-2 cm: {first[1e-6]}/{first[1e-4]}/{first[1e-2]}")
    print("   max gap per stage (cm):", " ".join(f"{per_stage[s]:.1e}" for s in stage_order))
