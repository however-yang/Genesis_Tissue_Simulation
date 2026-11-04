"""Compare the movable node positions of runs at one stage: each pair's largest and median distance in mm.

Usage: python spread.py <stage> <run dir> <run dir> [...]
"""
import itertools
import sys
from pathlib import Path

import numpy as np

stage, runs = sys.argv[1], [Path(p) for p in sys.argv[2:]]
nodes = {}
for run in runs:
    with np.load(run / f"{stage}.npz") as data:
        nodes[run.name] = data["nodes_cm"]
first = next(iter(nodes.values()))
with np.load(runs[0] / "settle.npz") as data:
    moved = np.linalg.norm(first - data["nodes_cm"], axis=1) > 0
for a, b in itertools.combinations(nodes, 2):
    distance = np.linalg.norm(nodes[a] - nodes[b], axis=1)[moved] * 10
    print(f"{a:>12s} vs {b:<12s} max {distance.max():.4f} mm  p99 {np.percentile(distance, 99):.4f} mm  median {np.median(distance):.5f} mm")
