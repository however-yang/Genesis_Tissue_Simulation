"""Print the tumor's pinch at the stage snapshots of kidney runs: the z thickness of the tumor's rest column x = 0 in its
middle 3 mm against the mean of the columns at x = +-3 and +-4 mm, at reposition_2 (before the last snip, under
traction) and at hold (excised tumor held up).

Usage: python stage_notch.py <run folder> [<run folder> ...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from endoscopic_tissue import scenario  # noqa: E402

grid = scenario.KIDNEY_SCENE.build_grid(1050)
rest = grid.positions
center = scenario.KIDNEY.tumor_center
tumor = (grid.inside != 0) & (scenario.KIDNEY.tumor_weight(rest, 1e-3) > .5)
columns = np.round((rest[:, 0] - center[0]) / .1).astype(int)
middle = tumor & (np.abs(rest[:, 1] - center[1]) < .15)
for run in map(Path, sys.argv[1:]):
    line = []
    for stage in ("reposition_2", "hold"):
        path = run / f"{stage}.npz"
        if not path.exists():
            continue
        nodes = np.load(path)["nodes_cm"]
        centre = np.ptp(nodes[middle & (columns == 0), 2]) * 10
        sides = np.mean([np.ptp(nodes[middle & (columns == c), 2]) for c in (-4, -3, 3, 4)]) * 10
        line.append(f"{stage} {centre:4.1f}/{sides:4.1f}")
    print(f"{run.name}: centre/sides z thickness (mm) " + ", ".join(line))
