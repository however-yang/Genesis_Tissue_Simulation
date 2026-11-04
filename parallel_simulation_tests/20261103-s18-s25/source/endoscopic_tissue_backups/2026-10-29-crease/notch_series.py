"""Print, for each per-node dump of a kidney run (debug_crease_dump.py), how deep the +z face of the tumor is pinched in
at its centre column: the z thickness of the tumor's rest column x = 0 in its middle 3 mm, against the mean of the
columns at x = +-3 and +-4 mm. A smooth tumor has both near 12 mm at hold; a folded one keeps its centre near 8 mm.

Usage: python notch_series.py <run folder> [<run folder> ...]
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
    for path in sorted(Path(f"{run}-dump").glob("crease-*.bin")):
        n = np.fromfile(path, dtype=np.int32, count=1)[0]
        positions = np.fromfile(path, dtype=np.float32, offset=8, count=3 * n).reshape(n, 3)
        centre = np.ptp(positions[middle & (columns == 0), 2]) * 10
        sides = np.mean([np.ptp(positions[middle & (columns == c), 2]) for c in (-4, -3, 3, 4)]) * 10
        line.append(f"{path.stem.split('-')[1]}: {centre:4.1f}/{sides:4.1f}")
    print(f"{run.name}: centre/sides z thickness (mm) " + ", ".join(line))
