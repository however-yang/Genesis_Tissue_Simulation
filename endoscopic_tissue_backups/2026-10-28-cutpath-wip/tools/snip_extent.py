"""Print, per snip, the x range of the material columns it severed (rest coordinates), for one or more runs."""
import sys
from pathlib import Path

import numpy as np

from endoscopic_tissue.grid import build_grid
from endoscopic_tissue.scenario import SCISSORS_SCENES, SCISSORS_SPACING, scissors_settings

for run, name in zip(sys.argv[1::2], sys.argv[2::2]):
    scene = SCISSORS_SCENES[name]
    grid = build_grid(SCISSORS_SPACING, density=scissors_settings().density_kg_m3, vertical_spacing=scene.vertical_spacing,
                      tab=scene.tab)
    inside, axis = grid.inside != 0, scene.plane_axis
    files = sorted(Path(run).glob("*.npz"), key=lambda p: p.stat().st_mtime)
    previous = grid.neighbors
    out = []
    for f in files:
        links = np.load(f)["neighbors"]
        if f.stem.startswith("incise"):
            ids = np.flatnonzero((previous[:, 2 * axis + 1] >= 0) & (links[:, 2 * axis + 1] < 0))
            other = previous[ids, 2 * axis + 1]
            ids = ids[inside[ids] | inside[other]]
            x = grid.positions[ids, 0]
            out.append(f"{f.stem}: {len(ids)} links x {x.min():.2f}-{x.max():.2f}" if len(ids) else f"{f.stem}: none")
        previous = links
    print(Path(run).name, " | ".join(out))
