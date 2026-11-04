"""Re-evaluate the column acceptance of finished scissors runs from their last saved state with the work copy's code."""
import sys
from pathlib import Path

import numpy as np

from endoscopic_tissue.grid import build_grid
from endoscopic_tissue.native_backend import Observation
from endoscopic_tissue.scenario import SCISSORS_SCENES, SCISSORS_SPACING, Sequence, scissors_settings

for run, name in zip(sys.argv[1::2], sys.argv[2::2]):
    scene = SCISSORS_SCENES[name]
    settings = scissors_settings()
    grid = build_grid(SCISSORS_SPACING, density=settings.density_kg_m3, vertical_spacing=scene.vertical_spacing, tab=scene.tab)
    last = np.load(sorted(Path(run).glob("*.npz"), key=lambda p: p.stat().st_mtime)[-1])
    initial = Observation(grid.positions, grid.neighbors, np.zeros(len(grid.positions)), np.zeros(3), 1., 0, 0., np.zeros(10),
                          np.zeros(3), None)
    sequence = Sequence(initial, settings, scene, grid.positions, grid.inside)
    positions, cuts = sequence.column_cuts(last["neighbors"], scene.cut_low, scene.cut_high)
    print(Path(run).name, "columns", len(cuts), "uncut", int((cuts == 0).sum()), "cut twice", int((cuts > 1).sum()),
          "uncut at x", sorted(set(np.round(positions[cuts == 0, 0], 2))))
    print("   stray skin cuts", sequence.stray_skin_cuts(last["neighbors"]))
