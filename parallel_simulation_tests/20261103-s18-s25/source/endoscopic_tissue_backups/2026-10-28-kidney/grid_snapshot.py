"""Save every array of the box-scene grids to compare them across a refactor of grid.py."""
import sys
from dataclasses import fields

import numpy as np

from endoscopic_tissue.grid import build_grid
from endoscopic_tissue.native_backend import Settings
from endoscopic_tissue.scenario import SCISSORS_SCENES, SCISSORS_SPACING, scissors_settings

arrays = {}
cases = {"knife": build_grid(.2, density=Settings().density_kg_m3, vertical_spacing=.05)}
for name, scene in SCISSORS_SCENES.items():
    cases[name] = build_grid(SCISSORS_SPACING, density=scissors_settings().density_kg_m3,
                             vertical_spacing=scene.vertical_spacing, tab=scene.tab)
for name, grid in cases.items():
    for f in fields(grid):
        value = getattr(grid, f.name)
        if isinstance(value, np.ndarray):
            arrays[f"{name}.{f.name}"] = value.view(np.uint8) if value.dtype.names else value
        else:
            arrays[f"{name}.{f.name}"] = np.asarray(value)
np.savez(sys.argv[1], **arrays)
print("saved", len(arrays), "arrays")
