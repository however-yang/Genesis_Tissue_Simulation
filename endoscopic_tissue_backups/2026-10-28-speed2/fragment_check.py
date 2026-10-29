"""Locate the small lattice pieces of a kidney run and show which snip severed each of their links.

Usage: python fragment_check.py <run output dir>
"""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from endoscopic_tissue.scenario import KIDNEY_SCENE

run = Path(sys.argv[1])
grid = KIDNEY_SCENE.build_grid(1050)
rest, inside, active = grid.positions, grid.inside != 0, grid.active != 0
stages = {name: np.load(run / f"{name}.npz") for name in ("blade_approach", "incise_1", "incise_2", "hold")
          if (run / f"{name}.npz").exists()}
final = stages["hold"]["neighbors"] if "hold" in stages else stages["incise_2"]["neighbors"]
n = len(final)
rows = np.repeat(np.arange(n), 6)
cols = final.reshape(-1)
keep = cols >= 0
pieces = connected_components(coo_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(n, n)), directed=False)[1]
labels, counts = np.unique(pieces[active], return_counts=True)
for label in labels[counts < 100]:
    members = np.flatnonzero(active & (pieces == label))
    print(f"piece {label}: {len(members)} nodes, {inside[members].sum()} inside")
    for i in members:
        print(f"  node {i} rest {rest[i].round(3)} inside {inside[i]}")
        for slot in range(6):
            j = stages["blade_approach"]["neighbors"][i, slot]
            if j < 0:
                continue
            when = "intact"
            for name in ("incise_1", "incise_2", "hold"):
                if name in stages and stages[name]["neighbors"][i, slot] < 0:
                    when = f"cut by {name}"
                    break
            print(f"    slot {slot} -> {j} rest {rest[j].round(3)} inside {inside[j]}: {when}")
