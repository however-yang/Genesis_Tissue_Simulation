"""Print and plot where a scissors run severed each material column of the cut box, in rest coordinates.

Usage: cut_levels.py <run dir> <scene> [png]. A column's level is the rest midpoint, along the plane axis, of its severed
material link. Columns severed twice print as '!!', uncut ones as '--'. Also counts severed links across the plane axis
(the steps between columns cut at different levels).
"""
import sys
from pathlib import Path

import numpy as np

from endoscopic_tissue.grid import build_grid
from endoscopic_tissue.scenario import SCISSORS_SCENES, SCISSORS_SPACING, scissors_settings

run, scene = Path(sys.argv[1]), SCISSORS_SCENES[sys.argv[2]]
grid = build_grid(SCISSORS_SPACING, density=scissors_settings().density_kg_m3, vertical_spacing=scene.vertical_spacing,
                  tab=scene.tab)
rest, inside, initial = grid.positions, grid.inside != 0, grid.neighbors
final = np.load(sorted(run.glob("*.npz"), key=lambda p: p.stat().st_mtime)[-1])["neighbors"]
axis = scene.plane_axis
across = [k for k in range(3) if k != axis]
severed = (initial >= 0) & (final < 0)
ids = np.flatnonzero(severed[:, 2 * axis + 1])
other = initial[ids, 2 * axis + 1]
ids, other = ids[inside[ids] | inside[other]], other[inside[ids] | inside[other]]
level = (rest[ids, axis] + rest[other, axis]) / 2
low, high = np.array(scene.cut_low)[across], np.array(scene.cut_high)[across]
steps = sum(int(np.count_nonzero(severed[inside, 2 * k + 1])) for k in across)
print(f"{len(ids)} severed links along the plane axis, {steps} across it; plane at {scene.plane} cm")
keys = np.round(rest[ids][:, across], 4)
rows = np.unique(keys[:, 1])
columns_x = np.unique(np.round(rest[inside][:, across[0]], 4))
columns_x = columns_x[(columns_x > low[0]) & (columns_x < high[0])]
profile = {}
for row in rows:
    if not low[1] < row < high[1]:
        continue
    cells = []
    for x in columns_x:
        hits = level[(keys[:, 0] == x) & (keys[:, 1] == row)]
        cells.append("--" if len(hits) == 0 else "!!" if len(hits) > 1 else f"{(hits[0] - scene.plane) * 10:+.1f}")
        profile.setdefault(row, []).append(np.nan if len(hits) != 1 else (hits[0] - scene.plane) * 10)
    print(f"row {row:.2f}: " + " ".join(f"{c:>5s}" for c in cells))
print("columns along x:", " ".join(f"{x:.2f}" for x in columns_x))
if len(sys.argv) > 3:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    for row, values in profile.items():
        ax.step(columns_x * 10, values, where="mid", label=f"depth {row * 10:.0f} mm")
    ax.axhline(0, color="gray", lw=.8, ls="--")
    ax.set_xlabel("position along the wound (mm)")
    ax.set_ylabel("cut level vs. planned plane (mm)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(sys.argv[3], dpi=150)
