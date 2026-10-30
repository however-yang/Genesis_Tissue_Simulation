"""Compare how the tumor deforms, stage by stage, between kidney runs that saved every stage snapshot: its extents, the
z thickness of its rest lattice columns across x at three heights, where the grasper grips it, and how far apart the
runs have drifted.

Usage: python fold_profile.py <run folder> [<run folder> ...]

The jaws close along z, so a crease down a pad face shows as a dip of the z thickness at some x.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from endoscopic_tissue import scenario  # noqa: E402

STAGES = ("lift", "blade_approach", "incise_1", "open_1", "reposition_2", "incise_2", "withdraw", "release", "extract",
          "hold")
grid = scenario.KIDNEY_SCENE.build_grid(1050)
rest = grid.positions
tumor = (grid.inside != 0) & (scenario.KIDNEY.tumor_weight(rest, 1e-3) > .5)
center = scenario.KIDNEY.tumor_center
columns = np.round((rest[tumor, 0] - center[0]) / .1).astype(int)
heights = rest[tumor, 1] - center[1]
bands = {"upper": (.15, .45), "middle": (-.15, .15), "lower": (-.45, -.15)}
runs = [Path(p) for p in sys.argv[1:]]
print(f"tumor nodes {tumor.sum()}, rest columns x {columns.min()}..{columns.max()} mm from the tumor axis")
for stage in STAGES:
    snapshots = [np.load(run / f"{stage}.npz") for run in runs]
    print(f"== {stage}")
    for run, snapshot in zip(runs, snapshots):
        nodes = snapshot["nodes_cm"][tumor]
        gripped = (snapshot["grip_flags"] != 0) & tumor
        grip_x = np.round((rest[gripped, 0] - center[0]) / .1).astype(int)
        line = (f"  {run.name}: gap {float(snapshot['jaw_gap_cm']) * 10:.2f} mm, extents "
                f"{np.ptp(nodes, axis=0).round(3)} cm, gripped tumor nodes {gripped.sum()}"
                + (f" at rest x {grip_x.min()}..{grip_x.max()} mm" if gripped.any() else ""))
        print(line)
        for band, (lo, hi) in bands.items():
            inside = (heights >= lo) & (heights < hi)
            profile = []
            for column in range(-7, 8):
                pick = inside & (columns == column)
                profile.append(np.ptp(nodes[pick, 2]) * 10 if pick.sum() > 1 else np.nan)
            print(f"    {band:6s} z thickness (mm) x=-7..7: " + " ".join(f"{v:5.1f}" for v in profile))
    if len(runs) > 1:
        gap = np.linalg.norm(snapshots[0]["nodes_cm"][tumor] - snapshots[1]["nodes_cm"][tumor], axis=1) * 10
        print(f"  drift between the first two runs: tumor nodes rms {np.sqrt(np.mean(gap ** 2)):.3f} mm, max {gap.max():.3f} mm")
