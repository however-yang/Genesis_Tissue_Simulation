"""Report how a kidney run's stage left the tissue: its pieces, where the tumor and the grasp went, and the cut.

Usage: python excision_check.py <run output dir> [stage]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from endoscopic_tissue.scenario import KIDNEY_SCENE

out = Path(sys.argv[1])
stage = sys.argv[2] if len(sys.argv) > 2 else "hold"
kidney = KIDNEY_SCENE.shape
grid = KIDNEY_SCENE.build_grid(1050)
with np.load(out / f"{stage}.npz") as data:
    neighbors, nodes, grip = data["neighbors"], data["nodes_cm"], data["grip_flags"] != 0
inside = grid.inside != 0
ids = np.flatnonzero(inside)
rows, cols = np.repeat(ids, 6), neighbors[ids].reshape(-1)
keep = (cols >= 0) & inside[np.maximum(cols, 0)]
graph = coo_matrix((np.ones(np.count_nonzero(keep)), (rows[keep], cols[keep])), shape=(len(neighbors),) * 2)
labels = connected_components(graph, directed=False)[1]
pieces, sizes = np.unique(labels[ids], return_counts=True)
tumor = inside & (kidney.tumor_weight(grid.positions, 1e-3) > .5)
pinned = grid.pinned != 0
print(f"{stage}: {len(pieces)} material pieces, sizes {sizes.tolist()}")
for piece, size in zip(pieces, sizes):
    members = labels == piece
    print(f"  piece of {size} nodes: tumor nodes {np.count_nonzero(members & tumor)}/{np.count_nonzero(tumor)}, "
          f"pinned {np.count_nonzero(members & pinned)}, gripped {np.count_nonzero(members & grip)}, "
          f"present y {nodes[members & inside, 1].min():.2f}..{nodes[members & inside, 1].max():.2f}")
severed = (grid.neighbors >= 0) & (neighbors < 0)
a, slot = np.nonzero(severed[:, 1::2])
b = grid.neighbors[a, 2 * slot + 1]
material = inside[a] | inside[b]
mid = (grid.positions[a] + grid.positions[b]) / 2
print(f"  severed links {len(a)} (material {np.count_nonzero(material)}), by axis {np.bincount(slot, minlength=3).tolist()}")
if np.any(material):
    m = mid[material]
    print(f"  severed material rest x {m[:, 0].min():.2f}..{m[:, 0].max():.2f}, y {m[:, 1].min():.2f}..{m[:, 1].max():.2f}, "
          f"z {m[:, 2].min():.2f}..{m[:, 2].max():.2f}; through tumor {np.count_nonzero(tumor[a] & tumor[b] & material)}")
report = json.loads((out / "report.json").read_text(encoding="utf8"))
metrics = report["metrics"] or {}
print("  report:", report["status"], report["last_stage"], report["error"])
for key in ("lift_m", "fractures", "two_blade_squeeze_m", "fracture_left_n", "fracture_right_n", "max_blade_body_overlap_m",
            "max_knife_body_penetration_m", "peak_blade_force_n", "uncut_columns", "multiply_cut_columns",
            "stray_skin_cuts", "edges_cut_outside_tab", "edges_cut_without_fracture", "new_edges_during_withdrawal",
            "material_components", "grip_lost_after_capture", "final_grip_points"):
    print(f"    {key} = {metrics.get(key)}")
print("    snips:", metrics.get("snips"))
audits = json.loads((out / "surface-audit.json").read_text(encoding="utf8"))
print("  non-manifold edges by stage:", {a["stage"]: a.get("recorded_nonmanifold_edges", a["positional_nonmanifold_edges"])
                                         for a in audits if a.get("recorded_nonmanifold_edges", a["positional_nonmanifold_edges"])})
