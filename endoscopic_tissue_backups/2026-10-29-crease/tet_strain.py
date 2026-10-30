"""Report the deformation of the tumor's lattice tetrahedra stage by stage: how many are inverted or nearly flat, where
they sit at rest, and the spread of their volume ratio and smallest principal stretch.

Usage: python tet_strain.py <run folder> [<run folder> ...]

The tetrahedra are the six Kuhn tetrahedra of every lattice cell whose eight corners are active, as in grid.py. Rest
positions are in cm with x and z measured from the tumor axis and y from the tumor centre.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from endoscopic_tissue import scenario  # noqa: E402

STAGES = ("lift", "reposition_2", "incise_2", "withdraw", "hold")
grid = scenario.KIDNEY_SCENE.build_grid(1050)
rest = grid.positions
binary_corners = grid.corners[:, (0, 1, 3, 2, 4, 5, 7, 6)]
supported = np.all(grid.active[binary_corners] != 0, axis=1)
kuhn = np.array(((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)))
tets = binary_corners[supported][:, kuhn].reshape(-1, 4)
centroids = rest[tets].mean(axis=1)
in_tumor = scenario.KIDNEY.tumor_weight(centroids, 1e-3) > .5
tets, centroids = tets[in_tumor], centroids[in_tumor] - scenario.KIDNEY.tumor_center
rest_edges = rest[tets[:, 1:]] - rest[tets[:, :1]]
inverse = np.linalg.inv(np.transpose(rest_edges, (0, 2, 1)))
print(f"tumor tetrahedra {len(tets)}")
for stage in STAGES:
    print(f"== {stage}")
    for run in map(Path, sys.argv[1:]):
        nodes = np.load(run / f"{stage}.npz")["nodes_cm"]
        edges = nodes[tets[:, 1:]] - nodes[tets[:, :1]]
        gradient = np.transpose(edges, (0, 2, 1)) @ inverse
        volume = np.linalg.det(gradient)
        stretch = np.linalg.svd(gradient, compute_uv=False)
        smallest = stretch[:, -1] * np.sign(volume)
        flat = volume < .2
        line = (f"  {run.name}: J p1/p50/p99 {np.percentile(volume, 1):.3f}/{np.median(volume):.3f}/"
                f"{np.percentile(volume, 99):.3f}, inverted {np.count_nonzero(volume <= 0)}, J<0.2 {np.count_nonzero(flat)}, "
                f"smallest stretch p1/p50 {np.percentile(smallest, 1):.3f}/{np.median(smallest):.3f}, "
                f"largest stretch p99/max {np.percentile(stretch[:, 0], 99):.2f}/{stretch[:, 0].max():.2f}")
        print(line)
        if flat.any():
            spots = np.round(centroids[flat] * 10).astype(int)
            values, counts = np.unique(spots, axis=0, return_counts=True)
            order = np.argsort(-counts)[:8]
            print("    flat tets at rest (x, y, z) mm: " + ", ".join(f"{tuple(values[k])}x{counts[k]}" for k in order))
