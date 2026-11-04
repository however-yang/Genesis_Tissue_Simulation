"""List the intact links where the shell around one material piece meets the shell around another.

Each shell node belongs to the material piece it reaches in the fewest shell links; a link whose two ends belong to
different pieces ties them together. Usage: python bridge_check.py <run output dir> [stage]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from endoscopic_tissue.scenario import KIDNEY_PLANE_Y, KIDNEY_SCENE

out = Path(sys.argv[1])
stage = sys.argv[2] if len(sys.argv) > 2 else "hold"
grid = KIDNEY_SCENE.build_grid(1050)
with np.load(out / f"{stage}.npz") as data:
    neighbors, nodes = data["neighbors"], data["nodes_cm"]
inside = grid.inside != 0
n = len(neighbors)
ids = np.repeat(np.arange(n), 6)
others = neighbors.reshape(-1)
live = others >= 0
a, b = ids[live], others[live]
material = inside[a] & inside[b]
graph = coo_matrix((np.ones(np.count_nonzero(material)), (a[material], b[material])), shape=(n, n))
pieces = connected_components(graph, directed=False)[1]
material_pieces = np.unique(pieces[inside])
print(f"{stage}: material pieces {len(material_pieces)}")
# Shell-only graph plus links from material into shell; distances from each piece's material nodes.
shell_links = ~(inside[a] & inside[b])
shell_graph = coo_matrix((np.ones(np.count_nonzero(shell_links)), (a[shell_links], b[shell_links])), shape=(n, n)).tocsr()
distances = []
for piece in material_pieces:
    sources = np.flatnonzero(inside & (pieces == piece))
    # A super source: link it to every material node of the piece through a virtual node.
    rows = np.concatenate((shell_graph.nonzero()[0], np.full(len(sources), n)))
    cols = np.concatenate((shell_graph.nonzero()[1], sources))
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n + 1, n + 1)).tocsr()
    distances.append(dijkstra(g, directed=False, indices=n, unweighted=True)[:n])
distances = np.array(distances)
owner = np.where(inside, pieces, material_pieces[np.argmin(distances, axis=0)])
reached = np.isfinite(distances).any(axis=0)
seam = (owner[a] != owner[b]) & reached[a] & reached[b] & (a < b)
print(f"intact links between the two sides: {np.count_nonzero(seam)}")
dims = np.array(grid.dims) + 1
for s, t in zip(a[seam][:30], b[seam][:30]):
    axis = int(np.argmax(np.abs(grid.positions[t] - grid.positions[s])))
    crossing = (nodes[s, 1] - KIDNEY_PLANE_Y) * (nodes[t, 1] - KIDNEY_PLANE_Y) < 0
    print(f"  {'yxz'[[1, 0, 2].index(axis)] if False else 'xyz'[axis]}-link rest {grid.positions[s].round(3)} -> {grid.positions[t].round(3)} "
          f"present y {nodes[s, 1]:.3f} -> {nodes[t, 1]:.3f} inside {inside[s]}/{inside[t]} crosses plane {crossing}")
