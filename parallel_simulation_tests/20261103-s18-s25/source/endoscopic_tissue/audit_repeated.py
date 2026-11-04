"""Audit saved repeated-cut checkpoints independently of the native solver."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .grid import build_grid
from .probe_cut_topology import material_cut_edges
from .surface_diagnostics import load_state, topology_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--include-debug', action='store_true')
    args = parser.parse_args()
    grid = build_grid(.2, vertical_spacing=.05)
    fractions = np.load(args.directory / "cut-fractions.npy")
    material = material_cut_edges(grid, fractions)
    audits = []
    for path in sorted(args.directory.glob('*.npz')):
        if path.name.startswith('debug-') and not args.include_debug:
            continue
        obs, surface = load_state(path)
        live = obs.neighbors >= 0
        rows = np.broadcast_to(np.arange(len(obs.nodes))[:, None], live.shape)[live]
        cols = obs.neighbors[live]
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(obs.nodes), len(obs.nodes)))
        _, labels = connected_components(graph)
        sizes = np.bincount(labels, weights=grid.active)
        roots = np.flatnonzero(sizes > 0)
        main_root = roots[np.argmax(sizes[roots])]
        detached = (grid.active != 0) & (labels != main_root)
        audit = topology_statistics(surface)
        removed = (grid.neighbors >= 0) & (obs.neighbors < 0)
        audit.update(material_cut_edges=int(np.count_nonzero((removed & material)[:, 1::2])),
                     air_cut_edges=int(np.count_nonzero((removed & ~material)[:, 1::2])), stage=path.stem, material_components=len(roots), detached_nodes=int(detached.sum()),
                     detached_inside_nodes=int(np.count_nonzero(detached & (grid.inside != 0))),
                     detached_min_y_mm=float(obs.nodes[detached, 1].min()) * 10 if np.any(detached) else None,
                     max_displacement_mm=float(np.linalg.norm(obs.nodes - grid.positions, axis=1).max()) * 10,
                     grip_points=obs.grip_points)
        audits.append(audit)
    (args.directory / 'checkpoint-audit.json').write_text(json.dumps(audits, indent=2), encoding='utf8')
    failed = [item['stage'] for item in audits if item['degenerate_triangles']
              or item['positional_boundary_edges'] or item['positional_nonmanifold_edges']
              or item['duplicate_triangles'] or not item['winding_consistent']
              or item['max_displacement_mm'] > 60]
    if not audits:
        failed.append('no checkpoints')
    print(json.dumps({'checkpoints': len(audits), 'invalid_checkpoints': failed,
                      'report': str(args.directory / 'checkpoint-audit.json')}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
