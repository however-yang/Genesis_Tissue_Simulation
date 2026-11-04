"""Classify the existing stray-skin diagnostic without changing its verdict.

An outside-outside lattice edge is not necessarily an edge of the exported
surface. Report its incident occupied/intersected cells and material owners.
"""
import argparse
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.release.resolve(strict=True)
    sys.path.insert(0, str(root))
    from scipy.spatial import cKDTree
    from endoscopic_tissue.scenario import SCISSORS_SCENES, scissors_settings, SCISSORS_SPACING
    from endoscopic_tissue.training.data import atomic_json
    from endoscopic_tissue.training.provenance import sha256
    report = json.loads((args.case/'report.json').read_text())
    scene = SCISSORS_SCENES[report['metadata']['scene']]
    settings = replace(scissors_settings(), **report['metadata']['settings'])
    grid = scene.build_grid(settings.density_kg_m3)
    dims = np.asarray(grid.dims, np.int64)
    cdims = dims+1
    material = np.asarray(grid.inside, bool)
    initial = grid.neighbors
    result = dict(case=str(args.case), original_acceptance_unchanged=True,
                  physical_or_performance_accepted=False, stages=[],
                  worker_sha256=sha256(__file__))
    args.output.mkdir(parents=True, exist_ok=False)
    for stage in ['incise_1', 'incise_2', 'hold', 'open_retract']:
        path = args.case/(stage+'.npz')
        with np.load(path, allow_pickle=False) as state:
            rest, neighbors = state['rest_cm'], state['neighbors']
            assert np.array_equal(rest, grid.positions)
            assert np.array_equal(state['inside'], grid.inside)
            a, axes = np.where((initial[:, 1::2] >= 0) & (neighbors[:, 1::2] < 0))
            b = initial[a, 2*axes+1]
            tissue = material[a] | material[b]
            mid = (rest[a]+rest[b])*.5
            distance = cKDTree(mid[tissue]).query(mid[~tissue], distance_upper_bound=1.5*SCISSORS_SPACING)[0]
            selected = np.flatnonzero(~tissue)[np.isinf(distance)]
            records = []
            for index in selected:
                ca, cb, axis = int(a[index]), int(b[index]), int(axes[index])
                base = np.array([ca % cdims[0], ca//cdims[0] % cdims[1], ca//(cdims[0]*cdims[1])])
                across = [i for i in range(3) if i != axis]
                cells = []
                for offsets in itertools.product((-1, 0), repeat=2):
                    coord = base.copy()
                    coord[across] += offsets
                    if np.all(coord >= 0) and np.all(coord < dims):
                        vid = int(coord[0]+dims[0]*(coord[1]+dims[1]*coord[2]))
                        cells.append(dict(id=vid, occupied=bool(grid.occupied[vid]),
                                          original_surface_intersections=int(grid.counts[vid])))
                assert len(cells) > 0
                records.append(dict(a=ca, b=cb, axis=axis, rest_midpoint_mm=(mid[index]*10).tolist(),
                    active_endpoints=[bool(grid.active[ca]), bool(grid.active[cb])],
                    incident_cells=cells, has_surface_support=any(
                        c['occupied'] or c['original_surface_intersections'] for c in cells)))
            item = dict(stage=stage, input_sha256=sha256(path), stray_skin_count=len(records),
                incident_surface_support_count=sum(r['has_surface_support'] for r in records),
                outside_surface_support_count=sum(not r['has_surface_support'] for r in records),
                links=records)
            result['stages'].append(item)
            print(json.dumps({k:v for k,v in item.items() if k != 'links'}), flush=True)
    atomic_json(args.output/'result.json', result)


if __name__ == '__main__':
    main()
