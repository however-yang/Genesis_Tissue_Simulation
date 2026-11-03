"""Trace stray outside links to cut records and actually exported feature IDs."""
import argparse
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys

import numpy as np

POINT = np.dtype([('offset', '<f4', 3), ('owner', '<i4'), ('edge', '<i4'), ('normal', '<f4', 3)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.release.resolve(strict=True)))
    from scipy.spatial import cKDTree
    from endoscopic_tissue.grid import CORNER_OFFSETS
    from endoscopic_tissue.scenario import SCISSORS_SCENES, scissors_settings, SCISSORS_SPACING
    from endoscopic_tissue.training.data import atomic_json
    from endoscopic_tissue.training.provenance import sha256
    report = json.loads((args.case/'report.json').read_text())
    scene = SCISSORS_SCENES[report['metadata']['scene']]
    settings = replace(scissors_settings(), **report['metadata']['settings'])
    grid = scene.build_grid(settings.density_kg_m3)
    dims, initial = np.asarray(grid.dims, np.int64), grid.neighbors
    cdims = dims+1
    inside = grid.inside != 0
    result = dict(case=str(args.case), stages=[], original_acceptance_unchanged=True,
                  physical_or_performance_accepted=False, worker_sha256=sha256(__file__))
    args.output.mkdir(parents=True, exist_ok=False)
    for path in sorted((args.case/'cut-witnesses').glob('*.npz')):
        with np.load(path, allow_pickle=False) as state:
            assert np.array_equal(state['rest'], grid.positions)
            assert np.array_equal(state['inside'], grid.inside)
            a, axes = np.where((initial[:, 1::2] >= 0) & (state['neighbors'][:, 1::2] < 0))
            b = initial[a, 2*axes+1]
            material = inside[a] | inside[b]
            mid = (grid.positions[a]+grid.positions[b])*.5
            distance = cKDTree(mid[material]).query(mid[~material], distance_upper_bound=1.5*SCISSORS_SPACING)[0]
            selected = np.flatnonzero(~material)[np.isinf(distance)]
            edges = state['edges']
            edge_base = edges[:, 1]+cdims[0]*(edges[:, 2]+cdims[1]*edges[:, 3])
            edge_lookup = np.full((len(grid.positions), 3), -1, np.int32)
            edge_lookup[edge_base, edges[:, 0]] = np.arange(len(edges), dtype=np.int32)
            points = state['points'].reshape(-1).view(POINT)
            used = set((state['feature_ids'].astype(np.uint32)&np.uint32(0xbfffffff)).tolist())
            records = []
            for index in selected:
                ca, cb, axis = int(a[index]), int(b[index]), int(axes[index])
                eid = int(edge_lookup[ca, axis])
                assert eid >= 0
                edge = edges[eid]
                across = [i for i in range(3) if i != axis]
                impacts = []
                for cp in points[points['edge'] == eid]:
                    owner = int(cp['owner'])
                    coord = np.array([owner % cdims[0], owner//cdims[0] % cdims[1], owner//(cdims[0]*cdims[1])])
                    for offsets in itertools.product((-1, 0), repeat=2):
                        cell = edge[1:].copy()
                        cell[across] += offsets
                        if np.any(cell < 0) or np.any(cell >= dims):
                            continue
                        vid = int(cell[0]+dims[0]*(cell[1]+dims[1]*cell[2]))
                        if not grid.occupied[vid]:
                            continue
                        candidates = np.flatnonzero(np.all(CORNER_OFFSETS == coord-cell, axis=1))
                        assert len(candidates) == 1
                        corner = int(candidates[0])
                        mask = int(state['boundary_masks'][vid]) & 4095
                        component = int(grid.connectivity[mask, corner+1])
                        key = 0x80000000 | (8*vid+component)
                        if key in used:
                            support = grid.corners[vid][grid.connectivity[mask, 1:] == component]
                            impacts.append(dict(owner=owner, voxel=vid, component=component,
                                feature_id=key, material_support_corners=int(inside[support].sum())))
                records.append(dict(a=ca, b=cb, axis=axis, edge_id=eid,
                    cut_t=float(state['cuts'][ca, 2*axis+1]),
                    native_material_interval_hit=False,
                    cut_records=int(np.count_nonzero(points['edge'] == eid)),
                    used_feature_contributions=impacts))
            item = dict(stage=path.stem, input_sha256=sha256(path), stray_skin_links=len(records),
                with_cut_records=sum(r['cut_records'] > 0 for r in records),
                contributing_to_exported_features=sum(bool(r['used_feature_contributions']) for r in records),
                native_cut_wall_emitter_accepts=0, links=records)
            result['stages'].append(item)
            print(json.dumps({k:v for k,v in item.items() if k != 'links'}), flush=True)
    atomic_json(args.output/'result.json', result)


if __name__ == '__main__':
    main()
