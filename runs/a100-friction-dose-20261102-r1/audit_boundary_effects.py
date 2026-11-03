"""Measure mask-mediated effects of reported ghost links on frozen native data.

This counterfactual changes NumPy copies only. It never changes simulation,
acceptance criteria, cut locations, or the original failed reports.
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
    p.add_argument('--witness-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.release.resolve(strict=True)))
    from endoscopic_tissue.grid import EDGE_CORNERS
    from endoscopic_tissue.scenario import SCISSORS_SCENES, scissors_settings
    from endoscopic_tissue.training.data import atomic_json
    from endoscopic_tissue.training.provenance import sha256
    report = json.loads((args.case/'report.json').read_text())
    settings = replace(scissors_settings(), **report['metadata']['settings'])
    grid = SCISSORS_SCENES[report['metadata']['scene']].build_grid(settings.density_kg_m3)
    witness = json.loads(args.witness_audit.read_text())
    assert Path(witness['case']).resolve() == args.case.resolve()
    dims = np.asarray(grid.dims, np.int64)
    cdims = dims+1
    corners_inside = grid.inside[grid.corners] != 0
    air = np.zeros(len(grid.corners), np.uint32)
    for bit, (a, b) in enumerate(EDGE_CORNERS):
        air |= (~(corners_inside[:, a] & corners_inside[:, b])).astype(np.uint32) << bit
    result = dict(case=str(args.case), worker_sha256=sha256(__file__),
        witness_audit_sha256=sha256(args.witness_audit), stages=[],
        original_acceptance_unchanged=True, counterfactual_only=True,
        physical_or_performance_accepted=False)
    args.output.mkdir(parents=True, exist_ok=False)
    for stage in witness['stages']:
        path = args.case/'cut-witnesses'/(stage['stage']+'.npz')
        assert sha256(path) == stage['input_sha256']
        with np.load(path, allow_pickle=False) as state:
            assert np.array_equal(state['rest'], grid.positions)
            assert np.array_equal(state['inside'], grid.inside)
            actual = state['actual_masks'] & np.uint32(4095)
            boundary = state['boundary_masks'] & np.uint32(4095)
            expected = np.where(actual != 0, actual | air, 0).astype(np.uint32)
            assert np.array_equal(boundary, expected), 'Unexpected native boundary construction'
            removed = actual.copy()
            bits = []
            for link in stage['links']:
                ca, cb, axis = (int(link[k]) for k in ('a', 'b', 'axis'))
                assert not grid.inside[ca] and not grid.inside[cb]
                assert link['cut_records'] == 0 and not link['used_feature_contributions']
                base = np.array([ca % cdims[0], ca//cdims[0] % cdims[1], ca//(cdims[0]*cdims[1])])
                across = [i for i in range(3) if i != axis]
                for offset in itertools.product((-1, 0), repeat=2):
                    coord = base.copy()
                    coord[across] += offset
                    if np.any(coord < 0) or np.any(coord >= dims):
                        continue
                    v = int(coord[0]+dims[0]*(coord[1]+dims[1]*coord[2]))
                    if not grid.occupied[v]:
                        continue
                    local_a = int(np.flatnonzero(grid.corners[v] == ca)[0])
                    local_b = int(np.flatnonzero(grid.corners[v] == cb)[0])
                    candidates = [i for i, pair in enumerate(EDGE_CORNERS)
                                  if set(map(int, pair)) == {local_a, local_b}]
                    assert len(candidates) == 1
                    bit = candidates[0]
                    assert actual[v] & (1 << bit)
                    removed[v] &= np.uint32(4095 ^ (1 << bit))
                    bits.append((v, bit))
            hypothetical = np.where(removed != 0, removed | air, 0).astype(np.uint32)
            tags = np.unique(state['feature_ids'].astype(np.uint32) & np.uint32(0xbfffffff))
            cut = (tags & np.uint32(0x80000000)) != 0
            indices = tags & np.uint32(0x3fffffff)
            voxels = np.where(cut, indices//8, indices)
            records = []
            for v in np.flatnonzero(actual != removed):
                v = int(v)
                old, new = int(actual[v]), int(removed[v])
                old_groups, new_groups = grid.connectivity[[old, new], 1:]
                selected = np.flatnonzero(corners_inside[v])
                old_partition = old_groups[selected, None] == old_groups[None, selected]
                new_partition = new_groups[selected, None] == new_groups[None, selected]
                used = tags[voxels == v]
                support_changes = []
                for tag in used:
                    if not tag & np.uint32(0x80000000):
                        continue
                    component = int(tag & np.uint32(7))
                    old_support = np.flatnonzero(grid.connectivity[int(boundary[v]), 1:] == component)
                    if len(old_support) == 0:
                        raise ValueError('Exported feature lacks its original support')
                    new_component = grid.connectivity[int(hypothetical[v]), int(old_support[0])+1]
                    new_support = np.flatnonzero(grid.connectivity[int(hypothetical[v]), 1:] == new_component)
                    if not np.array_equal(old_support, new_support):
                        support_changes.append(dict(feature_id=int(tag), old=old_support.tolist(),
                                                    hypothetical=new_support.tolist()))
                records.append(dict(voxel=v, original_actual=old, hypothetical_actual=new,
                    original_boundary=int(boundary[v]), hypothetical_boundary=int(hypothetical[v]),
                    boundary_changed=bool(boundary[v] != hypothetical[v]),
                    material_partition_changed=not np.array_equal(old_partition, new_partition),
                    exported_features=list(map(int, used)), support_changes=support_changes))
            item = dict(stage=stage['stage'], input_sha256=sha256(path),
                stray_links=len(stage['links']), affected_occupied_cells=len(records),
                boundary_changed_cells=sum(r['boundary_changed'] for r in records),
                used_boundary_changed_cells=sum(r['boundary_changed'] and bool(r['exported_features']) for r in records),
                material_partition_changed_cells=sum(r['material_partition_changed'] for r in records),
                used_features_with_changed_support=sum(len(r['support_changes']) for r in records),
                native_contact_cache_keys_changed_cells=sum(bool(r['exported_features']) for r in records),
                note='Contact cache keys include actual masks; reconstruction and interpolation use boundary masks. '
                     'This frozen comparison is not a dynamic proof and does not remove ghost-link release.',
                cells=records)
            result['stages'].append(item)
            print(json.dumps({k:v for k,v in item.items() if k != 'cells'}), flush=True)
    atomic_json(args.output/'result.json', result)


if __name__ == '__main__':
    main()
