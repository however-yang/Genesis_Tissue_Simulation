"""Inspect retained R7 regional weights, without executing a simulation.

The final coarseWeight array belongs to mode zero. SafeCoarse aggregates retain
the prior regional direction instead. These are last-iteration diagnostics,
not a reconstruction of all iterations or of the current exact gradient.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np
from threadpoolctl import threadpool_limits


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--contact-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.release.resolve(strict=True)
    sys.path[:0] = [str(root), str(root/'endoscopic_tissue_backups/2026-11-02-acceptance-fixes')]
    from inspect_native_contact import Archive
    from endoscopic_tissue.training.snapshot import EnvSnapshot
    from endoscopic_tissue.training.provenance import sha256
    with args.snapshot.open('rb') as stream:
        _, n, _, _ = struct.unpack('<8sQQQ', stream.read(32))
        metadata = json.loads(stream.read(n))
    assert metadata['dll_sha256'] == '6dbc03fab2427f8a5579ac349aea4a3a04b0663f5a55d572f8db81966d20a0e2'
    snap = EnvSnapshot.load(args.snapshot, dll_sha256=metadata['dll_sha256'],
                            physics_config_sha256=metadata['physics_config_sha256'])
    archive = Archive(snap.native_bytes())
    contact = json.loads(args.contact_audit.read_text())
    assert Path(contact['snapshot']).resolve() == args.snapshot.resolve()
    assert contact['dll_sha256'] == metadata['dll_sha256']
    v3 = ('<f4', (3,))
    coarse = np.dtype([('gradient', *v3), ('step', *v3)] +
        [(n, '<f4') for n in ('xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'scale')] +
        [('external', '<i4'), ('center', *v3), ('pivot', *v3), ('mass', '<f4'), ('pins', '<i4')])
    safe = np.dtype([('weight', '<f4'), ('gripWeight', '<f4'), ('count', '<i4'),
                     ('grips', '<i4'), ('pinned', '<i4')])
    assert coarse.itemsize == 88 and safe.itemsize == 20
    positions = archive.array('xpbd.d_pos', '<f4').reshape(-1, 3)
    n = len(positions)
    roots = archive.array('xpbd.d_unifiedComponents', '<i4')[:n]
    active = archive.array('xpbd.g_liveActive', '<i4')[:n] != 0
    cluster = archive.array('xpbd.d_safeCluster', '<i4')[:n]
    aggregates = archive.array('xpbd.d_safeAggregate', safe)[:n]
    blocks = archive.array('xpbd.d_globalCoarse', coarse)[:n]
    grip = archive.array('xpbd.d_gripDegree', '<i4')[:n] > 0
    joined = int(archive.array('xpbd.d_safeJoined', '<i4')[0])
    assert joined, 'No retained SafeCoarse aggregates; do not infer exact prior weights'
    assert np.all((cluster[active] >= 0) & (cluster[active] < n))
    a = aggregates[cluster[active]]
    assert (a['count'] > 0).all()
    values = np.where(a['pinned'] != 0, 0., np.where(a['grips'] > 0,
        a['gripWeight']/np.maximum(a['grips'], 1), a['weight']/a['count']))
    weights = np.full(n, np.nan)
    weights[active] = values
    assert np.isfinite(weights[active]).all()
    basis = blocks[0]
    record = dict(snapshot=str(args.snapshot), snapshot_sha256=sha256(args.snapshot),
        dll_sha256=metadata['dll_sha256'], worker_sha256=sha256(__file__),
        contact_audit_sha256=sha256(args.contact_audit), native_library_loaded=False,
        retained_regional_only=True, safe_joined=joined,
        local_basis_scratch=dict(active=bool(basis['external']), center_cm=basis['center'].tolist(),
                                radius_squared_cm2=float(basis['mass'])),
        total_grip_nodes=int(grip.sum()),
        grip_weights=[float(weights[grip].min()), float(weights[grip].max())] if grip.any() else [],
        clusters_with_grips=int(np.count_nonzero(aggregates['grips'])), faces=[])
    for face in contact['faces']:
        entries = []
        for linear in face['linearizations']:
            for item in linear['manifold']:
                ids = np.asarray(item['ids'])
                if not len(ids):
                    continue
                assert active[ids].all()
                coeff = np.asarray(item['weights'])
                aggregate = aggregates[cluster[ids]]
                displacement = (coeff[:, None]*weights[ids, None]*
                                blocks[roots[ids]]['step']*blocks[roots[ids]]['scale'][:, None]).sum(axis=0)
                entries.append(dict(weight_min_max=[float(weights[ids].min()), float(weights[ids].max())],
                    interpolated_weight=float(np.dot(coeff, weights[ids])),
                    support_in_grip_cluster=int(np.count_nonzero(aggregate['grips'])),
                    support_in_pinned_cluster=int(np.count_nonzero(aggregate['pinned'])),
                    support_nodes_with_grip=int(grip[ids].sum()),
                    cluster_sizes=aggregate['count'].tolist(),
                    regional_normal_displacement_mm=float(np.dot(displacement, item['normal'])*10),
                    retained_normal_gap_mm=item['normal_gap_mm'],
                    retained_pressure_n=item['evaluated_pressure_n_at_reference_pose']))
        record['faces'].append(dict(face=face['face'], depth_mm=face['depth_mm'], contacts=entries))
    record['blocks'] = [dict(root=int(i), step_cm=blocks[i]['step'].tolist(), scale=float(blocks[i]['scale']),
        gradient=blocks[i]['gradient'].tolist(), diagonal=[float(blocks[i][k]) for k in ('xx', 'yy', 'zz')])
        for i in np.unique(roots[active])]
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
