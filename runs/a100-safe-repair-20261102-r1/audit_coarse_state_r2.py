"""Read retained coarse-step bounds without loading a simulation library."""
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
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--contact-audit', type=Path)
    args = p.parse_args()
    root = args.release.resolve(strict=True)
    sys.path[:0] = [str(root), str(root/'endoscopic_tissue_backups/2026-11-02-acceptance-fixes')]
    from inspect_native_contact import Archive
    from endoscopic_tissue.training.snapshot import EnvSnapshot
    from endoscopic_tissue.training.provenance import sha256
    with args.snapshot.open('rb') as stream:
        _, m, _, _ = struct.unpack('<8sQQQ', stream.read(32))
        metadata = json.loads(stream.read(m))
    snap = EnvSnapshot.load(args.snapshot, dll_sha256=metadata['dll_sha256'],
                            physics_config_sha256=metadata['physics_config_sha256'])
    archive = Archive(snap.native_bytes())
    v3 = ('<f4', (3,))
    dtype = np.dtype([('gradient', *v3), ('step', *v3)] +
        [(n, '<f4') for n in ('xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'scale')] +
        [('external', '<i4'), ('center', *v3), ('pivot', *v3), ('mass', '<f4'), ('pins', '<i4')])
    assert dtype.itemsize == 88
    pos = archive.array('xpbd.d_pos', '<f4').reshape(-1, 3)
    roots = archive.array('xpbd.d_unifiedComponents', '<i4')[:len(pos)]
    active = archive.array('xpbd.g_liveActive', '<i4')[:len(pos)] != 0
    blocks = archive.array('xpbd.d_globalCoarse', dtype)
    assert len(blocks) == len(pos), 'CoarseBlock layout/allocation mismatch'
    records = []
    for index in np.unique(roots[active]):
        b = blocks[index]
        values = np.concatenate((b['gradient'], b['step'], [b['scale']]))
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite retained block')
        records.append(dict(root=int(index), nodes=int(np.count_nonzero(active & (roots == index))),
            gradient=b['gradient'].tolist(), step_cm=b['step'].tolist(), scale=float(b['scale']),
            bounded_step_mm=float(np.linalg.norm(b['step'])*b['scale']*10)))
    result = dict(snapshot=str(args.snapshot), snapshot_sha256=sha256(args.snapshot),
        dll_sha256=metadata['dll_sha256'], worker_sha256=sha256(__file__),
        global_blocks=records, native_library_loaded=False,
        interpretation='Retained last regional coarse block, not a per-iteration trace. '
                       'Final positions follow additional vertex solves and surface reconstruction. '
                       'Scale also includes contact/energy restrictions; small scale alone does not identify its cause.')
    weights = archive.array('xpbd.d_coarseWeight', '<f4')[:len(pos)]
    grip = archive.array('xpbd.d_gripDegree', '<i4')[:len(pos)] > 0
    result['retained_grip_weights'] = dict(nodes=int(grip.sum()),
        min=float(weights[grip].min()) if grip.any() else None,
        max=float(weights[grip].max()) if grip.any() else None)
    if args.contact_audit:
        contact = json.loads(args.contact_audit.read_text())
        assert Path(contact['snapshot']).resolve() == args.snapshot.resolve()
        result['contact_audit_sha256'] = sha256(args.contact_audit)
        result['retained_failed_contact_weights'] = []
        for face in contact['faces']:
            for linear in face['linearizations']:
                for c in linear['manifold']:
                    ids = np.asarray(c['ids'])
                    if not len(ids):
                        continue
                    result['retained_failed_contact_weights'].append(dict(
                        face=face['face'], min=float(weights[ids].min()), max=float(weights[ids].max()),
                        weighted=float(np.dot(weights[ids], c['weights'])),
                        contact_nodes_with_grips=int(grip[ids].sum())))
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
