"""Read-only geometric sensitivity of the retained repair direction, not a solve."""
import json
from pathlib import Path
import struct

import numpy as np
from threadpoolctl import threadpool_limits

from inspect_native_contact import Archive, EnvSnapshot, ROOT

CASE = ROOT/'endoscopic_tissue/outputs/acceptance-fixes-20261102/regrasp-boundary-forensics-r1/first-failures/01-tick-954'


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    source = CASE/'state.lcssnap'
    with source.open('rb') as stream:
        _, m, _, _ = struct.unpack('<8sQQQ', stream.read(32))
        metadata = json.loads(stream.read(m))
    snapshot = EnvSnapshot.load(source, dll_sha256=metadata['dll_sha256'],
                                physics_config_sha256=metadata['physics_config_sha256'])
    archive = Archive(snapshot.native_bytes())
    pos = archive.array('xpbd.d_pos', '<f4').reshape(-1, 3)
    np.testing.assert_array_equal(pos, snapshot.python_state()['observed'].nodes)
    ids = archive.array('xpbd.d_tetIds', '<i4').reshape(-1, 4)
    live = archive.array('xpbd.d_tetActive', '<i4') != 0
    metric = archive.array('xpbd.d_unifiedRestMetric', '<f4').reshape(-1, 7)[:, 6]
    points = pos[ids]
    e = points[:, 1:]-points[:, :1]
    J = np.einsum('ij,ij->i', e[:, 0], np.cross(e[:, 1], e[:, 2]))*metric
    minimum = archive.scalar('xpbd.g_minTetVolume', '<f')
    assert abs(minimum-.15) < 1e-7, 'This diagnostic was reviewed for the frozen 0.15 floor'
    near = live & (J <= minimum+max(1e-5, 1e-3*minimum))
    weights = archive.array('xpbd.d_coarseWeight', '<f4').copy()
    inverse = archive.array('xpbd.d_invMass', '<f4')
    grip = archive.array('xpbd.d_gripDegree', '<i4') > 0
    active = archive.array('xpbd.d_activeNodes', '<i4')[:archive.scalar('xpbd.g_activeNodes')]
    parent = np.arange(len(pos))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    for tet in ids[near]:
        for i in tet[1:]:
            x, y = root(int(tet[0])), root(int(i))
            parent[max(x, y)] = min(x, y)
    groups = {}
    for i in active:
        groups.setdefault(root(int(i)), []).append(int(i))
    grouped = weights.copy()
    multi = []
    for nodes in groups.values():
        if len(nodes) < 2:
            continue
        nodes = np.asarray(nodes)
        pinned = bool(np.any(inverse[nodes] <= 0))
        held = nodes[grip[nodes]]
        value = 0. if pinned else float(weights[held if len(held) else nodes].mean(dtype=np.float64))
        grouped[nodes] = value
        multi.append({'nodes': len(nodes), 'pinned': pinned,
                      'original_weight_range': float(np.ptp(weights[nodes])), 'group_weight': value})

    def sensitivity(w):
        dw = w[ids[:, 1:]]-w[ids[:, :1]]
        g = (dw[:, 0, None]*np.cross(e[:, 1], e[:, 2])
             + dw[:, 1, None]*np.cross(e[:, 2], e[:, 0])
             + dw[:, 2, None]*np.cross(e[:, 0], e[:, 1]))*metric[:, None]
        return np.linalg.norm(g, axis=1)

    before, after = sensitivity(weights), sensitivity(grouped)
    block_type = np.dtype([('gradient', '<f4', (3,)), ('step', '<f4', (3,)),
                          *[(name, '<f4') for name in ('xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'scale')],
                          ('external', '<i4'), ('center', '<f4', (3,)), ('pivot', '<f4', (3,)),
                          ('mass', '<f4'), ('pins', '<i4')])
    assert block_type.itemsize == 88
    components = archive.array('xpbd.d_unifiedComponents', '<i4')
    global_blocks = archive.array('xpbd.d_globalCoarse', block_type)
    joint = archive.array('xpbd.d_coarseBlock', block_type)[0]
    used_roots = np.unique(components[ids[live, 0]])

    def block_record(b):
        return {'gradient': b['gradient'].tolist(), 'step_cm': b['step'].tolist(),
                'scale': float(b['scale']), 'applied_step_mm': (b['step']*b['scale']*10).tolist()}

    record = {'diagnostic_only': True, 'native_dll_loaded': False, 'candidate_adopted': False,
        'snapshot_sha256': snapshot.sha256, 'live_tets': int(live.sum()),
        'minimum_volume_ratio_from_snapshot': minimum,
        'saved_joint_block': block_record(joint),
        'saved_component_blocks': {str(int(i)): block_record(global_blocks[i]) for i in used_roots},
        'near_floor_tets': int(near.sum()), 'multi_node_clusters': multi,
        'near_floor_original_max_weight_span': float(np.ptp(weights[ids[near]], axis=1).max(initial=0)),
        'near_floor_original_max_dJ_per_cm': float(before[near].max(initial=0)),
        'near_floor_grouped_max_dJ_per_cm': float(after[near].max(initial=0)),
        'nodes_with_changed_weights': int(np.count_nonzero(weights != grouped)),
        'limits': ['Uses saved direction weights at the failure endpoint, not freshly reconstructed Newton directions.',
                   'CPU cluster averaging is float64 diagnostic arithmetic, not a GPU bitwise reproduction.',
                   'Zero dJ only proves a geometric property of this trial subspace. It does not prove contact improvement, full-step quality, timing, or S15 equivalence.']}
    target = CASE/'repair-weight-sensitivity-blocks.json'
    with target.open('x', encoding='utf8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in record.items() if k not in ('multi_node_clusters', 'limits')}))


if __name__ == '__main__':
    main()
