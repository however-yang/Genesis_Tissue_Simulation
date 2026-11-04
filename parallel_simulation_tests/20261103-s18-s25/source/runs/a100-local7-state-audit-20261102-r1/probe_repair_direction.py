"""Offline feasibility study of a retained three-DOF repair direction.

No DLL, time step, or source physics is changed. Current positions follow the
last regional solve, so this diagnoses a candidate mechanism rather than
reproducing the precise earlier limiter or proving a dynamic repair.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np
from scipy.optimize import minimize, LinearConstraint
from threadpoolctl import threadpool_limits


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.release), str(args.release/'endoscopic_tissue_backups/2026-11-02-acceptance-fixes')]
    from inspect_native_contact import Archive
    from endoscopic_tissue.training.snapshot import EnvSnapshot
    from endoscopic_tissue.training.provenance import sha256
    with args.snapshot.open('rb') as f:
        _, n, _, _ = struct.unpack('<8sQQQ', f.read(32))
        metadata = json.loads(f.read(n))
    assert metadata['dll_sha256'] == '6dbc03fab2427f8a5579ac349aea4a3a04b0663f5a55d572f8db81966d20a0e2'
    snap = EnvSnapshot.load(args.snapshot, dll_sha256=metadata['dll_sha256'],
                            physics_config_sha256=metadata['physics_config_sha256'])
    archive = Archive(snap.native_bytes())
    v3 = ('<f4', (3,))
    coarse = np.dtype([('gradient', *v3), ('step', *v3)] +
        [(n, '<f4') for n in ('xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'scale')] +
        [('external', '<i4'), ('center', *v3), ('pivot', *v3), ('mass', '<f4'), ('pins', '<i4')])
    safe = np.dtype([('weight', '<f4'), ('gripWeight', '<f4'), ('count', '<i4'),
                     ('grips', '<i4'), ('pinned', '<i4')])
    pos = archive.array('xpbd.d_pos', '<f4').reshape(-1, 3).astype(np.float64)
    roots = archive.array('xpbd.d_unifiedComponents', '<i4')[:len(pos)]
    cluster = archive.array('xpbd.d_safeCluster', '<i4')[:len(pos)]
    active = archive.array('xpbd.g_liveActive', '<i4')[:len(pos)] != 0
    assert ((cluster[active] >= 0) & (cluster[active] < len(pos))).all()
    ag = archive.array('xpbd.d_safeAggregate', safe)[:len(pos)][np.where(active, cluster, 0)]
    assert archive.array('xpbd.d_safeJoined', '<i4')[0]
    weight = np.where(ag['pinned'], 0., np.where(ag['grips'] > 0,
        ag['gripWeight']/np.maximum(ag['grips'], 1), ag['weight']/np.maximum(ag['count'], 1)))
    ids = archive.array('xpbd.d_tetIds', '<i4').reshape(-1, 4)
    live = archive.array('xpbd.d_tetActive', '<i4') != 0
    metric = archive.array('xpbd.d_unifiedRestMetric', '<f4').reshape(-1, 7)
    blocks = archive.array('xpbd.d_globalCoarse', coarse)
    minimum = archive.scalar('xpbd.g_minTetVolume', '<f')
    records = []
    for root in np.unique(roots[ids[live, 0]]):
        t = np.flatnonzero(live & (roots[ids[:, 0]] == root))
        nodes = ids[t]
        e = pos[nodes[:, 1:]]-pos[nodes[:, :1]]
        dw = weight[nodes[:, 1:]]-weight[nodes[:, :1]]
        cross = np.stack([np.cross(e[:, 1], e[:, 2]), np.cross(e[:, 2], e[:, 0]),
                          np.cross(e[:, 0], e[:, 1])], axis=1)
        volume = np.einsum('ij,ij->i', e[:, 0], cross[:, 0])*metric[t, 6]
        A = (cross*dw[:, :, None]).sum(axis=1)*metric[t, 6, None]
        # Equal scalar weights are a pure translation with identically zero dJ.
        valid = np.linalg.norm(A, axis=1) > 1e-10
        t, nodes, volume, A = t[valid], nodes[valid], volume[valid], A[valid]
        margin = np.maximum(0., .9*(volume-minimum))
        b = blocks[root]
        H = np.array([[b['xx'], b['xy'], b['xz']], [b['xy'], b['yy'], b['yz']],
                      [b['xz'], b['yz'], b['zz']]], dtype=float)
        g = b['gradient'].astype(float)
        assert np.linalg.eigvalsh(H).min() > 0
        newton = -np.linalg.solve(H, g)
        derivative = A@newton
        alpha = np.ones(len(A))
        np.divide(margin, -derivative, out=alpha, where=derivative < 0)
        alpha = np.clip(alpha, 0, 1)
        clipped = newton*alpha.min(initial=1)
        scale = max(float(np.linalg.norm(g)), 1.)
        selected = set(np.argsort(A@newton+margin)[:32].tolist())
        direction = np.zeros(3)
        history = []
        for iteration in range(20):
            use = np.array(sorted(selected))
            result = minimize(lambda d: (.5*d@H@d+g@d)/scale, direction,
                jac=lambda d: (H@d+g)/scale,
                constraints=[LinearConstraint(A[use], -margin[use], np.inf)],
                method='SLSQP', options={'ftol': 1e-12, 'maxiter': 200})
            direction = result.x
            residual = A@direction+margin
            history.append(dict(iteration=iteration, constraints=len(use), optimizer_success=bool(result.success),
                                message=result.message, worst_residual=float(residual.min())))
            if residual.min() >= -1e-9:
                break
            selected.update(np.argsort(residual)[:32].tolist())
        constraints_passed = bool((A@direction+margin >= -1e-9).all())
        def objective(d): return float(.5*d@H@d+g@d)
        indices = np.argsort(alpha)[:12]
        records.append(dict(root=int(root), tet_constraints=len(A), current_min_J=float(volume.min()),
            retained_scale=float(b['scale']), current_direction_scale=float(alpha.min(initial=1)),
            newton_cm=newton.tolist(), clipped_cm=clipped.tolist(), projected_cm=direction.tolist(),
            quadratic_change_clipped=objective(clipped), quadratic_change_projected=objective(direction),
            constraint_check_passed=constraints_passed, history=history,
            limiting_tets=[dict(tet=int(t[i]), J=float(volume[i]), alpha=float(alpha[i]),
                dJ=float(derivative[i]), nodes=nodes[i].tolist(), weights=weight[nodes[i]].tolist(),
                near_floor=float(volume[i]) <= minimum+max(1e-5, 1e-3*minimum)) for i in indices]))
    report = dict(diagnostic_only=True, current_positions_after_retained_solve=True,
        includes_contact_trust_region=False, native_library_loaded=False, candidate_adopted=False,
        snapshot_sha256=sha256(args.snapshot), worker_sha256=sha256(__file__), minimum_volume=minimum,
        records=records)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
