"""Diagnose the unchanged repair objective from a retained same-library state.

Extra calls are diagnostic only, never installed in the frame loop. This asks
whether an existing overlap is repair convergence or an apparent equilibrium.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', required=True, type=Path)
    p.add_argument('--library-dir', required=True, type=Path)
    p.add_argument('--snapshot', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    root = args.release.resolve(strict=True)
    sys.path[:0] = [str(root), str(root / 'endoscopic_tissue_backups/2026-11-01-s15-s17')]
    from threadpoolctl import threadpool_limits
    from validation_support import environment
    from endoscopic_tissue.instrument_geometry import load_scissors, scissors_body_depths
    from endoscopic_tissue.training.data import atomic_json
    from endoscopic_tissue.training.provenance import json_hash
    from endoscopic_tissue.training.snapshot import EnvSnapshot, SnapshotBackend, binding
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(diagnostic_only=True, default_replaced=False, simulation_time_advanced=False,
                  snapshot=str(args.snapshot), samples=[], completed=False)
    try:
        with threadpool_limits(limits=1, user_api='blas'), environment(dll=str(args.library_dir)) as env:
            env.reset(0)
            snap = EnvSnapshot.load(args.snapshot, dll_sha256=env.context['dll_sha256'],
                                    physics_config_sha256=json_hash(binding(env)))
            env.restore(snap, branch_label='isolated-repair-diagnostic')
            assert SnapshotBackend(env.native).save() == snap.native_bytes()
            record.update(native_import_exact=True, dll_sha256=env.context['dll_sha256'],
                          snapshot_sha256=snap.sha256, native_tick=env.native.tick)
            geometry = load_scissors()
            original = env.native.observe().nodes.copy()
            initial_surface = env.native.surface()
            def measure(calls, elapsed):
                observed, surface = env.native.observe(), env.native.surface()
                pose = observed.scissors
                body, wrist = scissors_body_depths(surface.vertices, surface.triangles, geometry,
                                                   pose.hinge, pose.axis, pose.normal, pose.gap)
                item = dict(repair_calls=calls, elapsed_ms=elapsed, body_overlap_mm=float(body*10),
                            wrist_overlap_mm=float(wrist*10), native_depth_mm=env.native.contact_depth_m()*1000,
                            material_quality=env.native.material_quality().tolist(),
                            max_node_motion_mm=float(np.linalg.norm(observed.nodes-original, axis=1).max()*10),
                            scissors_hinge_cm=pose.hinge.tolist(), scissors_gap_mm=float(pose.gap*10))
                record['samples'].append(item)
                atomic_json(args.output/'result.json', record)
                print(json.dumps(item), flush=True)
            measure(0, 0.)
            elapsed = 0.
            for i in range(1, 9):
                started = time.perf_counter()
                env.native.check('LCS_RefineContactSurface', env.settings.contact_refine_passes)
                elapsed += (time.perf_counter()-started)*1000
                if i in (1, 2, 4, 8):
                    measure(i, elapsed)
            surface = env.native.surface()
            np.savez_compressed(args.output/'before-after.npz', before_nodes=original,
                                after_nodes=env.native.observe().nodes,
                                before_vertices=initial_surface.vertices, before_faces=initial_surface.triangles,
                                after_vertices=surface.vertices, after_faces=surface.triangles)
            record['completed'] = True
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        atomic_json(args.output/'result.json', record)


if __name__ == '__main__':
    main()
