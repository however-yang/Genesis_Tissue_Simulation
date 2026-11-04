"""Full original damage script plus strict audits of each refined surface."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--library-dir', type=Path, required=True)
    p.add_argument('--name', required=True)
    args, forwarded = p.parse_known_args()
    root = args.release.resolve(strict=True)
    sys.path[:0] = [str(root), str(root/'endoscopic_tissue_backups/2026-11-01-release-next')]
    from audit_frozen_boundary import audit
    from endoscopic_tissue.training.env import NativeEnvironment
    from endoscopic_tissue.training.data import atomic_json
    from endoscopic_tissue.training.provenance import sha256
    # Uploaded together in this isolated directory, never imported from a
    # modified production worker or a directory belonging to another project.
    import functional_worker
    out = root/'endoscopic_tissue/outputs/acceptance-fixes-20261102'/args.name
    events, saved = [], []
    failures, samples = 0, 0
    original = NativeEnvironment.replay_command
    def step(env, command):
        nonlocal failures, samples
        transition = original(env, command)
        if transition.after is None:
            return transition
        stats = np.zeros(4, np.uint32)
        read = env.native.dll.LCS_ReadBoundaryRefineStats
        read.argtypes = [ct.POINTER(ct.c_uint)]
        read.restype = ct.c_int
        error = read(stats.ctypes.data_as(ct.POINTER(ct.c_uint)))
        if error:
            raise RuntimeError(f'Boundary refinement telemetry unavailable: {error}')
        samples += 1
        if stats[1] or stats[2] or stats[3]:
            counts = np.zeros(5, np.int32)
            ip, up = ct.POINTER(ct.c_int), ct.POINTER(ct.c_uint)
            read_topology = env.native.dll.LCS_ReadSurfaceTopology
            read_topology.argtypes = [ip, ip, ct.c_int, up, ct.c_int]
            read_topology.restype = ct.c_int
            assert read_topology(counts.ctypes.data_as(ip), None, 0, None, 0) == 0
            tags, masks = np.empty(int(counts[4]), np.int32), np.empty(int(counts[3]), np.uint32)
            assert read_topology(counts.ctypes.data_as(ip), tags.ctypes.data_as(ip), len(tags), masks.ctypes.data_as(up), len(masks)) == 0
            surface = env.last_surface
            expanded = surface.vertices[surface.triangles.reshape(-1)]
            assert len(tags) == len(expanded)
            keys, first, inverse = np.unique(tags.astype(np.uint32)&np.uint32(0xbfffffff), return_index=True, return_inverse=True)
            result = audit(expanded[first], inverse.reshape(-1, 3), keys)
            passed = bool(result['watertight'] and result['winding_consistent'] and not result['vertex_link_failures'])
            failures += not passed
            event = dict(tick=env.native.tick, statistics=stats.tolist(), vertex_audit=result, passed=passed)
            events.append(event)
            out.mkdir(parents=True, exist_ok=True)
            atomic_json(out/'boundary-progress.json', dict(samples=samples, events=events, failures=failures))
            if stats[2] and not saved:
                directory = out/'first-refined-state'
                directory.mkdir(exist_ok=False)
                atomic_json(directory/'metadata.json', transition.after.metadata)
                np.savez_compressed(directory/'surface.npz', feature_ids=tags, vertices=surface.vertices,
                                    faces=surface.triangles, anchors=surface.anchors)
                env.snapshot().save(directory/'state.lcssnap')
                saved.append(env.native.tick)
            print('Boundary event', json.dumps(event), flush=True)
        return transition
    NativeEnvironment.replay_command = step
    sys.argv = ['functional_worker', '--release', str(root), '--library-dir', str(args.library_dir),
                '--mode', 'damage', '--name', args.name, '--save-quality-witnesses', *forwarded]
    exit_code = None
    try:
        exit_code = functional_worker.main()
        return 2 if failures else exit_code
    finally:
        NativeEnvironment.replay_command = original
        if out.is_dir():
            atomic_json(out/'boundary-audit.json', dict(samples=samples, events=events, failures=failures,
                saved_refined_states=saved, audit_passed=failures == 0,
                refinement_exercised=bool(saved), worker_sha256=sha256(__file__),
                performance_accepted=False, default_replaced=False))
            path = out/'result.json'
            if path.exists():
                record = json.loads(path.read_text())
                record['original_quality_passed'] = record.get('passed', False)
                record['boundary_vertex_audit_passed'] = failures == 0
                record['boundary_refinement_events'] = len(events)
                record['passed'] = record['original_quality_passed'] and failures == 0
                atomic_json(path, record)


if __name__ == '__main__':
    raise SystemExit(main())
