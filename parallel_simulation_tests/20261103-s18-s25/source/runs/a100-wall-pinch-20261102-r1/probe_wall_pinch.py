"""Probe a frozen mesh on A100; never restore a snapshot into a different DLL."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

GPU = 'GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'


def canonical(triangles):
    keys = triangles.astype(np.uint32) & np.uint32(0xbfffffff)
    start = keys.argmin(axis=1)
    oriented = np.take_along_axis(keys, (start[:, None]+np.arange(3)) % 3, axis=1)
    order = np.lexsort(oriented[:, ::-1].T)
    return oriented[order]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', required=True, type=Path)
    p.add_argument('--library', required=True, type=Path)
    p.add_argument('--snapshot', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise RuntimeError('Probe must use the authorized second A100')
    root = args.release.resolve(strict=True)
    sys.path[:0] = [str(root), str(root/'endoscopic_tissue_backups/2026-11-02-acceptance-fixes'),
                    str(root/'endoscopic_tissue_backups/2026-11-01-release-next')]
    from inspect_native_contact import Archive
    from audit_frozen_boundary import audit
    from endoscopic_tissue.training.snapshot import EnvSnapshot
    from endoscopic_tissue.training.data import atomic_json
    from endoscopic_tissue.training.provenance import sha256
    from endoscopic_tissue.native_backend import Surface
    from endoscopic_tissue.surface_diagnostics import topology_statistics
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(passed=False, gpu_uuid=GPU, library_sha256=sha256(args.library),
                  snapshot_sha256=sha256(args.snapshot), default_replaced=False,
                  cross_library_state_restore=False, scope='Frozen surface probe only')
    try:
        with args.snapshot.open('rb') as stream:
            _, size, _, _ = struct.unpack('<8sQQQ', stream.read(32))
            metadata = json.loads(stream.read(size))
        snap = EnvSnapshot.load(args.snapshot, dll_sha256=metadata['dll_sha256'],
                                physics_config_sha256=metadata['physics_config_sha256'])
        archive = Archive(snap.native_bytes())
        count = int(archive.array('recon.d_triCounter', '<u4')[0])
        original = archive.array('recon.d_tri', '<i4')[:count].reshape(-1, 3).copy()
        points = archive.array('cut.d_cutFP', '<f4').reshape(-1, 4).copy()
        external = archive.array('recon.d_voxelExternalFP', '<f4').reshape(-1, 4)
        dims = np.frombuffer(archive.field('recon.g_dims'), '<i4').copy()
        dll = ct.CDLL(str(args.library))
        function = dll.LCS_ProbePinchSurface
        ip = ct.POINTER(ct.c_int)
        up = ct.POINTER(ct.c_uint)
        fp = ct.POINTER(ct.c_float)
        function.argtypes = [ct.c_int, ip, ct.c_int, fp, ip, ip, up]
        function.restype = ct.c_int
        def run(triangles):
            triangles = np.ascontiguousarray(triangles, dtype=np.int32)
            output = np.empty_like(triangles)
            statistics = np.zeros(4, np.uint32)
            started = time.perf_counter()
            error = function(triangles.size, triangles.ctypes.data_as(ip), len(points),
                points.ctypes.data_as(fp), dims.ctypes.data_as(ip), output.ctypes.data_as(ip),
                statistics.ctypes.data_as(up))
            if error:
                raise RuntimeError(f'GPU surface probe failed: {error}')
            return output, statistics.tolist(), (time.perf_counter()-started)*1000
        def inspect(triangles):
            keys = triangles.astype(np.uint32)&np.uint32(0xbfffffff)
            unique, inv = np.unique(keys, return_inverse=True)
            inv = inv.reshape(-1)
            cut = (unique&0x80000000) != 0
            indices = unique&0x3fffffff
            verts = np.empty((len(unique), 3), np.float32)
            verts[cut] = points[indices[cut], :3]
            verts[~cut] = external[indices[~cut], :3]
            expanded = verts[inv]
            anchors = np.zeros((len(expanded), 4), np.float32)
            anchors[:, 3] = ((triangles.ravel().astype(np.uint32)>>30)&1)
            surface = Surface(expanded, np.zeros_like(expanded),
                              np.arange(len(expanded), dtype=np.int32).reshape(-1, 3), anchors)
            return audit(verts, inv.reshape(-1, 3), unique), topology_statistics(surface), surface
        before, before_quality, _ = inspect(original)
        output, stats, ms = run(original)
        after, after_quality, surface = inspect(output)
        result.update(before=before, after=after, before_quality=before_quality,
                      after_quality=after_quality, stats=stats, probe_including_upload_ms=ms)
        np.savez_compressed(args.output/'surface-after.npz', vertices=surface.vertices,
                            faces=surface.triangles, anchors=surface.anchors)
        atomic_json(args.output/'progress.json', result)
        assert before_quality['positional_nonmanifold_edges'] == 1, 'Expected retained one-edge counterexample'
        assert stats[1:3] == [1, 1], 'Expected exactly one accepted local repair'
        assert after['watertight'] and after['winding_consistent'] and not after['vertex_link_failures']
        for key in ('positional_boundary_edges', 'positional_nonmanifold_edges',
                    'duplicate_triangles', 'degenerate_triangles'):
            assert after_quality[key] == 0, key
        assert after_quality['winding_consistent']
        assert np.count_nonzero(np.any(original != output, axis=1)) == 4
        assert np.array_equal(np.unique(original), np.unique(output)), 'Feature set changed'
        second, second_stats, _ = run(output)
        assert np.array_equal(second, output) and second_stats[2] == 0, 'Repair must be idempotent'
        rng = np.random.default_rng(429)
        permutations = []
        for repeat in range(8):
            permutation = rng.permutation(len(original))
            rotated = np.take_along_axis(original, (rng.integers(0, 3, (len(original), 1))+np.arange(3)) % 3, axis=1)
            shuffled, counters, _ = run(rotated[permutation])
            assert counters[2] == 1 and np.array_equal(canonical(output), canonical(shuffled))
            permutations.append(True)
        # A skin/wall mixed ambiguity must not be treated as the retained wall
        # stencil. Removing the wall marker changes no geometric positions.
        changed = np.flatnonzero(np.any(original != output, axis=1))
        mixed = original.copy()
        mixed[changed[0]] = (mixed[changed[0]].astype(np.uint32)&np.uint32(0xbfffffff)).view(np.int32)
        rejected, rejected_stats, _ = run(mixed)
        assert np.array_equal(rejected, mixed) and rejected_stats[2] == 0
        result.update(passed=True, permutations_exact=len(permutations), idempotent=True,
                      unsupported_skin_mix_unchanged=True, positions_unchanged=True,
                      material_and_cut_records_untouched=True)
        return 0
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        atomic_json(args.output/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        raise SystemExit(main())
