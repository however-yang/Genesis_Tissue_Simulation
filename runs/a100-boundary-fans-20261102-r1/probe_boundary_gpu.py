"""A100 frozen-state geometry and material-binding checks for R5, without restore."""
import argparse
import ctypes as ct
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
    keys = triangles.astype(np.uint32)&np.uint32(0xbfffffff)
    start = keys.argmin(axis=1)
    oriented = np.take_along_axis(keys, (start[:, None]+np.arange(3)) % 3, axis=1)
    return oriented[np.lexsort(oriented[:, ::-1].T)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
    sys.path[:0] = [str(args.release), str(args.release/'endoscopic_tissue_backups/2026-11-02-acceptance-fixes'),
                    str(args.release/'endoscopic_tissue_backups/2026-11-01-release-next')]
    from inspect_native_contact import Archive
    from audit_frozen_boundary import audit
    from endoscopic_tissue.training.snapshot import EnvSnapshot
    from endoscopic_tissue.training.data import atomic_json
    from endoscopic_tissue.training.provenance import sha256
    from endoscopic_tissue.native_backend import Surface
    from endoscopic_tissue.surface_diagnostics import topology_statistics
    from scipy.spatial import cKDTree
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(passed=False, gpu_uuid=GPU, library_sha256=sha256(args.library),
                  snapshot_sha256=sha256(args.snapshot), default_replaced=False,
                  cross_library_restore=False, scope='Frozen geometry only, not dynamic acceptance')
    try:
        with args.snapshot.open('rb') as f:
            _, nmeta, _, _ = struct.unpack('<8sQQQ', f.read(32))
            meta = json.loads(f.read(nmeta))
        snap = EnvSnapshot.load(args.snapshot, dll_sha256=meta['dll_sha256'], physics_config_sha256=meta['physics_config_sha256'])
        state = Archive(snap.native_bytes())
        dims = np.frombuffer(state.field('cut.g_cdims'), '<i4')
        nv, nc = int(np.prod(dims)), int(np.prod(dims+1))
        ncp = int(state.array('cut.d_cutPointCounter', '<u4')[0])
        edges = state.array('cut.d_gridEdges', '<i4').reshape(-1, 4)
        count = int(state.array('recon.d_triCounter', '<u4')[0])
        original = state.array('recon.d_tri', '<i4')[:count].reshape(-1, 3).copy()
        features = state.array('cut.d_cutFP', '<f4').reshape(-1, 4).copy()
        external = state.array('recon.d_voxelExternalFP', '<f4').reshape(-1, 4)
        buffers = [state.array('cut.d_cutPoint', np.uint8)[:ncp*32], edges,
                   state.array('cut.c_particleRot', '<f4')[:nc*9],
                   state.array('cut.c_cornerPos', '<f4')[:nc*3],
                   state.array('cut.c_cornerRest', '<f4')[:nc*3],
                   state.array('cut.c_voxelCorner', '<i4')[:nv*8],
                   state.array('cut.d_conn4096', '<i4')[:4096*9],
                   state.array('cut.d_surfaceBoundaryMask', '<u4')[:nv],
                   state.array('cut.c_voxelIsectCount', '<i4')[:nv],
                   state.array('cut.c_active', '<i4')[:nc],
                   state.array('cut.d_cornerInside', '<i4')[:nc]]
        buffers = [np.ascontiguousarray(b) for b in buffers]
        addresses = (ct.c_void_p*len(buffers))(*(b.ctypes.data for b in buffers))
        counts = np.array([*dims, nv, nc, len(edges), ncp], np.int32)
        dll = ct.CDLL(str(args.library))
        ip, fp, up = ct.POINTER(ct.c_int), ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint)
        function = dll.LCS_ProbeBoundarySurface
        function.argtypes = [ct.c_int, ip, fp, ip, ct.POINTER(ct.c_void_p), ip, fp, ip, up]
        function.restype = ct.c_int
        def run(triangles):
            triangles = np.ascontiguousarray(triangles, dtype=np.int32)
            out, output_features = np.empty_like(triangles), np.empty_like(features)
            bindings, stats = np.empty(nv*8, np.int32), np.zeros(4, np.uint32)
            start = time.perf_counter()
            error = function(triangles.size, triangles.ctypes.data_as(ip), features.ctypes.data_as(fp),
                counts.ctypes.data_as(ip), addresses, out.ctypes.data_as(ip), output_features.ctypes.data_as(fp),
                bindings.ctypes.data_as(ip), stats.ctypes.data_as(up))
            assert error == 0, f'Native boundary probe error {error}'
            return out, output_features, bindings, stats.tolist(), (time.perf_counter()-start)*1000
        def surface(triangles, points):
            ids = triangles.astype(np.uint32)&np.uint32(0xbfffffff)
            unique, inverse = np.unique(ids, return_inverse=True)
            inverse = inverse.reshape(-1)
            cut = (unique&0x80000000) != 0
            index = unique&0x3fffffff
            verts = np.empty((len(unique), 3), np.float32)
            verts[cut] = points[index[cut], :3]
            verts[~cut] = external[index[~cut], :3]
            expanded = verts[inverse]
            anchors = np.zeros((len(expanded), 4), np.float32)
            anchors[:, 3] = (triangles.ravel().astype(np.uint32)>>30)&1
            s = Surface(expanded, np.zeros_like(expanded), np.arange(len(expanded), dtype=np.int32).reshape(-1, 3), anchors)
            return audit(verts, inverse.reshape(-1, 3), unique), topology_statistics(s), expanded
        out, points, bindings, stats, ms = run(original)
        before, before_quality, before_pos = surface(original, features)
        after, after_quality, after_pos = surface(out, points)
        record.update(before=before, after=after, before_quality=before_quality, after_quality=after_quality,
                      stats=stats, probe_with_upload_ms=ms)
        atomic_json(args.output/'progress.json', record)
        assert stats[1:3] == [1, 1]
        assert after['watertight'] and after['winding_consistent'] and not after['vertex_link_failures']
        for key in ('positional_nonmanifold_edges', 'positional_boundary_edges', 'duplicate_triangles', 'degenerate_triangles'):
            assert after_quality[key] == 0, key
        changed = np.flatnonzero((points != features).any(axis=1))
        assert len(changed) == 4
        assert np.all((bindings[changed]&7) == 0), 'Changed material support'
        for v in np.unique(changed//8):
            assert len(np.unique(bindings[8*v:8*v+2])) == 2 and bindings[8*v:8*v+2].min() > 7
        all_indices = np.arange(len(bindings))
        unchanged = np.ones(len(bindings), bool)
        unchanged[changed] = False
        assert np.array_equal(bindings[unchanged], all_indices[unchanged] % 8)
        original_n = np.cross(before_pos.reshape(-1, 3, 3)[:, 1]-before_pos.reshape(-1, 3, 3)[:, 0],
                              before_pos.reshape(-1, 3, 3)[:, 2]-before_pos.reshape(-1, 3, 3)[:, 0])
        after_n = np.cross(after_pos.reshape(-1, 3, 3)[:, 1]-after_pos.reshape(-1, 3, 3)[:, 0],
                           after_pos.reshape(-1, 3, 3)[:, 2]-after_pos.reshape(-1, 3, 3)[:, 0])
        assert np.all((original_n*after_n).sum(axis=1) > 0), 'A surface face flipped'
        with np.load(args.reference/'refined-surface.npz') as independent:
            expected = independent['refined_positions']
        # Same triangle order is retained; reference aliases may have opposite
        # numeric slot labels, so compare the actual coordinates instead.
        error_mm = float(np.max(np.linalg.norm(expected-after_pos, axis=1))*10)
        assert error_mm < 1e-4, 'Independent cut-point construction disagrees'
        rng = np.random.default_rng(22035)
        for _ in range(8):
            rotated = np.take_along_axis(original, (rng.integers(0, 3, (len(original), 1))+np.arange(3)) % 3, axis=1)
            shuffled, p2, b2, s2, _ = run(rotated[rng.permutation(len(original))])
            assert s2[2] == 1 and np.array_equal(canonical(out), canonical(shuffled))
            assert np.array_equal(points, p2) and np.array_equal(bindings, b2)
        clean = original[(original[:, 0].astype(np.uint32)&0x40000000) == 0]
        c_out, c_points, c_bindings, c_stats, _ = run(clean)
        assert np.array_equal(clean, c_out) and np.array_equal(features, c_points) and c_stats[2] == 0
        np.savez_compressed(args.output/'refined-surface.npz', vertices=after_pos,
                            feature_ids=out, bindings=bindings, cut_features=points)
        record.update(passed=True, permutation_checks=8, clean_surface_unchanged=True,
                      unchanged_material_support=True, distinct_contact_patch_keys=True,
                      independent_construction_max_error_mm=error_mm, changed_feature_slots=changed.tolist(),
                      changed_bindings=bindings[changed].tolist())
        return 0
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        atomic_json(args.output/'result.json', record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        raise SystemExit(main())
