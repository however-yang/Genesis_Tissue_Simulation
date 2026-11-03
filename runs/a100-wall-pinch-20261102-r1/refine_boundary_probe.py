"""Offline hypothesis: separate boundary fans using their recorded cut points.

No material edge is severed, no native state is restored, and no default file
is changed. This is restricted to the retained interior, wall-only ambiguity.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np
from threadpoolctl import threadpool_limits


def fans(faces, center, opposite):
    incident = np.flatnonzero((faces == center).any(axis=1))
    by_neighbor = {}
    for f in incident:
        for key in faces[f]:
            if key not in (center, opposite):
                by_neighbor.setdefault(int(key), []).append(int(f))
    links = {int(f): set() for f in incident}
    for group in by_neighbor.values():
        for f in group:
            links[f].update(set(group)-{f})
    remaining, components = set(links), []
    while remaining:
        todo, seen = [min(remaining)], set()
        while todo:
            f = todo.pop()
            if f in seen:
                continue
            seen.add(f)
            todo.extend(links[f]-seen)
        remaining -= seen
        components.append(sorted(seen))
    return components


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.release), str(args.release/'endoscopic_tissue_backups/2026-11-02-acceptance-fixes'),
                    str(args.release/'endoscopic_tissue_backups/2026-11-01-release-next')]
    from inspect_native_contact import Archive
    from audit_frozen_boundary import audit
    from endoscopic_tissue.training.snapshot import EnvSnapshot
    from endoscopic_tissue.training.data import atomic_json
    from endoscopic_tissue.training.provenance import sha256
    from endoscopic_tissue.native_backend import Surface
    from endoscopic_tissue.surface_diagnostics import topology_statistics
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(passed=False, default_replaced=False, native_state_restored=False,
                  snapshot_sha256=sha256(args.snapshot), scope='Offline geometry hypothesis, not physical acceptance')
    try:
        with args.snapshot.open('rb') as f:
            _, length, _, _ = struct.unpack('<8sQQQ', f.read(32))
            metadata = json.loads(f.read(length))
        snap = EnvSnapshot.load(args.snapshot, dll_sha256=metadata['dll_sha256'],
                                physics_config_sha256=metadata['physics_config_sha256'])
        state = Archive(snap.native_bytes())
        count = int(state.array('recon.d_triCounter', '<u4')[0])
        raw = state.array('recon.d_tri', '<u4')[:count].reshape(-1, 3).copy()
        keys = raw & np.uint32(0xbfffffff)
        dimensions = np.frombuffer(state.field('cut.g_cdims'), '<i4')
        nx, ny, nz = map(int, dimensions)
        def voxel(key):
            v = (int(key)&0x3fffffff)//8
            return np.array([v % nx, (v//nx) % ny, v//(nx*ny)])
        edges = np.sort(keys[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        edge_keys, counts = np.unique(edges, axis=0, return_counts=True)
        bad = edge_keys[counts > 2]
        assert len(bad) == 1 and counts[counts > 2].tolist() == [4]
        a, b = map(int, bad[0])
        cutpoints_dtype = np.dtype([('offset', '<f4', (3,)), ('owner', '<i4'), ('edge', '<i4'), ('normal', '<f4', (3,))])
        npoints = int(state.array('cut.d_cutPointCounter', '<u4')[0])
        points = state.array('cut.d_cutPoint', cutpoints_dtype)[:npoints]
        grid_edges = state.array('cut.d_gridEdges', '<i4').reshape(-1, 4)
        edge_ids = {tuple(map(int, edge)): i for i, edge in enumerate(grid_edges)}
        corners = state.array('cut.c_voxelCorner', '<i4').reshape(-1, 8)
        positions = state.array('cut.c_cornerPos', '<f4').reshape(-1, 3)
        rest = state.array('cut.c_cornerRest', '<f4').reshape(-1, 3)
        rotations = state.array('cut.c_particleRot', '<f4').reshape(-1, 3, 3)
        active = state.array('cut.c_active', '<i4')
        boundary = state.array('cut.d_surfaceBoundaryMask', '<u4')
        conn = state.array('cut.d_conn4096', '<i4').reshape(-1, 9)
        iso_count = state.array('cut.c_voxelIsectCount', '<i4')
        original_fp = state.array('cut.d_cutFP', '<f4').reshape(-1, 4)
        fp = original_fp.copy()
        external = state.array('recon.d_voxelExternalFP', '<f4').reshape(-1, 4)
        updated = raw.copy()
        changes = []
        for center, opposite in ((a, b), (b, a)):
            v = (center&0x3fffffff)//8
            old_component = (center&0x3fffffff) % 8
            component_count = int(conn[boundary[v], 0])
            assert component_count == 1 and old_component == 0 and iso_count[v] == 0
            groups = fans(keys, center, opposite)
            assert len(groups) == 2
            descriptions = []
            for group in groups:
                assert np.all((raw[group] & 0xc0000000) == 0xc0000000)
                sources = set()
                for face in group:
                    vc = np.stack([voxel(k) for k in keys[face]])
                    width = np.ptp(vc, axis=0)
                    assert sorted(width.tolist()) == [0, 1, 1]
                    axis = int(np.flatnonzero(width == 0)[0])
                    base = vc.max(axis=0)
                    sources.add(edge_ids[(axis, *map(int, base))])
                descriptions.append((tuple(sorted(sources)), group))
            descriptions.sort()
            owned = corners[v][conn[boundary[v], 1:] == old_component]
            box_center = positions[owned].mean(axis=0)
            half = np.abs(rest[corners[v, 6]]-rest[corners[v, 0]])*.5
            for alias, (sources, group) in enumerate(descriptions):
                selected = points[np.isin(points['edge'], sources) & np.isin(points['owner'], owned)
                                  & (active[points['owner']] != 0)]
                assert len(selected) >= 2
                world = positions[selected['owner']] + np.einsum('nij,nj->ni', rotations[selected['owner']], selected['offset'])
                normals = np.einsum('nij,nj->ni', rotations[selected['owner']], selected['normal'])
                # Original cut-only seed, component box, and coherent-plane
                # projection, using only this boundary fan's true cut records.
                centroid = world.mean(axis=0)
                point = np.clip(centroid, box_center-half, box_center+half)
                accumulated = np.rint(normals*16384).astype(np.int64).sum(axis=0)/16384
                magnitude = np.linalg.norm(accumulated)
                if magnitude > 1e-6 and magnitude > .25*len(selected):
                    normal = accumulated/magnitude
                    point = point - np.dot(normal, point-centroid)*normal
                tag = np.uint32(0x80000000 | (8*v+alias))
                if alias:
                    assert not np.any(keys == tag) and original_fp[8*v+alias, 3] == 0
                fp[8*v+alias] = [*point, 1]
                for face in group:
                    for k in np.flatnonzero(keys[face] == center):
                        updated[face, k] = tag | np.uint32(0x40000000)
                changes.append(dict(voxel=v, original_component=old_component, alias=alias,
                    corner_support=owned.tolist(), cut_edge_ids=list(sources), cutpoints=len(selected),
                    original_mm=(original_fp[8*v, :3]*10).tolist(), refined_mm=(point*10).tolist(),
                    shift_mm=float(np.linalg.norm(point-original_fp[8*v, :3])*10), faces=group))
        def inspect(triangles, features):
            ids = triangles&np.uint32(0xbfffffff)
            unique, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
            inverse = inverse.reshape(-1)
            cut = (unique & 0x80000000) != 0
            index = unique & 0x3fffffff
            vertices = np.empty((len(unique), 3), np.float32)
            vertices[cut] = features[index[cut], :3]
            vertices[~cut] = external[index[~cut], :3]
            expanded = vertices[inverse]
            anchors = np.zeros((len(expanded), 4), np.float32)
            anchors[:, 3] = ((triangles.ravel() >> 30) & 1)
            surface = Surface(expanded, np.zeros_like(expanded), np.arange(len(expanded), dtype=np.int32).reshape(-1, 3), anchors)
            return audit(vertices, inverse.reshape(-1, 3), unique), topology_statistics(surface), expanded
        before, before_quality, old_surface = inspect(raw, original_fp)
        after, after_quality, new_surface = inspect(updated, fp)
        def normals(expanded):
            tris = expanded.reshape(-1, 3, 3)
            return np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0])
        before_n, after_n = normals(old_surface), normals(new_surface)
        changed_faces = np.flatnonzero(np.any(old_surface.reshape(-1, 9) != new_surface.reshape(-1, 9), axis=1))
        cosines = (before_n*after_n).sum(axis=1)/(np.linalg.norm(before_n, axis=1)*np.linalg.norm(after_n, axis=1))
        result.update(before=before, after=after, before_quality=before_quality, after_quality=after_quality,
                      changes=changes, affected_faces=len(changed_faces),
                      normal_cosines=cosines[changed_faces].tolist(),
                      inverted_faces=changed_faces[cosines[changed_faces] <= 0].tolist())
        np.savez_compressed(args.output/'refined-surface.npz', original_ids=raw, refined_ids=updated,
                            original_positions=old_surface, refined_positions=new_surface, dimensions=dimensions,
                            cut_features=fp, material_bindings=np.array([[c['voxel'], c['alias'], c['original_component']] for c in changes]))
        result['passed'] = bool(after['watertight'] and after['winding_consistent'] and not after['vertex_link_failures']
            and not result['inverted_faces'] and not after_quality['positional_boundary_edges']
            and not after_quality['positional_nonmanifold_edges'] and not after_quality['degenerate_triangles']
            and not after_quality['duplicate_triangles'])
        return 0 if result['passed'] else 2
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        atomic_json(args.output/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        raise SystemExit(main())
