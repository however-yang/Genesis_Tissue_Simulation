"""Offline reconstruction hypothesis, with a native-ID control before any use.

The experiment retains ALL recorded skin/cut witnesses, including air-endpoint
owners. It changes no material link, live state, cut mask, or production source.
The trial graph excludes paths through nonmaterial corners. A closed result is
only a topology result, never permission to deploy an inferred crack front.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue.grid import CORNER_OFFSETS, EDGE_CORNERS, connectivity_table

STENCIL = np.array([[[0, 0, 0], [0, -1, 0], [0, -1, -1], [0, 0, -1]],
                    [[0, 0, 0], [0, 0, -1], [-1, 0, -1], [-1, 0, 0]],
                    [[0, 0, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]]])
CP = np.dtype([('offset', '<f4', 3), ('owner', '<i4'), ('edge', '<i4'), ('normal', '<f4', 3)])


def canonical(triangles):
    data = np.sort(np.asarray(triangles, dtype=np.uint32), axis=1)
    return data[np.lexsort(data.T[::-1])]


def topology(triangles):
    ids = np.asarray(triangles, np.uint32) & np.uint32(0xbfffffff)
    edges = np.sort(ids[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return dict(triangles=len(ids), boundary_edges=int(np.count_nonzero(counts == 1)),
                nonmanifold_edges=int(np.count_nonzero(counts > 2)),
                degenerate_ids=int(np.count_nonzero(np.diff(np.sort(ids, axis=1), axis=1) == 0)))


def native_internal_face_cleanup(triangles):
    """Mirror surface_cleanup.cuh: cancel opposite native feature-ID faces.

    This is an existing production step, required in both the control and trial.
    No positions or newly invented repair-specific face removals are used.
    """
    groups = {}
    for triangle in triangles:
        ids = [int(x) & 0xbfffffff for x in triangle]
        odd = sum(ids[a] > ids[b] for a, b in ((0, 1), (0, 2), (1, 2))) % 2
        key = tuple(sorted(ids))
        entry = groups.setdefault(key, [0, None, None])
        entry[0] += -1 if odd else 1
        if entry[1+odd] is None:
            entry[1+odd] = triangle
    return np.asarray([entry[1 if entry[0] > 0 else 2] for entry in groups.values() if entry[0]], np.uint32)


class Frozen:
    def __init__(self, directory, step):
        # Load each compressed array once. Repeated NpzFile indexing would
        # decompress multi-megabyte arrays for every emitted triangle.
        def read(path):
            with np.load(path) as archive:
                return {key: archive[key] for key in archive.files}
        self.grid = g = read(directory/'reconstruction-grid.npz')
        self.state = d = read(directory/f'native-topology-{step:04d}.npz')
        self.witness = w = read(directory/f'cut-witnesses-{step:04d}.npz')
        self.dims = g['dims'].astype(int)
        self.strides = np.array([1, self.dims[0]+1, (self.dims[0]+1)*(self.dims[1]+1)])
        self.vstrides = np.array([1, self.dims[0], self.dims[0]*self.dims[1]])
        self.masks = d['voxel_masks'] & 4095
        self.table = connectivity_table()
        self.cut_voxels = np.flatnonzero((self.masks != 0) & (g['occupied'] != 0))
        self.cp = w['cut_points'].copy().reshape(-1).view(CP)
        self.cp_by_voxel = defaultdict(list)
        for i, cp in enumerate(self.cp):
            if g['active'][cp['owner']] == 0:
                continue
            axis, *base = w['grid_edges'][cp['edge']]
            for coord in np.array(base)+STENCIL[axis]:
                v = int(coord@self.vstrides)
                if g['occupied'][v]:
                    self.cp_by_voxel[v].append(i)
        self.native_pos = {}
        native = d['feature_ids'].reshape(-1) & np.uint32(0xbfffffff)
        for raw, pos in zip(native, d['vertices_cm'][d['faces'].reshape(-1)]):
            self.native_pos[int(raw)] = pos.astype(float)
        if 'cut_fp' in w:
            for v in self.cut_voxels:
                for comp in range(8):
                    fp = w['cut_fp'][v, comp]
                    if fp[3] > .5:
                        self.native_pos[0x80000000 | (8*int(v)+comp)] = fp[:3].astype(float)
        # Reconstruct isosurface witnesses using the actual owner and rotation.
        isects = g['intersections']
        a, b, t = isects['a'], isects['b'], isects['t'].astype(float)
        slots = self.slots(a, b)
        cut = w['cut_t'].reshape(-1)[slots]
        owner_a = np.where((cut >= 0) & (cut <= 1) & (np.abs(t-cut) > 1e-6), t < cut, isects['inside_a'] != 0)
        self.iso_owner = np.where(owner_a, a, b)
        nodes, rot = d['nodes_cm'].astype(float), w['rotations'].astype(float)
        self.iso_world = nodes[a]+(nodes[b]-nodes[a])*t[:, None]
        severed = d['neighbors'].reshape(-1)[slots] < 0
        rest_delta = g['positions'][b]-g['positions'][a]
        rotated = np.einsum('nij,nj->ni', rot[self.iso_owner], rest_delta)
        framed = np.where(owner_a[:, None], nodes[a]+rotated*t[:, None], nodes[b]+rotated*(t-1)[:, None])
        self.iso_world[severed] = framed[severed]
        self.cp_world = nodes[self.cp['owner']]+np.einsum('nij,nj->ni', rot[self.cp['owner']], self.cp['offset'])
        self.cp_normals = np.einsum('nij,nj->ni', rot[self.cp['owner']], self.cp['normal'])

    def slots(self, a, b):
        diff = b-a
        axis = np.where(np.abs(diff) == 1, 0, np.where(np.abs(diff) == self.strides[1], 1, 2))
        return 6*a+2*axis+(diff > 0)

    def coord(self, node):
        return np.array([node % (self.dims[0]+1), node//(self.dims[0]+1) % (self.dims[1]+1), node//self.strides[2]])

    def local(self, corner, vcoord):
        return int(np.flatnonzero(np.all(CORNER_OFFSETS == self.coord(corner)-vcoord, axis=1))[0])

    def reconstruct(self, mode):
        g, d, w = self.grid, self.state, self.witness
        labels, positions = {}, dict(self.native_pos)
        valid = set()
        max_control_error = 0.
        witness_uses = 0
        changed_voxels = []
        for v in self.cut_voxels:
            mask = int(self.masks[v])
            corners = g['corners'][v]
            effective = mask
            if mode == 'material_intervals':
                inside = g['inside'][corners] != 0
                for e, (a, b) in enumerate(EDGE_CORNERS):
                    if not (inside[a] and inside[b]):
                        effective |= 1 << e
            conn = self.table[effective, 1:]
            labels[v] = conn
            if not np.array_equal(conn, self.table[mask, 1:]):
                changed_voxels.append(int(v))
            off, cnt = int(g['offsets'][v]), int(g['counts'][v])
            iso_ids = np.arange(off, off+cnt)
            vcoord = np.array([v % self.dims[0], v//self.dims[0] % self.dims[1], v//self.vstrides[2]])
            iso_local = np.array([self.local(int(self.iso_owner[i]), vcoord) for i in iso_ids], int)
            cp_ids = np.array(self.cp_by_voxel[v], int)
            cp_local = np.array([self.local(int(self.cp['owner'][i]), vcoord) for i in cp_ids], int)
            for comp in range(int(self.table[effective, 0])):
                key = 0x80000000 | (8*int(v)+comp)
                si = iso_ids[conn[iso_local] == comp]
                ci = cp_ids[conn[cp_local] == comp]
                if not len(si)+len(ci):
                    continue
                witness_uses += len(si)+len(ci)
                valid.add(key)
                selected = corners[conn == comp]
                bc = d['nodes_cm'][selected].astype(float).mean(axis=0)
                half = np.abs(g['positions'][corners[6]]-g['positions'][corners[0]])*.5
                x = (self.iso_world[si].sum(axis=0)+self.cp_world[ci].sum(axis=0))/(len(si)+len(ci))
                normals = g['intersections']['normal'][si].astype(float)
                for k in range(8):
                    force = (normals*np.einsum('ij,ij->i', normals, self.iso_world[si]-x)[:, None]).sum(axis=0)
                    proposal = x+.1*(1-k/8)*force
                    if np.any(np.abs(proposal-bc) >= half):
                        break
                    x = proposal
                x = np.clip(x, bc-half, bc+half)
                if len(ci):
                    nsum = np.rint(self.cp_normals[ci]*16384).sum(axis=0)/16384
                    norm = np.linalg.norm(nsum)
                    if norm > 1e-6 and norm > .25*len(ci):
                        normal = nsum/norm
                        x -= normal@ (x-self.cp_world[ci].mean(axis=0))*normal
                if mode == 'native':
                    if key in self.native_pos:
                        max_control_error = max(max_control_error, float(np.linalg.norm(x-self.native_pos[key])*10))
                        # Topology control uses the actual native positions. No
                        # reconstruction-roundoff tolerance can hide an ID mismatch.
                        x = self.native_pos[key]
                positions[key] = x

        def vertex(vcoord, owner):
            v = int(vcoord@self.vstrides)
            if v not in labels:
                return v
            comp = int(labels[v][self.local(owner, vcoord)])
            key = 0x80000000 | (8*v+comp)
            return key if key in valid else v

        triangles = []

        def emit(ids, wall):
            # This is the unchanged native degeneracy rule, not a repair.
            try:
                p = np.array([positions[int(key)] for key in ids])
            except KeyError as error:
                raise RuntimeError(f'Missing native feature {error}') from error
            # Native stores positions and performs the degeneracy check as
            # float32. Preserve that precision for the frozen control.
            p = p.astype(np.float32)
            normal = np.cross(p[1]-p[0], p[2]-p[0])
            if normal@normal < 1e-16:
                return
            triangles.append([int(key) | (0x40000000 if wall else 0) for key in ids])

        for edge in g['surface_edges']:
            axis, x, y, z, inside_a = map(int, edge[:5])
            base = np.array([x, y, z])
            a = int(base@self.strides)
            b = a+int(self.strides[axis])
            cut, skin = w['cut_t'][a, 2*axis+1], w['skin_t'][a, 2*axis+1]
            owner_a = skin < cut if 0 <= cut <= 1 and abs(float(skin-cut)) > 1e-6 else inside_a
            owner = a if owner_a else b
            ids = [vertex(coord, owner) for coord in base+STENCIL[axis]]
            if not inside_a:
                ids[1], ids[3] = ids[3], ids[1]
            emit([ids[0], ids[1], ids[2]], False)
            emit([ids[0], ids[2], ids[3]], False)

        ge = w['grid_edges']
        node_a = ge[:, 1:]@self.strides
        fractions = w['cut_t'][node_a, 2*ge[:, 0]+1]
        for e in np.flatnonzero((fractions >= 0) & (fractions <= 1)):
            axis, *base_list = ge[e]
            base = np.array(base_list)
            a = int(base@self.strides)
            b = a+int(self.strides[axis])
            inside_a, inside_b = g['inside'][a], g['inside'][b]
            skin = w['skin_t'][a, 2*axis+1]
            if not (inside_a and inside_b):
                if not (inside_a or inside_b) or not 0 <= skin <= 1:
                    continue
                if (fractions[e] > skin if inside_a else fractions[e] < skin):
                    continue
            for owner, other, flip in [(a, b, False), (b, a, True)]:
                ids, shared = [], []
                for coord in base+STENCIL[axis]:
                    v = int(coord@self.vstrides)
                    if not g['occupied'][v] or v not in labels:
                        continue
                    key = vertex(coord, owner)
                    if key not in valid:
                        continue
                    ids.append(key)
                    shared.append(key == vertex(coord, other))
                if len(ids) < 3:
                    continue
                if len(ids) == 4 and shared[0] and shared[2] and not shared[1] and not shared[3]:
                    ids, shared = ids[1:]+ids[:1], shared[1:]+shared[:1]
                for k in range(1, len(ids)-1):
                    if shared[0] and shared[k] and shared[k+1]:
                        continue
                    indices = [0, k+1, k] if flip else [0, k, k+1]
                    emit([ids[i] for i in indices], True)
        triangles = native_internal_face_cleanup(triangles)
        info = topology(triangles)
        info.update(mode=mode, changed_cut_voxels=len(changed_voxels),
                    skin_and_cut_witness_uses=witness_uses,
                    cpu_feature_max_control_error_mm=max_control_error)
        return triangles, positions, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--step', type=int, default=600)
    args = parser.parse_args()
    frozen = Frozen(args.directory, args.step)
    native, _, control = frozen.reconstruct('native')
    actual = frozen.state['feature_ids']
    control['exact_native_feature_incidence'] = np.array_equal(canonical(native), canonical(actual))
    if not control['exact_native_feature_incidence']:
        expected = Counter(map(tuple, canonical(actual)))
        computed = Counter(map(tuple, canonical(native)))
        extra = [[int(x) for x in key] for key in (computed-expected).elements()]
        missing = [[int(x) for x in key] for key in (expected-computed).elements()]
        result = dict(step=args.step, control=control, native=topology(actual), experiment_run=False,
                      extra_faces=extra, missing_faces=missing)
    else:
        trial, positions, hypothesis = frozen.reconstruct('material_intervals')
        assert hypothesis['skin_and_cut_witness_uses'] == control['skin_and_cut_witness_uses']
        keys = np.unique(trial & np.uint32(0xbfffffff))
        mesh_ids = np.searchsorted(keys, trial & np.uint32(0xbfffffff))
        vertices = np.array([positions[int(key)] for key in keys])
        np.savez_compressed(args.directory/f'offline-material-intervals-{args.step:04d}.npz',
            vertices_cm=vertices, triangles=mesh_ids, feature_ids=trial, feature_keys=keys)
        result = dict(step=args.step, control=control, hypothesis=hypothesis, experiment_run=True,
            topology_only=True, physical_links_changed=False, actual_cut_witnesses_preserved=True,
            deployable=False, note='No inferred finite crack front may be promoted on topology counts alone.')
    (args.directory/f'offline-topology-{args.step:04d}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
