"""Read a retained snapshot offline; never load a DLL or modify simulation state."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import struct
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue.training.snapshot import EnvSnapshot
from endoscopic_tissue.instrument_geometry import load_scissors, scissors_body_depths


def name_hash(text):
    data = text.encode()
    value = 1469598103934665603 ^ len(data)
    end = len(data)//8*8
    for offset in range(0, end, 8):
        value = ((value ^ int.from_bytes(data[offset:offset+8], 'little'))*1099511628211) & ((1 << 64)-1)
    return ((value ^ int.from_bytes(data[end:], 'little'))*1099511628211) & ((1 << 64)-1)


class Archive:
    def __init__(self, raw):
        self.raw = raw
        magic, size, checksum, next_id, abi, allocations, fields, reserved = struct.unpack_from('<QQQQIIII', raw)
        if magic != 0x314554415453434c or size != len(raw) or abi != 1 or reserved:
            raise ValueError('Unexpected native archive header')
        self.allocations, self.fields = {}, {}
        offset = 48
        for _ in range(allocations):
            ident, key, size, stride, pointers, pad = struct.unpack_from('<QQQQII', raw, offset)
            offset += 40 + pointers*4
            if ident in self.allocations or pad or offset+size > len(raw):
                raise ValueError('Invalid allocation extent')
            self.allocations[ident] = memoryview(raw)[offset:offset+size]
            offset += size
        for _ in range(fields):
            key, size, stride, pointers, item_bytes = struct.unpack_from('<QQQII', raw, offset)
            offset += 32 + pointers*4
            if key in self.fields or offset+size > len(raw):
                raise ValueError('Invalid field extent')
            self.fields[key] = memoryview(raw)[offset:offset+size]
            offset += size
        if offset != len(raw):
            raise ValueError('Native archive has trailing bytes')

    def field(self, name):
        return self.fields[name_hash(name)]

    def scalar(self, name, fmt='<i'):
        return struct.unpack(fmt, self.field(name))[0]

    def array(self, name, dtype):
        token = self.scalar(name, '<Q')
        data = self.allocations[token >> 32][token & 0xffffffff:]
        return np.frombuffer(data, dtype=dtype)


V3 = ('<f4', (3,))
SHAPE = np.dtype([(name, *V3) for name in ('center', 'u', 'n', 'w', 'half', 'a', 'b')]
                 + [('radius', '<f4'), ('capsule', '<i4')])
PLANE = np.dtype([(name, *V3) for name in ('normal', 'target', 'motion')] + [('valid', '<i4')])
VERTEX = np.dtype([('id', '<i4', (8,)), ('count', '<i4'), ('voxel', '<i4'), ('component', '<i4'),
                   ('mask', '<u4'), ('weight', '<f4', (8,)), ('offset', *V3)])
CONTACT = np.dtype([('id', '<i4', (24,)), ('count', '<i4'), ('kind', '<i4'), ('weight', '<f4', (24,)),
    ('offset', *V3), ('previous', *V3), ('plane', PLANE), ('lambdaN', '<f4'), ('lambdaT', *V3),
    ('owner', '<i4', (3,)), ('component', '<i4', (3,)), ('mask', '<u4', (3,)),
    ('toolOrigin', *V3), ('gapGradient', *V3), ('toolGap', '<f4'), ('shape', SHAPE), ('incoming', '<i4'),
    ('stick', *V3), ('stickSpring', *V3), ('stickStiffness', '<f4')])


def point(record, positions):
    count = int(record['count'])
    if not 0 <= count <= len(record['id']):
        raise ValueError('Contact struct layout/count mismatch')
    return record['offset'] + (positions[record['id'][:count]]*record['weight'][:count, None]).sum(axis=0)


def contact_summary(record, positions, h):
    if not record['count']:
        return {'count': 0, 'kind': int(record['kind']), 'valid_contact': False}
    x = point(record, positions)
    normal = record['plane']['normal']
    return {'count': int(record['count']), 'kind': int(record['kind']),
        'normal': normal.tolist(), 'plane_target_cm': record['plane']['target'].tolist(),
        'normal_gap_mm': float(np.dot(x-record['plane']['target'], normal)*10),
        'point_cm': x.tolist(), 'lambdaN': float(record['lambdaN']),
        'cached_normal_force_n_may_precede_final_repair': float(record['lambdaN']/(h*h)*.01),
        'ids': record['id'][:record['count']].tolist(),
        'weights': record['weight'][:record['count']].tolist(),
        'owner': record['owner'].tolist(), 'component': record['component'].tolist(),
        'mask': record['mask'].tolist(), 'reference_center_cm': record['shape']['center'].tolist(),
        'reference_half_cm': record['shape']['half'].tolist()}


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    p = argparse.ArgumentParser()
    p.add_argument('snapshot', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    with args.snapshot.open('rb') as stream:
        magic, m, n, py = struct.unpack('<8sQQQ', stream.read(32))
        metadata = json.loads(stream.read(m))
    # The standard reader verifies compressed SHA and size. Reading offline uses
    # the file's binding; it grants no permission to restore into a different DLL.
    snap = EnvSnapshot.load(args.snapshot, dll_sha256=metadata['dll_sha256'],
                            physics_config_sha256=metadata['physics_config_sha256'])
    state = snap.python_state()
    archive = Archive(snap.native_bytes())
    observed, surface = state['observed'], state['last_surface']
    positions = archive.array('xpbd.d_pos', '<f4').reshape(-1, 3)
    if not np.array_equal(positions, observed.nodes):
        raise ValueError('Native/Python positions disagree')
    triangles = surface.vertices[surface.triangles]
    native_triangles = archive.scalar('xpbd.g_surfaceCount')
    expanded = archive.array('xpbd.g_surfaceExpanded', '<f4')[:9*native_triangles].reshape(-1, 3, 3)
    if not np.array_equal(expanded, triangles):
        raise ValueError('Native/Python expanded surface order differs')
    vertices = archive.array('xpbd.d_surfaceVertices', VERTEX)[:3*native_triangles]
    # Finalize/reconstruction atomically emits triangles in a new order. The
    # contact arrays belong to the preceding linearization, not the final mesh's
    # triangle numbers. Match the material key, just as the native cache does.
    tags = archive.array('xpbd.g_surfaceTags', '<u4')[:3*native_triangles].reshape(-1, 3)
    masks = archive.array('xpbd.g_surfaceMasks', '<u4')
    material_keys = {}
    for face, group in enumerate(vertices.reshape(-1, 3)):
        key = tuple(sorted((int(v['voxel']), int(v['component']), int(v['mask'])) for v in group))
        material_keys.setdefault(key, []).append(face)
    contacts = archive.array('xpbd.d_bladeContacts', CONTACT)
    manifold = archive.array('xpbd.d_bladeManifold', CONTACT)
    count = int(archive.array('xpbd.d_bladeManifoldCount', '<i4')[0])
    owners = archive.array('xpbd.d_bladeOwners', '<i4')[:count]
    quadrature = archive.array('xpbd.d_bladeQuadrature', '<f4')[:count]
    h = archive.scalar('xpbd.g_hFrame', '<f')
    inv_mass = archive.array('xpbd.d_invMass', '<f4')
    tet_ids = archive.array('xpbd.d_tetIds', '<i4').reshape(-1, 4)
    tet_live = archive.array('xpbd.d_tetActive', '<i4') != 0
    rest_metric = archive.array('xpbd.d_unifiedRestMetric', '<f4').reshape(-1, 7)

    def evaluated_contact(i):
        c = manifold[i]
        result = contact_summary(c, positions, h)
        n, x = c['plane']['normal'], point(c, positions)
        gap = float(np.dot(x-c['plane']['target'], n))
        old_depth = max(0., -float(np.dot(c['previous']-c['plane']['target']+c['plane']['motion'], n)))
        inverse = float((c['weight'][:c['count']]**2*inv_mass[c['id'][:c['count']]]).sum())
        stiffness = 200000.*float(quadrature[i])
        damping = 2*np.sqrt(stiffness/inverse)/h if inverse > 0 else 0.
        result.update(quadrature=float(quadrature[i]),
                      elastic_pressure_n=max(0., -stiffness*gap)*.01,
                      evaluated_pressure_n_at_reference_pose=max(0., -stiffness*gap+damping*(-gap-old_depth))*.01 if gap < 0 else 0.,
                      previous_depth_mm=old_depth*10)
        return result
    geometry, scissors = load_scissors(), observed.scissors
    samples = np.concatenate((triangles, triangles.mean(axis=1)[:, None],
                              (triangles+np.roll(triangles, -1, axis=1))*.5), axis=1)
    relative = samples-scissors.hinge
    width = np.cross(scissors.axis, scissors.normal)
    sn = np.clip(scissors.gap/(2*geometry.jaws.length), 0, .95)
    cs = np.sqrt(1-sn*sn)
    results = []
    for side, kind, box in zip((1, -1), (3, 4), geometry.backs):
        orientation = np.stack((box[6:9], box[9:12], np.cross(box[6:9], box[9:12])))
        frame = np.stack((scissors.axis*cs+side*scissors.normal*sn,
                          scissors.normal*cs-side*scissors.axis*sn, width))
        local = (relative @ frame.T-box[:3]) @ orientation.T
        depth = (box[3:6]-np.abs(local)).min(axis=2)
        for face in np.argsort(depth.max(axis=1))[-4:][::-1]:
            if depth[face].max() <= 0:
                continue
            current_key = []
            for tag in tags[face]:
                cut, index = bool(tag & 0x80000000), int(tag & 0x3fffffff)
                voxel, component = (index//8, index % 8) if cut else (index, -1)
                current_key.append((voxel, component, int(masks[voxel] & 4095)))
            prior_faces = material_keys.get(tuple(sorted(current_key)), [])
            linearizations = []
            for prior_face in prior_faces:
                slot = prior_face*6+kind
                matches = np.flatnonzero(owners == slot)
                stencil = vertices[3*prior_face:3*prior_face+3]
                bound = np.stack([point(v, positions) for v in stencil])
                order = [(int(v['voxel']), int(v['component']), int(v['mask'])) for v in stencil]
                reordered = bound[[order.index(key) for key in current_key]]
                node_ids = np.unique(np.concatenate([v['id'][:v['count']] for v in stencil]))
                affected = np.flatnonzero(tet_live & np.isin(tet_ids, node_ids).any(axis=1))
                p4 = positions[tet_ids[affected]]
                volume = np.einsum('ij,ij->i', p4[:, 1]-p4[:, 0],
                                   np.cross(p4[:, 2]-p4[:, 0], p4[:, 3]-p4[:, 0]))*rest_metric[affected, 6]
                linearizations.append({'prior_face': prior_face,
                    'embedded_vertices_cm': reordered.tolist(), 'vertex_counts': stencil['count'].tolist(),
                    'embedding_error_mm': float(np.linalg.norm(reordered-triangles[face], axis=1).max()*10),
                    'adjacent_live_tets': len(affected),
                    'adjacent_min_J': float(volume.min()) if len(volume) else None,
                    'adjacent_tets_below_J_0151': int(np.count_nonzero(volume < .151)),
                    'base': contact_summary(contacts[slot], positions, h),
                    'manifold': [evaluated_contact(i) for i in matches]})
            results.append({'side': side, 'kind': kind, 'face': int(face),
                'depth_mm': float(max(0, depth[face].max())*10),
                'wall_flag': float(surface.anchors[surface.triangles[face, 0], 3]),
                'expanded_vertices_cm': triangles[face].tolist(),
                'material_key': current_key, 'linearizations': linearizations})
    depths = scissors_body_depths(surface.vertices, surface.triangles, geometry, scissors.hinge,
                                  scissors.axis, scissors.normal, scissors.gap)
    result = {'diagnostic_only': True, 'snapshot': str(args.snapshot), 'dll_sha256': metadata['dll_sha256'],
        'native_python_arrays_exact': True, 'native_contact_record_bytes': CONTACT.itemsize,
        'native_tick': state['native']['tick'], 'quality': state['state'].metadata['quality'],
        'body_depth_mm': float(depths[0]*10), 'wrist_depth_mm': float(depths[1]*10),
        'scissors_loads': asdict(scissors.loads), 'manifold_count': count, 'faces': results}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('faces', 'quality')}), flush=True)


if __name__ == '__main__':
    main()
