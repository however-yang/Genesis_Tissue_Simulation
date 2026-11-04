"""Read-only localisation of positional nonmanifold edges in saved R20 surfaces.

Changing weld precision here is a sensitivity diagnostic, never an acceptance
threshold change or a repair of the native surface.
"""
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue.training.provenance import sha256

P = Path(__file__).resolve().parent
OUT = ROOT / 'endoscopic_tissue/outputs/training-completion-20261101/r20-kidney'


def inspect(path):
    with np.load(path, allow_pickle=False) as data:
        vertices, faces, anchors = (data[k] for k in ('vertices_cm', 'faces', 'anchors'))
    scans = []
    for digits in (4, 5, 6, 7):
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        mesh.merge_vertices(digits_vertex=digits)
        incidence = np.bincount(mesh.edges_unique_inverse)
        bad = np.flatnonzero(incidence > 2)
        edges = []
        for edge in bad:
            face_ids = np.unique(mesh.edges_face[mesh.edges_unique_inverse == edge])
            edges.append({
                'edge_mm': (mesh.vertices[mesh.edges_unique[edge]] * 10).tolist(),
                'incident_face_ids': face_ids.tolist(),
                'incident_count': int(incidence[edge]),
                'wall_faces': (anchors[faces[face_ids, 0], 3] > .5).tolist(),
                'face_area_mm2': (mesh.area_faces[face_ids] * 100).tolist(),
                'face_normals': mesh.face_normals[face_ids].tolist(),
                'raw_face_anchors': anchors[faces[face_ids]].tolist(),
                'raw_faces_mm': (vertices[faces[face_ids]] * 10).tolist(),
            })
        scans.append({'native_cm_decimal_digits': digits,
                      'boundary_edges': int(np.count_nonzero(incidence == 1)),
                      'nonmanifold_edges': len(bad), 'edges': edges})
    return {'source': str(path), 'sha256': sha256(path), 'scans': scans}


def main():
    cases = [inspect(OUT / (name + '.npz')) for name in
             ('blade_approach', 'incise_1', 'incise_2', 'side_hold', 'jaw_open', 'open_retract')]
    result = {'read_only': True, 'native_geometry_modified': False,
              'weld_precision_does_not_change_acceptance': True, 'cases': cases}
    (P / 'surface-edge-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{'source': Path(c['source']).name,
                      'counts_by_precision': [[s['native_cm_decimal_digits'], s['boundary_edges'],
                                               s['nonmanifold_edges']] for s in c['scans']]}
                     for c in cases], indent=2))


if __name__ == '__main__':
    main()
