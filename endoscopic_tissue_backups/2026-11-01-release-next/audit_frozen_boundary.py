"""Check frozen boundary topology, including vertex links and tiny components."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import trimesh


def audit(vertices, faces, feature_keys):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    incident = defaultdict(list)
    for f, tri in enumerate(faces):
        for v in tri:
            if int(feature_keys[v]) & 0x80000000:
                incident[int(v)].append(f)
    bad_links = []
    for vertex, face_ids in incident.items():
        # The link of a surface vertex must be one cycle, not two cycles
        # touching at a single point. Edges alone cannot detect this defect.
        neighbors = defaultdict(set)
        for f in face_ids:
            other = [int(v) for v in faces[f] if v != vertex]
            assert len(other) == 2
            neighbors[other[0]].add(other[1])
            neighbors[other[1]].add(other[0])
        seen, todo = set(), [next(iter(neighbors))]
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            todo.extend(neighbors[n]-seen)
        if len(seen) != len(neighbors) or any(len(a) != 2 for a in neighbors.values()):
            bad_links.append(int(feature_keys[vertex]))
    parts = mesh.split(only_watertight=False)
    return dict(watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
                volume_mm3=float(mesh.volume*1000), area_mm2=float(mesh.area*100),
                vertex_link_failures=bad_links,
                components=sorted([dict(faces=len(m.faces), closed=bool(m.is_watertight),
                                        volume_mm3=float(m.volume*1000)) for m in parts], key=lambda p: -p['faces']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    results = []
    for step in (600, 840):
        with np.load(args.directory/f'native-topology-{step:04d}.npz') as data:
            raw = data['feature_ids'] & np.uint32(0xbfffffff)
            keys, first, inverse = np.unique(raw, return_index=True, return_inverse=True)
            old_pos = data['vertices_cm'][data['faces'].reshape(-1)][first]
            old_faces = inverse.reshape(-1, 3)
        with np.load(args.directory/f'offline-material-intervals-{step:04d}.npz') as data:
            new_pos, new_faces, new_keys = data['vertices_cm'], data['triangles'], data['feature_keys']
        forward = cKDTree(old_pos).query(new_pos)[0]*10
        backward = cKDTree(new_pos).query(old_pos)[0]*10
        results.append(dict(step=step, original=audit(old_pos, old_faces, keys),
                            hypothesis=audit(new_pos, new_faces, new_keys),
                            sampled_vertex_distance_mm=dict(maximum=float(max(forward.max(), backward.max())),
                                p95=float(np.percentile(np.r_[forward, backward], 95))),
                            physical_acceptance=False, note='Frozen geometry screen only; original nonclosed volume is diagnostic.'))
    (args.directory/'frozen-boundary-audit.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
