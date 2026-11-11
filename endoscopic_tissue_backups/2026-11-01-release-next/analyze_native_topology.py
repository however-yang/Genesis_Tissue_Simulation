"""Locate native four-face edges without positional welding or geometry modification."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue.grid import CORNER_OFFSETS, EDGE_CORNERS, connectivity_table


def analyze(path, grid_path):
    data = np.load(path)
    grid = np.load(grid_path)
    dims = data['counts'][:3]
    native = data['feature_ids']
    ids = native & np.uint32(0xbfffffff)
    edges = np.sort(ids[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
    unique, inverse, count = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    conn = connectivity_table()
    vertices, faces = data['vertices_cm'], data['faces']
    result = []

    def point(raw):
        cut = bool(int(raw) & 0x80000000)
        index = int(raw) & 0x3fffffff
        voxel, component = (index//8, index % 8) if cut else (index, None)
        coord = np.array([voxel % dims[0], voxel//dims[0] % dims[1], voxel//(dims[0]*dims[1])])
        corners = coord+CORNER_OFFSETS
        node_ids = corners[:, 0]+(dims[0]+1)*(corners[:, 1]+(dims[1]+1)*corners[:, 2])
        mask = int(data['voxel_masks'][voxel]) & 4095
        components = conn[mask, 1:]
        selected = np.ones(8, bool) if component is None else components == component
        return dict(raw_feature_id=int(raw), voxel=int(voxel), coord=coord.tolist(),
                    component=component, mask=mask, local_cut_edges=EDGE_CORNERS[[(mask >> i) & 1 == 1 for i in range(12)]].tolist(),
                    local_corner_components=components.tolist(),
                    selected_local_corners=np.flatnonzero(selected).tolist(),
                    corner_ids=node_ids.tolist(), corner_inside=grid['inside'][node_ids].tolist(),
                    corner_rest_cm=grid['rest_cm'][node_ids].tolist(),
                    corner_world_cm=data['nodes_cm'][node_ids].tolist())

    for edge_index in np.flatnonzero(count > 2):
        face_ids = np.unique(np.flatnonzero(inverse == edge_index)//3)
        raw_edge = unique[edge_index]
        details = [point(r) for r in raw_edge]
        delta = np.array(details[1]['coord'])-details[0]['coord']
        result.append(dict(edge_features=raw_edge.tolist(), points=details,
            voxel_delta=delta.tolist(), face_ids=face_ids.tolist(),
            wall_faces=((native[face_ids, 0] & 0x40000000) != 0).tolist(),
            face_features=ids[face_ids].tolist(), faces_mm=(vertices[faces[face_ids]]*10).tolist(),
            face_anchors=data['anchors'][faces[face_ids]].tolist()))
    return dict(source=str(path), positional_welding_used=False,
                native_nonmanifold_edges=len(result), native_boundary_edges=int(np.count_nonzero(count == 1)),
                edges=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--grid-state', type=Path, required=True)
    args = parser.parse_args()
    results = [analyze(p, args.grid_state) for p in sorted(args.directory.glob('native-topology-*.npz'))]
    (args.directory/'native-topology-audit.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps([dict(source=r['source'], native_nonmanifold_edges=r['native_nonmanifold_edges'],
                          native_boundary_edges=r['native_boundary_edges'],
                          edges=[dict(features=e['edge_features'], voxel_delta=e['voxel_delta'],
                                      walls=e['wall_faces'], points=[{k: p[k] for k in
                                      ('coord', 'component', 'mask', 'selected_local_corners', 'corner_inside')} for p in e['points']])
                                 for e in r['edges']]) for r in results], indent=2))
