"""Independently audit material/skin connectivity and the exported surface at saved states."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import trimesh
from threadpoolctl import threadpool_limits
from endoscopic_tissue.scenario import SCISSORS_SCENES


@threadpool_limits.wrap(limits=1,user_api='blas')
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_dir',type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.case_dir/'manifest.json').read_text(encoding='utf8'))
    scene=SCISSORS_SCENES[manifest['scene']]
    grid=scene.build_grid(manifest['settings']['density_kg_m3'])
    rows=[]
    for name in ('incise_1','incise_2','incise_repeat_1','incise_repeat_2','hold','side_hold','jaw_open','open_retract'):
        path=args.case_dir/(name+'.npz')
        if not path.exists():continue
        d=np.load(path)
        inside=d['inside'].astype(bool)
        a,slot=np.nonzero((grid.neighbors[:,1::2]>=0)&(d['neighbors'][:,1::2]<0))
        b=grid.neighbors[a,2*slot+1]
        mid=(grid.positions[a]+grid.positions[b])/2
        material=inside[a]|inside[b]
        skin=~material
        stray=skin.copy()
        if material.any() and skin.any():
            distances,_=cKDTree(mid[material]).query(mid[skin],distance_upper_bound=.15)
            stray[skin]=~np.isfinite(distances)
        lattice=d['lattice_labels']
        actual_pieces=np.unique(lattice[inside])
        both_parts=np.isin(lattice[a],actual_pieces)&np.isin(lattice[b],actual_pieces)
        row=dict(stage=name,severed_links=len(a),material_links=int(material.sum()),stray_skin_links=int(stray.sum()),
            stray_links_between_distinct_material_components=int(np.count_nonzero(stray&both_parts&(lattice[a]!=lattice[b]))),
            stray_links_within_same_component=int(np.count_nonzero(stray&(lattice[a]==lattice[b]))),
            stray_links_involving_skin_only_component=int(np.count_nonzero(stray&~both_parts)))
        labels,sizes=np.unique(lattice[inside],return_counts=True)
        order=np.argsort(sizes)[::-1]
        if len(order)>1:
            body=inside&(lattice==labels[order[0]])
            piece=inside&(lattice==labels[order[1]])
            distance,_=cKDTree(d['nodes_cm'][body]).query(d['nodes_cm'][piece])
            row['nearest_material_node_gap_mm']=float(distance.min()*10)
            row['body_com_mm']=(d['nodes_cm'][body].mean(axis=0)*10).tolist()
            row['piece_com_mm']=(d['nodes_cm'][piece].mean(axis=0)*10).tolist()
        mesh=trimesh.Trimesh(d['vertices_cm'],d['faces'],process=False)
        mesh.merge_vertices(digits_vertex=5)
        edges=mesh.edges_unique
        graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(mesh.vertices),)*2)
        labels=connected_components(graph,directed=False)[1]
        _,tri_counts=np.unique(labels[mesh.faces[:,0]],return_counts=True)
        triangles=d['vertices_cm'][d['faces']]
        longest=np.linalg.norm(triangles-np.roll(triangles,1,axis=1),axis=2).max(axis=1)*10
        twice_area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)*100
        aspect=longest**2/np.maximum(twice_area,1e-12)
        row.update(positional_surface_component_triangle_counts=sorted(tri_counts.tolist(),reverse=True),
            max_triangle_edge_mm=float(longest.max()),triangles_longer_than_5mm=int(np.count_nonzero(longest>5)),
            thin_long_triangles=int(np.count_nonzero((longest>3)&(aspect>20))),
            note='Length/aspect flags are visual-review cues, not calibrated realism thresholds; positional welding is diagnostic only.')
        rows.append(row)
    (args.case_dir/'cut-connectivity-audit.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    for row in rows: print(json.dumps(row),flush=True)


if __name__=='__main__':main()
