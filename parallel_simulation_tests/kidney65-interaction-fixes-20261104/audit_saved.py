"""Read-only topology/force diagnosis of the previous complete kidney runs."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PREVIOUS = ROOT.parent / "kidney65-scene-20261104"
SOURCE = PROJECT / "parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/python-gpu-check"
sys.path[:0] = [str(PREVIOUS), str(SOURCE)]
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from endoscopic_tissue import run
from endoscopic_tissue.grid import EDGE_CORNERS
from scene_config import configure


def labels(neighbors, members):
    ids = np.flatnonzero(members)
    a = np.repeat(ids, 6)
    b = neighbors[ids].reshape(-1)
    keep = (b >= 0) & members[np.maximum(b, 0)]
    graph = coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(len(members),)*2)
    return connected_components(graph, directed=False)[1]


def audit(grid, path):
    state = np.load(path)
    neighbors = state['neighbors']
    inside = grid.inside != 0
    active = grid.active != 0
    material_labels = labels(neighbors, inside)
    all_labels = labels(neighbors, active)
    a, axes = np.nonzero((grid.neighbors[:, 1::2] >= 0) & (neighbors[:, 1::2] < 0))
    b = grid.neighbors[a, 2*axes+1]
    mids = (grid.positions[a] + grid.positions[b])*.5
    material = inside[a] | inside[b]
    distances, _ = cKDTree(mids[material]).query(mids[~material])
    skin_indices = np.flatnonzero(~material)
    stray = skin_indices[distances > 1.5*grid.spacing]
    occupied_edges = np.zeros((len(inside),3), bool)
    corners = grid.corners[grid.occupied != 0]
    for x,y in EDGE_CORNERS:
        c,d = np.minimum(corners[:,x],corners[:,y]),np.maximum(corners[:,x],corners[:,y])
        k = np.argmax(np.abs(grid.positions[d]-grid.positions[c]),axis=1)
        occupied_edges[c,k] = True
    unique_material = np.unique(material_labels[inside])
    # Deterministic, synchronous multi-source propagation, to convergence.
    owners = np.full(len(inside), np.iinfo(np.uint64).max, np.uint64)
    owners[inside] = material_labels[inside].astype(np.uint64)
    for passes in range(1,65):
        before = owners.copy()
        for slot in range(6):
            valid = (neighbors[:,slot] >= 0) & active
            ids = np.flatnonzero(valid)
            other = neighbors[ids,slot]
            valid = before[other] != np.iinfo(np.uint64).max
            ids,other = ids[valid],other[valid]
            owners[ids] = np.minimum(owners[ids], before[other] + np.uint64(1<<32))
        if np.array_equal(owners,before): break
    owned = active & (owners != np.iinfo(np.uint64).max)
    hops = (owners[owned] >> np.uint64(32)).astype(int)
    seams=[]
    for axis in range(3):
        aa=np.flatnonzero(neighbors[:,2*axis+1]>=0);bb=neighbors[aa,2*axis+1]
        valid=owned[aa]&owned[bb]&((owners[aa]&np.uint64(0xffffffff))!=(owners[bb]&np.uint64(0xffffffff)))
        for x,y in zip(aa[valid],bb[valid]):
            seams.append(dict(a=int(x),b=int(y),axis=axis,rest_a_cm=grid.positions[x].tolist(),rest_b_cm=grid.positions[y].tolist(),a_hops=int(owners[x]>>np.uint64(32)),b_hops=int(owners[y]>>np.uint64(32)),occupied=bool(occupied_edges[x,axis])))
    bridge=[]
    if len(unique_material)>1 and len(np.unique(all_labels[inside]))==1:
        visited=np.full(len(inside),-1,int)
        seeds=np.flatnonzero(inside & (material_labels==unique_material[0]))
        visited[seeds]=seeds
        queue=deque(seeds.tolist());end=None
        while queue and end is None:
            x=queue.popleft()
            for y in neighbors[x]:
                if y<0 or visited[y]>=0:continue
                visited[y]=x
                if inside[y] and material_labels[y]!=unique_material[0]:end=int(y);break
                queue.append(int(y))
        if end is not None:
            while visited[end]!=end:
                bridge.append(end);end=int(visited[end])
            bridge.append(end);bridge.reverse()
    return dict(snapshot=str(path),material_components=len(unique_material),lattice_pieces=len(np.unique(all_labels[inside])),
                cut_edges=len(a),stray_count=len(stray),stray_in_occupied_cells=int(occupied_edges[a[stray],axes[stray]].sum()),
                stray_edges=[dict(a=int(a[i]),b=int(b[i]),axis=int(axes[i]),rest_mid_cm=mids[i].tolist(),occupied=bool(occupied_edges[a[i],axes[i]])) for i in stray],
                owners_iterations=passes,owner_distance_hist=np.bincount(hops).tolist(),unowned_active=int((active&~owned).sum()),
                remaining_owner_seams=seams,bridge=[dict(id=i,inside=bool(inside[i]),rest_cm=grid.positions[i].tolist()) for i in bridge])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',nargs='+',default=['frontal65-r3','frontal65-r5']);args=parser.parse_args()
    configure(run,'frontal65',1.25)
    grid=run.SCISSORS_SCENES['kidney'].build_grid(run.scissors_settings().density_kg_m3)
    result={}
    for case in args.case:
        folder=PREVIOUS/'outputs'/case
        result[case]={stage:audit(grid,folder/(stage+'.npz')) for stage in ('incise_1','incise_2','hold')}
        trajectory=[json.loads(line) for line in (folder/'trajectory.jsonl').read_text().splitlines()]
        top=sorted(trajectory,key=lambda row:np.linalg.norm(row['blade_force_n']),reverse=True)[:8]
        result[case]['force_peaks']=top
        print(case,json.dumps({k:{key:value for key,value in v.items() if key not in ('stray_edges','bridge')} for k,v in result[case].items() if k!='force_peaks'},indent=2),flush=True)
    output=ROOT/'results';output.mkdir(exist_ok=True)
    target=output/'saved-topology-audit.json'
    with target.open('x') as stream:json.dump(result,stream,indent=2)


if __name__=='__main__':main()
