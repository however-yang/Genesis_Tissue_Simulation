"""Check for an intact top-to-bottom path inside the intended right-hand wall cut span.

The left attachment and the base block are excluded, so they cannot cause this path.
This diagnostic does not alter the solver, mesh, or existing acceptance criteria.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from endoscopic_tissue.scenario import SCISSORS_SCENES


def find_path(neighbors,members,top,bottom):
    parent={int(i):-1 for i in np.flatnonzero(top&members)}
    queue=deque(parent)
    while queue:
        i=queue.popleft()
        if bottom[i]:
            path=[i]
            while parent[path[-1]]>=0:path.append(parent[path[-1]])
            return path[::-1]
        for j in neighbors[i]:
            j=int(j)
            if j>=0 and members[j] and j not in parent:
                parent[j]=i;queue.append(j)
    return []


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case_dir',type=Path)
    a=p.parse_args()
    m=json.loads((a.case_dir/'manifest.json').read_text(encoding='utf8'))
    if m['scene'] not in ('wall','wall-traction'):raise ValueError('Only upright wall fixtures are supported')
    scene=SCISSORS_SCENES[m['scene']]
    grid=scene.build_grid(m['settings']['density_kg_m3'])
    rest=grid.positions
    lo=np.array(scene.tab.low);hi=np.array(scene.tab.high)
    lo[0]=scene.cut_low[0]
    members=grid.inside.astype(bool)&np.all((rest>=lo-1e-5)&(rest<=hi+1e-5),axis=1)&(rest[:,1]>lo[1]+1e-5)
    min_y=rest[members,1].min();max_y=rest[members,1].max()
    top=members&(rest[:,1]>=max_y-1e-5)
    bottom=members&(rest[:,1]<=min_y+1e-5)
    assert find_path(grid.neighbors,members,top,bottom),'Control: uncut wall must connect across this region'
    rows=[]
    for stage in ('incise_1','incise_2','incise_3','hold','failure-state'):
        file=a.case_dir/(stage+'.npz')
        if not file.exists():continue
        d=np.load(file)
        path=find_path(d['neighbors'],members,top,bottom)
        rows.append(dict(stage=stage,top_to_bottom_path_remains=bool(path),path_node_ids=path,
            path_rest_mm=(rest[path]*10).tolist(),path_deformed_mm=(d['nodes_cm'][path]*10).tolist()))
    out=dict(region_rest_mm=[(lo*10).tolist(),(hi*10).tolist()],material_nodes=int(members.sum()),
        top_seed_count=int(top.sum()),bottom_seed_count=int(bottom.sum()),
        note='An intact neighbor path here is a remaining local material connection, not a rendering-only line. This right-hand region excludes the deliberately uncut left attachment and base block; it does not establish whether a particular retained link was physically reached by the blade.',stages=rows)
    (a.case_dir/'wall-bridge-audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps(out))
if __name__=='__main__':main()
