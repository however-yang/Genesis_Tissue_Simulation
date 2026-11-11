"""Trace real connectivity from the upper wall to its fixed base, excluding only the intended uncut left attachment."""
import json,sys
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;R=W.parents[1];sys.path[:0]=[str(R),str(R/'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation')]
from inspect_wall_bridges import find_path
from endoscopic_tissue.scenario import SCISSORS_SCENES
for folder in map(Path,sys.argv[1:]):
 m=json.loads((folder/'manifest.json').read_text());s=SCISSORS_SCENES[m['scene']];g=s.build_grid(m['settings']['density_kg_m3']);p=g.positions;mat=g.inside!=0;right=p[:,0]>=s.cut_low[0]-1e-5;wall=mat&right&np.all((p>=np.array(s.tab.low)-1e-5)&(p<=np.array(s.tab.high)+1e-5),axis=1);top=wall&(p[:,1]>=p[wall,1].max()-1e-5);base=right&(g.pinned!=0)
 rows=[]
 for name in ['incise_1','incise_2','incise_3','incise_completion','hold','failure-state']:
  path=folder/(name+'.npz')
  if not path.exists():continue
  z=np.load(path)
  for kind,members in [('material',mat&right),('all_active',(g.active!=0)&right)]:
   target=(mat&right&(p[:,1]<=s.tab.low[1]-.2)) if kind=='material' else base
   route=find_path(z['neighbors'],members,top,target);rows.append(dict(stage=name,graph=kind,path=route,rest_mm=(10*p[route]).tolist(),deformed_mm=(10*z['nodes_cm'][route]).tolist()))
 out=dict(case=str(folder),scope='All nodes in rest halfspace x >= cut_low.x, from upper-wall material nodes to interior base material (material-only graph) or pinned base nodes (all-active graph). Excludes only the intended uncut left attachment; includes lower base and all skin, so a wound below y=8mm is not mistaken for a residual bridge.',stages=rows)
 (folder/'wall-to-base-audit.json').write_text(json.dumps(out,indent=2));print(folder.name,json.dumps([dict(stage=r['stage'],graph=r['graph'],path_nodes=len(r['path'])) for r in rows]))
