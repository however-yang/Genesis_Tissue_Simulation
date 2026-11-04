import json,sys
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;R=W.parents[1];sys.path[:0]=[str(R),str(R/'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation')]
from inspect_wall_bridges import find_path
from endoscopic_tissue.scenario import SCISSORS_SCENES
for folder in map(Path,sys.argv[1:]):
 m=json.loads((folder/'manifest.json').read_text());scene=SCISSORS_SCENES[m['scene']];grid=scene.build_grid(m['settings']['density_kg_m3']);p=grid.positions;lo=np.array(scene.tab.low);hi=np.array(scene.tab.high);lo[0]=scene.cut_low[0]
 material=(grid.inside!=0)&np.all((p>=lo-1e-5)&(p<=hi+1e-5),axis=1)&(p[:,1]>lo[1]+1e-5);top=material&(p[:,1]>=p[material,1].max()-1e-5);bottom=material&(p[:,1]<=p[material,1].min()+1e-5)
 low=lo-np.array(grid.cell_size);high=hi+np.array(grid.cell_size);low[0]=lo[0];low[1]=lo[1]
 members=(grid.active!=0)&np.all((p>=low-1e-5)&(p<=high+1e-5),axis=1)&(p[:,1]>low[1]+1e-5)
 rows=[]
 for name in ['incise_1','incise_2','incise_3','incise_completion','hold','failure-state']:
  path=folder/(name+'.npz')
  if not path.exists():continue
  d=np.load(path);route=find_path(d['neighbors'],members,top,bottom);rows.append(dict(stage=name,path=route,rest_mm=(p[route]*10).tolist(),material=(grid.inside[route]!=0).tolist(),deformed_mm=(d['nodes_cm'][route]*10).tolist()))
 out=dict(case=str(folder),scope='Within right-hand intended cut span, allowing all active support nodes one cell beyond the tab in width and top; excludes left attachment and underlying base. Seeds remain real material nodes.',stages=rows)
 (folder/'wall-skin-bridge-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
