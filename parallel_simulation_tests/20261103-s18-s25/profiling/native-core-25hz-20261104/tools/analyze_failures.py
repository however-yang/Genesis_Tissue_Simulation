"""Offline CPU confirmation and localization; no topology thresholds are changed."""
import os,sys,json
from pathlib import Path
os.sched_setaffinity(0,[30,31]);os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
p=Path(__file__).resolve().parents[1];sys.path.insert(0,str(p/'python-gpu-check'))
import numpy as np
from endoscopic_tissue.training.snapshot import decode_python
from endoscopic_tissue.training.topology_fast import topology_statistics_fast
names=['query-v3-mode10-full-validation','query-v4-mode10-full-validation','material-v6-full-validation','final-repeat-0-tool-gpu-body'];out=[]
for name in names:
 d=p/'results'/name;r=json.loads((d/'result.json').read_text());data=decode_python((d/'failure-render-state.bin').read_bytes());surface=data['surface'];surface=surface if hasattr(surface,'vertices') else type('Surface',(),surface)
 cpu=topology_statistics_fast(surface);gpu=r['frames'][-1]['quality']['metrics']['surface_topology'];v=surface.vertices;tri=surface.triangles
 rounded=np.round(v.astype(np.float64)*100000.).astype(np.int64);_,inverse=np.unique(rounded,axis=0,return_inverse=True);ids=inverse[tri];edges={}
 for i,t in enumerate(ids):
  if len(set(map(int,t)))<3:continue
  for k in range(3):
   e=tuple(sorted((int(t[k]),int(t[(k+1)%3]))));edges.setdefault(e,[]).append(i)
 bad=[]
 for edge,incident in edges.items():
  if len(incident)<=2:continue
  endpoints=[]
  for index in edge:
   members=np.flatnonzero(inverse==index);points=v[members];spread=float(np.linalg.norm(points.astype(np.float64)-points[0],axis=1).max())
   endpoints.append(dict(point_ids=members.tolist(),positions_native_cm=points.tolist(),maximum_spread_micrometres=spread*10000.,aux=getattr(surface,'aux',np.empty((len(v),0)))[members].tolist()))
  bad.append(dict(incidence=len(incident),triangles=incident,endpoints=endpoints))
 keys=['positional_boundary_edges','positional_nonmanifold_edges','duplicate_triangles','degenerate_triangles','winding_consistent'];record=dict(run=name,frame_id=r['frames'][-1]['frame_id'],stage=r['frames'][-1]['stage'],cpu_topology=cpu,gpu_topology=gpu,selected_metrics_match=all(cpu[k]==gpu[k] for k in keys),nonmanifold_edges=bad,window_completed=r['completed'],approved=False)
 assert record['selected_metrics_match'];out.append(record)
(p/'results/failure-localization.json').write_text(json.dumps(out,indent=2));print(json.dumps([dict(run=x['run'],frame=x['frame_id'],matched=x['selected_metrics_match'],edges=[dict(incidence=e['incidence'],triangles=e['triangles'],spread_um=[v['maximum_spread_micrometres'] for v in e['endpoints']]) for e in x['nonmanifold_edges']]) for x in out],indent=2))
